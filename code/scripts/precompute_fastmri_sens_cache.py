import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_ROOT))


from data.fastmri_brain import (
    FastMRIBrainDataset,
    apply_cartesian_mask,
)

from operators.sense import (
    prepare_mask,
)

from operators.sensitivity_cache import (
    get_sensitivity_maps_cached,
)


TUNING_SEED = 20260904


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="extracted/multicoil_train",
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="AXT2",
    )

    parser.add_argument(
        "--num_volumes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=TUNING_SEED,
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--cache_root",
        type=str,
        default="code/.cache/espirit",
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
    )

    return parser.parse_args()


def get_raw_samples(dataset):
    if hasattr(dataset, "raw_samples"):
        return dataset.raw_samples

    if hasattr(dataset, "examples"):
        return dataset.examples

    raise RuntimeError(
        "Cannot find raw_samples/examples."
    )


def get_fname(raw_sample):
    if hasattr(raw_sample, "fname"):
        return Path(raw_sample.fname).name

    if isinstance(raw_sample, tuple):
        return Path(raw_sample[0]).name

    raise RuntimeError(
        f"Cannot parse raw sample: {raw_sample}"
    )


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError(
            "--num_shards must be >= 1"
        )

    if not (
        0 <= args.shard_id < args.num_shards
    ):
        raise ValueError(
            "Require 0 <= shard_id < num_shards"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = FastMRIBrainDataset(
        root=args.data_root,
        acquisition=args.acquisition,
        cache_dir="code/.cache",
    )

    raw_samples = get_raw_samples(
        dataset
    )

    volume_indices = defaultdict(list)

    for idx, raw_sample in enumerate(
        raw_samples
    ):
        fname = get_fname(
            raw_sample
        )

        volume_indices[
            fname
        ].append(idx)

    all_fnames = sorted(
        volume_indices.keys()
    )

    rng = random.Random(
        args.seed
    )

    selected = rng.sample(
        all_fnames,
        min(
            args.num_volumes,
            len(all_fnames),
        ),
    )

    selected = sorted(
        selected
    )

    full_selected = list(
        selected
    )

    selected = selected[
        args.shard_id
        ::args.num_shards
    ]

    print("=" * 80)
    print("FASTMRI ESPIRIT CACHE PRECOMPUTE")
    print("=" * 80)
    print("device:", device)
    print("acceleration:", args.acceleration)
    print("seed:", args.seed)
    print("full cohort:", len(full_selected))
    print(
        "shard:",
        f"{args.shard_id}/{args.num_shards}",
    )
    print(
        "volumes in shard:",
        len(selected),
    )

    total_hits = 0
    total_misses = 0
    total_slices = 0

    start_all = time.time()

    for volume_i, fname in enumerate(
        selected,
        start=1,
    ):
        print(
            f"\n[{volume_i}/{len(selected)}] "
            f"{fname}"
        )

        volume_hits = 0
        volume_misses = 0

        for idx in volume_indices[
            fname
        ]:
            sample = dataset[idx]

            (
                masked_kspace,
                raw_mask,
                num_low,
            ) = apply_cartesian_mask(
                sample,
                args.acceleration,
            )

            masked_kspace = (
                masked_kspace.to(
                    device
                )
            )

            mask = prepare_mask(
                raw_mask,
                width=
                    masked_kspace.shape[-1],
                device=device,
            )

            _, hit, cache_path = (
                get_sensitivity_maps_cached(
                    cache_root=
                        args.cache_root,

                    dataset_tag=
                        Path(
                            args.data_root
                        ).name,

                    acquisition=
                        args.acquisition,

                    acceleration=
                        args.acceleration,

                    fname=
                        sample["fname"],

                    slice_num=
                        sample["slice_num"],

                    masked_kspace=
                        masked_kspace,

                    mask=
                        mask,

                    num_low_frequencies=
                        num_low,

                    crop=
                        0.0,

                    device=
                        device,

                    recompute=
                        args.recompute,
                )
            )

            if hit:
                volume_hits += 1
                total_hits += 1
            else:
                volume_misses += 1
                total_misses += 1

            total_slices += 1

            print(
                f"  slice "
                f"{sample['slice_num']:02d} | "
                f"{'HIT' if hit else 'MISS'} | "
                f"{cache_path}"
            )

        print(
            "  volume cache:",
            f"{volume_hits} hits / "
            f"{volume_misses} misses",
        )

    runtime = (
        time.time()
        - start_all
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("slices:", total_slices)
    print("hits:", total_hits)
    print("misses:", total_misses)
    print("runtime sec:", runtime)

    assert (
        total_hits
        + total_misses
        == total_slices
    )

    print(
        "ESPIRIT CACHE PRECOMPUTE PASSED"
    )


if __name__ == "__main__":
    main()
