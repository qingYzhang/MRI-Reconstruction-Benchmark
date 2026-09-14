import argparse
import csv
import gc
import hashlib
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fastmri.data.transforms import center_crop


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_ROOT))


from data.fastmri_brain import (
    FastMRIBrainDataset,
    apply_cartesian_mask,
    zero_filled_rss,
)

from evaluation.metrics import (
    compute_volume_metrics,
)

from operators.sense import (
    prepare_mask,
    cg_sense,
)

from operators.sensitivity_cache import (
    get_sensitivity_maps_cached,
)

from methods.lacs import (
    reconstruct_lacs,
)

from methods.nerp import (
    reconstruct_nerp,
)


CG_LAMBDA = {
    4: 0.01,
    8: 0.01,
}

CG_ITERS = 50
ESPIRIT_CROP = 0.0
LACS_REQUESTED_WAVELET_LEVELS = 2

TUNING_SEED = 20260904
TUNING_VOLUMES = 30

CANONICAL_METHOD_ORDER = [
    "zf",
    "cg",
    "lacs",
    "nerp",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        choices=[
            "smoke",
            "tuning30",
            "full",
        ],
        required=True,
    )

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=CANONICAL_METHOD_ORDER,
        default=CANONICAL_METHOD_ORDER,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="AXT2",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=TUNING_SEED,
    )

    parser.add_argument(
        "--num_tuning_volumes",
        type=int,
        default=TUNING_VOLUMES,
    )

    parser.add_argument(
        "--nerp_iters",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--nerp_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--max_slices_per_volume",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--verbose_nerp",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
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
        "--sens_cache_root",
        type=str,
        default="code/.cache/espirit",
    )

    parser.add_argument(
        "--recompute_sens_cache",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    return parser.parse_args()

def max_native_dyadic_levels(
    height,
    width,
):
    """
    Maximum number of dyadic 2-D wavelet levels
    supported by the native reconstruction grid.

    Both spatial dimensions must remain even at
    every decomposition level.
    """

    height = int(height)
    width = int(width)

    levels = 0

    while (
        height > 0
        and width > 0
        and height % 2 == 0
        and width % 2 == 0
    ):
        height //= 2
        width //= 2
        levels += 1

    return levels


def get_lacs_wavelet_levels(
    measured_kspace,
):
    height = int(
        measured_kspace.shape[-2]
    )

    width = int(
        measured_kspace.shape[-1]
    )

    max_levels = (
        max_native_dyadic_levels(
            height,
            width,
        )
    )

    if max_levels < 1:
        raise RuntimeError(
            "Native reconstruction grid "
            f"{height}x{width} does not support "
            "even one dyadic wavelet level."
        )

    effective_levels = min(
        LACS_REQUESTED_WAVELET_LEVELS,
        max_levels,
    )

    return (
        effective_levels,
        max_levels,
    )

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


def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_start(device):

    synchronize(device)

    return time.perf_counter()


def timed_end(device, start):

    synchronize(device)

    return time.perf_counter() - start


def mask_hash(mask):

    array = (
        mask
        .detach()
        .cpu()
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()[:16]


def resolve_data_root(args):

    if args.data_root is not None:
        return args.data_root

    if args.subset in [
        "smoke",
        "tuning30",
    ]:
        return "extracted/multicoil_train"

    return "extracted/multicoil_val"


def build_volume_indices(dataset):

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

    return volume_indices


def select_cohort(
    all_fnames,
    subset,
    seed,
    num_tuning_volumes,
):

    all_fnames = sorted(
        all_fnames
    )

    if subset == "full":
        return all_fnames

    rng = random.Random(
        seed
    )

    frozen = rng.sample(
        all_fnames,
        min(
            num_tuning_volumes,
            len(all_fnames),
        ),
    )

    frozen = sorted(
        frozen
    )

    if subset == "tuning30":
        return frozen

    if subset == "smoke":
        #
        # Smoke is intentionally chosen from
        # the frozen tuning cohort.
        #
        return [frozen[0]]

    raise ValueError(
        f"Unknown subset: {subset}"
    )


def select_slice_indices(
    indices,
    max_slices,
):

    indices = list(indices)

    if (
        max_slices is None
        or max_slices >= len(indices)
    ):
        return indices

    if max_slices <= 0:
        raise ValueError(
            "max_slices_per_volume must be > 0"
        )

    #
    # Use central contiguous slices.
    #
    center = len(indices) // 2

    start = (
        center
        - max_slices // 2
    )

    start = max(
        0,
        start,
    )

    end = start + max_slices

    if end > len(indices):
        end = len(indices)
        start = end - max_slices

    return indices[
        start:end
    ]


def normalize_methods(methods):

    requested = set(methods)

    return [
        method
        for method in CANONICAL_METHOD_ORDER
        if method in requested
    ]


def load_existing_results(
    output,
    resume,
):

    if (
        not resume
        or not output.exists()
    ):
        return []

    with output.open(
        "r",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


FIELDNAMES = [
    "subset",
    "acceleration",
    "fname",
    "method",
    "slices_evaluated",
    "nmse",
    "psnr",
    "ssim",
    "method_runtime_sec",
    "shared_sens_runtime_sec",
    "sampling_fraction",
    "num_low_frequencies",
    "mask_sha256",
    "lacs_wavelet_levels",
    "nerp_iters",
]


def save_results(
    output,
    results,
):

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def save_cohort(
    output,
    selected_fnames,
):

    cohort_path = (
        output.parent
        / f"{output.stem}_cohort.txt"
    )

    with cohort_path.open(
        "w"
    ) as f:

        for fname in selected_fnames:
            f.write(
                fname + "\n"
            )

    return cohort_path


def run():

    args = parse_args()

    methods = normalize_methods(
        args.methods
    )

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    data_root = resolve_data_root(
        args
    )

    #
    # Smoke defaults to exactly one central slice.
    #
    if (
        args.subset == "smoke"
        and args.max_slices_per_volume is None
    ):
        max_slices_per_volume = 1

    else:
        max_slices_per_volume = (
            args.max_slices_per_volume
        )

    output = Path(
        args.output
    )

    print("=" * 100)
    print("UNIFIED FASTMRI PRIOR-FREE BENCHMARK")
    print("=" * 100)

    print("subset:", args.subset)
    print("data root:", data_root)
    print("acquisition:", args.acquisition)
    print("acceleration:", args.acceleration)
    print("methods:", methods)
    print("device:", device)

    print(
        "max slices / volume:",
        max_slices_per_volume,
    )

    print(
        "ESPIRiT crop:",
        ESPIRIT_CROP,
    )

    print(
        "CG lambda:",
        CG_LAMBDA[
            args.acceleration
        ],
    )

    print(
        "CG iterations:",
        CG_ITERS,
    )

    print(
        "NeRP iterations:",
        args.nerp_iters,
    )

    print(
        "NeRP learning rate:",
        args.nerp_lr,
    )

    print(
        "tuning seed:",
        args.seed,
    )

    dataset = FastMRIBrainDataset(
        root=data_root,
        acquisition=args.acquisition,
        cache_dir="code/.cache",
    )

    volume_indices = (
        build_volume_indices(
            dataset
        )
    )

    selected_fnames = (
        select_cohort(
            all_fnames=
                volume_indices.keys(),

            subset=
                args.subset,

            seed=
                args.seed,

            num_tuning_volumes=
                args.num_tuning_volumes,
        )
    )
    
    if args.num_shards < 1:
        raise ValueError(
            "--num_shards must be >= 1"
        )

    if not (
        0
        <= args.shard_id
        < args.num_shards
    ):
        raise ValueError(
            "--shard_id must satisfy "
            "0 <= shard_id < num_shards"
        )

    full_selected_fnames = list(
        selected_fnames
    )

    selected_fnames = (
        full_selected_fnames[
            args.shard_id
            ::args.num_shards
        ]
    )

    if not selected_fnames:
        raise RuntimeError(
            "This shard contains no volumes."
        )

    print(
        "\nShard:",
        f"{args.shard_id}/"
        f"{args.num_shards}",
    )

    print(
        "Full cohort size:",
        len(full_selected_fnames),
    )

    print(
        "Volumes in this shard:",
        len(selected_fnames),
    )

    print(
        "\nVolumes selected:",
        len(selected_fnames),
    )

    for fname in selected_fnames:
        print(" ", fname)

    cohort_path = save_cohort(
        output,
        selected_fnames,
    )

    print(
        "\nCohort saved:",
        cohort_path,
    )

    results = load_existing_results(
        output,
        args.resume,
    )

    completed = {
        (
            row["fname"],
            row["method"],
        )
        for row in results
        if (
            str(row["subset"])
            == str(args.subset)
            and int(row["acceleration"])
            == args.acceleration
        )
    }

    if results:
        print(
            "Existing rows loaded:",
            len(results),
        )

    for volume_number, fname in enumerate(
        selected_fnames,
        start=1,
    ):

        pending_methods = [
            method
            for method in methods
            if (
                fname,
                method,
            ) not in completed
        ]

        if not pending_methods:

            print(
                f"\n[{volume_number}/"
                f"{len(selected_fnames)}] "
                f"{fname}: already complete"
            )

            continue

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"[{volume_number}/"
            f"{len(selected_fnames)}] "
            f"{fname}"
        )

        print(
            "pending:",
            pending_methods,
        )

        print(
            "=" * 100
        )

        indices = select_slice_indices(
            volume_indices[
                fname
            ],
            max_slices_per_volume,
        )

        print(
            "slices evaluated:",
            len(indices),
        )

        target_slices = []

        predictions = {
            method: []
            for method in pending_methods
        }

        method_runtime = {
            method: 0.0
            for method in pending_methods
        }

        shared_sens_runtime = 0.0
        sens_cache_hits = 0
        sens_cache_misses = 0

        first_mask = None
        first_mask_hash = None
        first_num_low = None
        first_sampling_fraction = None
        first_lacs_wavelet_levels = None

        max_value = None

        need_sens = any(
            method in {
                "cg",
                "lacs",
                "nerp",
            }
            for method in pending_methods
        )

        for slice_position, idx in enumerate(
            indices,
            start=1,
        ):

            sample = dataset[idx]

            target = (
                sample["target"]
                .to(device)
            )

            (
                masked_kspace,
                raw_mask,
                num_low,
            ) = apply_cartesian_mask(
                sample,
                args.acceleration,
            )

            masked_kspace = (
                masked_kspace
                .to(device)
            )

            mask = prepare_mask(
                raw_mask,
                width=
                    masked_kspace.shape[-1],
                device=device,
            )

            current_mask_cpu = (
                mask
                .detach()
                .cpu()
            )

            if first_mask is None:

                first_mask = (
                    current_mask_cpu.clone()
                )

                first_mask_hash = (
                    mask_hash(mask)
                )

                first_num_low = int(
                    num_low
                )

                first_sampling_fraction = float(
                    mask.float()
                    .mean()
                    .item()
                )

            else:

                if not torch.equal(
                    current_mask_cpu,
                    first_mask,
                ):
                    raise RuntimeError(
                        f"Mask changed within volume "
                        f"{fname}."
                    )

                if int(num_low) != first_num_low:
                    raise RuntimeError(
                        f"ACS size changed within "
                        f"volume {fname}."
                    )

            print(
                f"  slice "
                f"{slice_position}/"
                f"{len(indices)} | "
                f"dataset idx={idx} | "
                f"slice_num="
                f"{sample['slice_num']}"
            )

            sens_maps = None

            if need_sens:

                start = timed_start(
                    device
                )

                (
                    sens_maps,
                    sens_cache_hit,
                    sens_cache_path,
                ) = (
                    get_sensitivity_maps_cached(
                        cache_root=
                            args.sens_cache_root,

                        dataset_tag=
                            Path(
                                data_root
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
                            ESPIRIT_CROP,

                        device=
                            device,

                        recompute=
                            args.recompute_sens_cache,
                    )
                )

                shared_sens_runtime += (
                    timed_end(
                        device,
                        start,
                    )
                )

                if sens_cache_hit:
                    sens_cache_hits += 1
                else:
                    sens_cache_misses += 1

                if not torch.isfinite(
                    sens_maps
                ).all():

                    raise RuntimeError(
                        "Non-finite ESPIRiT "
                        "sensitivity maps."
                    )
            #
            # ----------------------------------------
            # Zero-filled
            # ----------------------------------------
            #
            if "zf" in pending_methods:

                start = timed_start(
                    device
                )

                reconstruction = (
                    zero_filled_rss(
                        masked_kspace,
                        target.shape[-2:],
                    )
                )

                method_runtime[
                    "zf"
                ] += timed_end(
                    device,
                    start,
                )

                predictions[
                    "zf"
                ].append(
                    reconstruction
                    .detach()
                    .cpu()
                    .numpy()
                )

            #
            # ----------------------------------------
            # CG-SENSE
            # ----------------------------------------
            #
            if "cg" in pending_methods:

                start = timed_start(
                    device
                )

                reconstruction = cg_sense(
                    measured_kspace=
                        masked_kspace,

                    sens_maps=
                        sens_maps,

                    mask=
                        mask,

                    num_iters=
                        CG_ITERS,

                    lambda_reg=
                        CG_LAMBDA[
                            args.acceleration
                        ],
                )

                reconstruction = (
                    torch.abs(
                        reconstruction
                    )
                )

                reconstruction = (
                    center_crop(
                        reconstruction,
                        target.shape[-2:],
                    )
                )

                method_runtime[
                    "cg"
                ] += timed_end(
                    device,
                    start,
                )

                predictions[
                    "cg"
                ].append(
                    reconstruction
                    .detach()
                    .cpu()
                    .numpy()
                )

                del reconstruction

            #
            # ----------------------------------------
            # LACS no prior
            #
            # IMPORTANT:
            # unified source-faithful Wavelab/FISTA
            # path, NOT old ADMM.
            # ----------------------------------------
            #
            if "lacs" in pending_methods:

                start = timed_start(
                    device
                )

                (
                    lacs_wavelet_levels,
                    lacs_max_wavelet_levels,
                ) = get_lacs_wavelet_levels(
                    masked_kspace
                )

                if (
                    first_lacs_wavelet_levels
                    is None
                ):
                    first_lacs_wavelet_levels = (
                        lacs_wavelet_levels
                    )

                    print(
                        "    LACS wavelet levels:",
                        lacs_wavelet_levels,
                        "| requested:",
                        LACS_REQUESTED_WAVELET_LEVELS,
                        "| max supported:",
                        lacs_max_wavelet_levels,
                        "| native shape:",
                        tuple(
                            masked_kspace.shape[
                                -2:
                            ]
                        ),
                    )

                elif (
                    lacs_wavelet_levels
                    != first_lacs_wavelet_levels
                ):
                    raise RuntimeError(
                        "LACS wavelet level "
                        "changed within volume "
                        f"{fname}."
                    )

                lacs_result = (
                    reconstruct_lacs(
                        measured_kspace=
                            masked_kspace,

                        sens_maps=
                            sens_maps,

                        mask=
                            mask,

                        prior=None,

                        wavelet_levels=
                            lacs_wavelet_levels,

                        verbose=False,
                    )
                )

                if (
                    lacs_result["mode"]
                    != "no_prior"
                ):
                    raise RuntimeError(
                        "LACS did not enter "
                        "no-prior mode."
                    )

                reconstruction = (
                    torch.abs(
                        lacs_result[
                            "reconstruction"
                        ]
                    )
                )

                reconstruction = (
                    center_crop(
                        reconstruction,
                        target.shape[-2:],
                    )
                )

                method_runtime[
                    "lacs"
                ] += timed_end(
                    device,
                    start,
                )

                predictions[
                    "lacs"
                ].append(
                    reconstruction
                    .detach()
                    .cpu()
                    .numpy()
                )

                del reconstruction
                del lacs_result

            #
            # ----------------------------------------
            # NeRP w/o prior
            # ----------------------------------------
            #
            if "nerp" in pending_methods:

                start = timed_start(
                    device
                )

                nerp_result = (
                    reconstruct_nerp(
                        measured_kspace=
                            masked_kspace,

                        sens_maps=
                            sens_maps,

                        mask=
                            mask,

                        prior=None,

                        reconstruction_iters=
                            args.nerp_iters,

                        reconstruction_learning_rate=
                            args.nerp_lr,

                        embedding_size=256,
                        fourier_scale=3.0,
                        network_depth=8,
                        network_width=512,
                        w0=30.0,

                        seed=0,

                        verbose=
                            args.verbose_nerp,
                    )
                )

                if (
                    nerp_result["mode"]
                    != "no_prior"
                ):
                    raise RuntimeError(
                        "NeRP did not enter "
                        "no-prior mode."
                    )

                reconstruction = (
                    torch.abs(
                        nerp_result[
                            "reconstruction"
                        ]
                    )
                )

                reconstruction = (
                    center_crop(
                        reconstruction,
                        target.shape[-2:],
                    )
                )

                method_runtime[
                    "nerp"
                ] += timed_end(
                    device,
                    start,
                )

                predictions[
                    "nerp"
                ].append(
                    reconstruction
                    .detach()
                    .cpu()
                    .numpy()
                )

                del reconstruction
                del nerp_result

            target_slices.append(
                target
                .detach()
                .cpu()
                .numpy()
            )

            if max_value is None:

                max_value = float(
                    sample[
                        "max_value"
                    ]
                )

            if sens_maps is not None:
                del sens_maps

            del masked_kspace
            del mask
            del target

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        #
        # --------------------------------------------
        # Volume / selected-slice metrics
        # --------------------------------------------
        #

        target_volume = np.stack(
            target_slices,
            axis=0,
        )

        print(
            "\nVolume metrics:"
        )

        new_rows = []

        for method in pending_methods:

            prediction_volume = np.stack(
                predictions[
                    method
                ],
                axis=0,
            )

            metrics = (
                compute_volume_metrics(
                    target=
                        target_volume,

                    prediction=
                        prediction_volume,

                    max_value=
                        max_value,
                )
            )

            row = {
                "subset":
                    args.subset,

                "acceleration":
                    args.acceleration,

                "fname":
                    fname,

                "method":
                    method,

                "slices_evaluated":
                    len(indices),

                "nmse":
                    metrics["nmse"],

                "psnr":
                    metrics["psnr"],

                "ssim":
                    metrics["ssim"],

                "method_runtime_sec":
                    method_runtime[
                        method
                    ],

                "shared_sens_runtime_sec":
                    shared_sens_runtime,

                "sampling_fraction":
                    first_sampling_fraction,

                "num_low_frequencies":
                    first_num_low,

                "mask_sha256":
                    first_mask_hash,

                "lacs_wavelet_levels":
                    (
                        first_lacs_wavelet_levels
                        if method == "lacs"
                        else ""
                    ),

                "nerp_iters":
                    (
                        args.nerp_iters
                        if method == "nerp"
                        else ""
                    ),
            }

            new_rows.append(
                row
            )

            print(
                f"  {method:6s} | "
                f"NMSE="
                f"{metrics['nmse']:.6f} | "
                f"PSNR="
                f"{metrics['psnr']:.3f} | "
                f"SSIM="
                f"{metrics['ssim']:.4f} | "
                f"runtime="
                f"{method_runtime[method]:.2f}s"
            )

        results.extend(
            new_rows
        )

        for row in new_rows:
            completed.add(
                (
                    row["fname"],
                    row["method"],
                )
            )

        #
        # Incremental checkpoint.
        #
        save_results(
            output,
            results,
        )

        print(
            "sampling fraction:",
            first_sampling_fraction,
        )

        print(
            "ACS lines:",
            first_num_low,
        )

        print(
            "mask sha256:",
            first_mask_hash,
        )

        print(
            "shared ESPIRiT runtime:",
            shared_sens_runtime,
        )
        
        print(
            "sensitivity cache:",
            f"{sens_cache_hits} hits / "
            f"{sens_cache_misses} misses",
        )

        print(
            "checkpoint saved:",
            output,
        )

    #
    # ------------------------------------------------
    # Final summary.
    # ------------------------------------------------
    #

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 100
    )

    current_rows = [
        row
        for row in results
        if (
            str(row["subset"])
            == str(args.subset)
            and int(
                row["acceleration"]
            )
            == args.acceleration
            and row["method"]
            in methods
        )
    ]

    for method in methods:

        rows = [
            row
            for row in current_rows
            if row["method"] == method
        ]

        if not rows:
            continue

        print(
            f"\n{method.upper()} "
            f"({len(rows)} volumes)"
        )

        for metric_name in [
            "nmse",
            "psnr",
            "ssim",
        ]:

            values = np.asarray(
                [
                    float(
                        row[
                            metric_name
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )

            print(
                f"  "
                f"{metric_name.upper():5s}: "
                f"{values.mean():.6f} "
                f"+/- "
                f"{values.std():.6f}"
            )

    print(
        "\nSaved:",
        output,
    )

    print(
        "Cohort:",
        cohort_path,
    )


if __name__ == "__main__":
    run()
