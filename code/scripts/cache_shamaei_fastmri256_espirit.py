import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CODE_ROOT = (
    PROJECT_ROOT
    / "code"
)

sys.path.insert(
    0,
    str(CODE_ROOT),
)


from data.fastmri_brain import (
    FastMRIBrainDataset,
    zero_filled_rss,
)

from data.fastmri_256 import (
    FASTMRI_256_SIZE,
    build_fastmri_256_problem,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)


ESPIRIT_CROP = 0.0

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_training_manifest.json"
)

DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "shamaei_fastmri256"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
        ],
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(
            DEFAULT_CACHE_ROOT
        ),
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
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


def cache_path(
    cache_root,
    acceleration,
    split,
    fname,
    slice_num,
):

    stem = Path(
        fname
    ).stem

    return (
        Path(cache_root)
        / f"r{acceleration}"
        / split
        / (
            f"{stem}"
            f"_slice{int(slice_num):03d}.pt"
        )
    )


def validate_existing(
    path,
    acceleration,
    fname,
    slice_num,
):

    try:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        assert (
            int(
                data[
                    "acceleration"
                ]
            )
            == acceleration
        )

        assert (
            data[
                "fname"
            ]
            == fname
        )

        assert (
            int(
                data[
                    "slice_num"
                ]
            )
            == int(
                slice_num
            )
        )

        sens_ri = data[
            "sens_ri_fp16"
        ]

        assert (
            sens_ri.dtype
            == torch.float16
        )

        assert (
            sens_ri.shape[-1]
            == 2
        )

        assert torch.isfinite(
            sens_ri
        ).all()

        assert (
            float(
                data[
                    "measurement_scale"
                ]
            )
            > 0.0
        )

        return True

    except Exception:
        return False


def atomic_save(
    obj,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def sync():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():

    args = parse_args()

    if not (
        0
        <= args.shard_id
        < args.num_shards
    ):
        raise ValueError(
            "Invalid shard."
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested "
            "but unavailable."
        )

    with open(
        args.manifest
    ) as f:

        manifest = json.load(
            f
        )

    entries = list(
        manifest[
            args.split
        ][
            "samples"
        ]
    )

    entries = (
        entries[
            args.shard_id
            :: args.num_shards
        ]
    )

    if (
        args.limit
        is not None
    ):
        entries = entries[
            :args.limit
        ]

    if (
        args.split
        == "train"
    ):
        data_root = (
            "extracted/"
            "multicoil_train"
        )
    else:
        data_root = (
            "extracted/"
            "multicoil_val"
        )

    dataset = (
        FastMRIBrainDataset(
            root=data_root,
            acquisition="AXT2",
            cache_dir="code/.cache",
        )
    )

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "ESPIRIT CACHE"
    )

    print(
        "=" * 90
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "split:",
        args.split,
    )

    print(
        "device:",
        device,
    )

    print(
        "num shards:",
        args.num_shards,
    )

    print(
        "shard id:",
        args.shard_id,
    )

    print(
        "samples in shard:",
        len(entries),
    )

    print(
        "ESPIRiT crop:",
        ESPIRIT_CROP,
    )

    print()

    completed = 0
    skipped = 0
    actual_bytes = 0

    total_start = time.time()

    for position, entry in enumerate(
        entries,
        start=1,
    ):

        idx = int(
            entry[
                "dataset_index"
            ]
        )

        fname = entry[
            "fname"
        ]

        slice_num = int(
            entry[
                "slice_num"
            ]
        )

        out = cache_path(
            cache_root=
                args.cache_root,

            acceleration=
                args.acceleration,

            split=
                args.split,

            fname=
                fname,

            slice_num=
                slice_num,
        )

        if (
            args.resume
            and out.exists()
            and validate_existing(
                out,
                args.acceleration,
                fname,
                slice_num,
            )
        ):

            skipped += 1
            actual_bytes += (
                out.stat()
                .st_size
            )

            print(
                f"[{position:4d}/"
                f"{len(entries):4d}] "
                f"SKIP "
                f"{fname} "
                f"slice={slice_num}"
            )

            continue

        start = time.time()

        sample = dataset[
            idx
        ]

        if (
            sample[
                "fname"
            ]
            != fname
        ):
            raise RuntimeError(
                "fname mismatch: "
                f"{sample['fname']} "
                f"!= {fname}"
            )

        if (
            int(
                sample[
                    "slice_num"
                ]
            )
            != slice_num
        ):
            raise RuntimeError(
                "slice mismatch."
            )

        (
            sample_256,
            masked_kspace,
            raw_mask,
            num_low,
        ) = (
            build_fastmri_256_problem(
                sample,
                args.acceleration,
            )
        )

        masked_kspace = (
            masked_kspace
            .to(
                device,
                dtype=
                    torch.complex64,
            )
        )

        sync()

        espirit_start = (
            time.time()
        )

        sens_maps = (
            estimate_sens_espirit(
                masked_kspace,
                num_low_frequencies=
                    num_low,
                crop=
                    ESPIRIT_CROP,
            )
            .to(
                device,
                dtype=
                    torch.complex64,
            )
        )

        sync()

        espirit_sec = (
            time.time()
            - espirit_start
        )

        if not torch.isfinite(
            sens_maps
        ).all():
            raise RuntimeError(
                "Non-finite "
                "sensitivity maps."
            )

        #
        # Same measurement-derived
        # normalization used by our
        # successful overfit gate.
        #
        with torch.no_grad():

            zf = (
                zero_filled_rss(
                    masked_kspace,
                    (
                        FASTMRI_256_SIZE,
                        FASTMRI_256_SIZE,
                    ),
                )
                .to(
                    device,
                    dtype=
                        torch.float32,
                )
            )

            measurement_scale = (
                torch.quantile(
                    zf.flatten(),
                    0.999,
                )
                .clamp_min(
                    1e-8
                )
            )

        #
        # Store real/imag as FP16.
        #
        sens_ri_fp16 = (
            torch.view_as_real(
                sens_maps
            )
            .to(
                torch.float16
            )
            .cpu()
            .contiguous()
        )

        #
        # Quantization audit.
        #
        sens_roundtrip = (
            torch.view_as_complex(
                sens_ri_fp16
                .float()
                .contiguous()
            )
            .to(
                device
            )
        )

        quant_rel_error = (
            torch.linalg.vector_norm(
                sens_roundtrip
                - sens_maps
            )
            /
            torch.linalg.vector_norm(
                sens_maps
            )
            .clamp_min(
                1e-12
            )
        )

        quant_rel_error = float(
            quant_rel_error.item()
        )

        if (
            quant_rel_error
            > 2e-3
        ):
            raise RuntimeError(
                "FP16 sensitivity "
                "quantization too large: "
                f"{quant_rel_error}"
            )

        mask_1d = (
            raw_mask
            .squeeze()
            .float()
        )

        sampling_fraction = float(
            mask_1d.mean()
            .item()
        )

        payload = {
            "version":
                1,

            "fname":
                fname,

            "slice_num":
                slice_num,

            "dataset_index":
                idx,

            "split":
                args.split,

            "acceleration":
                args.acceleration,

            "grid":
                "256x256",

            "adaptation":
                "image_crop_256",

            "espirit_crop":
                ESPIRIT_CROP,

            "num_low_frequencies":
                int(
                    num_low
                ),

            "sampling_fraction":
                sampling_fraction,

            "measurement_scale":
                float(
                    measurement_scale
                    .item()
                ),

            "sensitivity_storage":
                (
                    "real_imag_fp16"
                ),

            "sensitivity_shape_complex":
                list(
                    sens_maps.shape
                ),

            "sens_ri_fp16":
                sens_ri_fp16,

            "quantization_relative_error":
                quant_rel_error,
        }

        atomic_save(
            payload,
            out,
        )

        size_bytes = (
            out.stat()
            .st_size
        )

        actual_bytes += (
            size_bytes
        )

        completed += 1

        elapsed = (
            time.time()
            - start
        )

        print(
            f"[{position:4d}/"
            f"{len(entries):4d}] "
            f"{fname} "
            f"slice={slice_num:02d} "
            f"| coils="
            f"{sens_maps.shape[0]} "
            f"| ACS={int(num_low)} "
            f"| ESPIRiT="
            f"{espirit_sec:.2f}s "
            f"| qerr="
            f"{quant_rel_error:.3e} "
            f"| file="
            f"{size_bytes / 2**20:.2f} MiB "
            f"| total="
            f"{elapsed:.2f}s"
        )

        del (
            sens_maps,
            sens_roundtrip,
            sens_ri_fp16,
            zf,
            masked_kspace,
        )

        if (
            position % 25
            == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    total_sec = (
        time.time()
        - total_start
    )

    print()
    print(
        "=" * 90
    )

    print(
        "CACHE SHARD COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "computed:",
        completed,
    )

    print(
        "skipped:",
        skipped,
    )

    print(
        "samples total:",
        len(entries),
    )

    print(
        "cache size in shard:",
        f"{actual_bytes / 2**30:.3f} GiB",
    )

    print(
        "runtime:",
        f"{total_sec / 3600:.3f} h",
    )


if __name__ == "__main__":
    main()
