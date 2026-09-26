import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CODE_ROOT = PROJECT_ROOT / "code"

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

from operators.sense import (
    prepare_mask,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)

from evaluation.metrics import (
    compute_volume_metrics,
)

from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
)


ESPIRIT_CROP = 0.0


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        default=(
            "outputs/"
            "shamaei_fastmri256_split.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
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
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--limit_volumes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


def make_mask(
    raw_mask,
    device,
):

    mask = prepare_mask(
        raw_mask,
        width=FASTMRI_256_SIZE,
        device=device,
    )

    mask = (
        mask
        .squeeze()
        .bool()
    )

    if mask.ndim == 1:

        if (
            mask.numel()
            != FASTMRI_256_SIZE
        ):
            raise RuntimeError(
                f"Unexpected mask: "
                f"{tuple(mask.shape)}"
            )

        mask = (
            mask[
                None,
                :
            ]
            .expand(
                FASTMRI_256_SIZE,
                FASTMRI_256_SIZE,
            )
        )

    if tuple(
        mask.shape
    ) != (
        FASTMRI_256_SIZE,
        FASTMRI_256_SIZE,
    ):
        raise RuntimeError(
            "Unexpected mask shape: "
            f"{tuple(mask.shape)}"
        )

    return (
        mask[
            None,
            None,
            ...
        ]
    )


def acquired_scale_alpha(
    predicted_kspace,
    measured_kspace,
    mask,
):

    if not torch.is_complex(
        predicted_kspace
    ):
        raise RuntimeError(
            "predicted_kspace "
            "must be complex."
        )

    acquired = (
        mask.expand_as(
            predicted_kspace
        )
    )

    kp = (
        predicted_kspace[
            acquired
        ]
    )

    y = (
        measured_kspace[
            acquired
        ]
    )

    denominator = (
        (
            kp.conj()
            * kp
        )
        .real
        .sum()
        .clamp_min(
            1e-12
        )
    )

    numerator = (
        (
            kp.conj()
            * y
        )
        .sum()
    )

    return (
        numerator
        / denominator
    )


def load_existing(
    path,
):

    completed = set()

    if not path.exists():
        return completed

    with path.open() as f:

        for row in csv.DictReader(
            f
        ):

            completed.add(
                row[
                    "fname"
                ]
            )

    return completed


def append_row(
    path,
    row,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        path.exists()
    )

    with path.open(
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


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

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    ckpt_r = int(
        checkpoint[
            "adaptation"
        ][
            "acceleration"
        ]
    )

    if (
        ckpt_r
        != args.acceleration
    ):
        raise RuntimeError(
            "Checkpoint acceleration "
            f"R{ckpt_r} != "
            f"requested R{args.acceleration}"
        )

    architecture = (
        checkpoint[
            "architecture"
        ]
    )

    model = (
        ShamaeiE2EVarNetFastMRI(
            num_layers=
                int(
                    architecture[
                        "num_layers"
                    ]
                ),

            regularizer_num_filters=
                int(
                    architecture[
                        "regularizer_num_filters"
                    ]
                ),

            regularizer_num_pull_layers=
                int(
                    architecture[
                        "regularizer_num_pull_layers"
                    ]
                ),

            regularizer_dropout=
                float(
                    architecture[
                        "regularizer_dropout"
                    ]
                ),
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    with open(
        args.split
    ) as f:

        split = json.load(
            f
        )

    benchmark_fnames = sorted(
        split[
            "benchmark"
        ][
            "fnames"
        ]
    )

    benchmark_fnames = (
        benchmark_fnames[
            args.shard_id
            :: args.num_shards
        ]
    )

    if (
        args.limit_volumes
        is not None
    ):
        benchmark_fnames = (
            benchmark_fnames[
                :args.limit_volumes
            ]
        )

    dataset = (
        FastMRIBrainDataset(
            root=(
                "extracted/"
                "multicoil_train"
            ),
            acquisition="AXT2",
            cache_dir="code/.cache",
        )
    )

    volume_indices = {}

    for idx, raw in enumerate(
        dataset.raw_samples
    ):

        fname = Path(
            raw.fname
        ).name

        if fname in benchmark_fnames:

            volume_indices.setdefault(
                fname,
                [],
            ).append(
                idx
            )

    missing = (
        set(
            benchmark_fnames
        )
        - set(
            volume_indices
        )
    )

    if missing:

        raise RuntimeError(
            "Missing benchmark volumes: "
            f"{sorted(missing)}"
        )

    output_path = Path(
        args.output
    )

    completed = (
        load_existing(
            output_path
        )
        if args.resume
        else set()
    )

    print(
        "=" * 100
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "TUNING30 EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "checkpoint epoch:",
        int(
            checkpoint[
                "epoch"
            ]
        )
        + 1,
    )

    print(
        "best val calibrated SSIM:",
        checkpoint[
            "best_val_calibrated_ssim"
        ],
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
        "volumes in shard:",
        len(
            benchmark_fnames
        ),
    )

    print(
        "already completed:",
        len(
            completed
        ),
    )

    print(
        "=" * 100
    )

    for volume_position, fname in enumerate(
        benchmark_fnames,
        start=1,
    ):

        if fname in completed:

            print(
                "SKIP",
                fname,
            )

            continue

        indices = (
            volume_indices[
                fname
            ]
        )

        target_slices = []
        prediction_slices = []

        calibration_scale_values = []
        sampling_values = []
        acs_values = []

        espirit_runtime = 0.0
        model_runtime = 0.0

        for idx in indices:

            sample = dataset[
                idx
            ]

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
                    dtype=torch.complex64,
                )
            )

            target = (
                sample_256[
                    "target"
                ]
                .to(
                    device,
                    dtype=torch.float32,
                )
            )

            mask = make_mask(
                raw_mask,
                device,
            )

            torch.cuda.synchronize()

            t0 = time.time()

            sens = (
                estimate_sens_espirit(
                    masked_kspace,
                    num_low_frequencies=
                        num_low,
                    crop=
                        ESPIRIT_CROP,
                )
                .to(
                    device,
                    dtype=torch.complex64,
                )
            )

            torch.cuda.synchronize()

            espirit_runtime += (
                time.time()
                - t0
            )

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
                        dtype=torch.float32,
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

                measured_scaled = (
                    masked_kspace
                    / measurement_scale
                )[
                    None,
                    ...
                ]

                sens_batched = (
                    sens[
                        None,
                        ...
                    ]
                )

                torch.cuda.synchronize()

                t0 = time.time()

                (
                    rss_scaled,
                    predicted_kspace,
                ) = (
                    model.reconstruct_rss(
                        masked_kspace=
                            measured_scaled,

                        mask=
                            mask,

                        sensitivity_maps=
                            sens_batched,
                    )
                )

                torch.cuda.synchronize()

                model_runtime += (
                    time.time()
                    - t0
                )

                #
                # Corrected target-free amplitude calibration.
                #
                # raw_image is first restored from normalized
                # model space to adapted fastMRI measurement units.
                #
                raw_image = (
                    rss_scaled[
                        0
                    ]
                    * measurement_scale
                )

                calibration_scale = (
                    torch.linalg.vector_norm(
                        zf
                    )
                    /
                    torch.linalg.vector_norm(
                        raw_image
                    ).clamp_min(
                        1e-12
                    )
                )

                prediction_raw = (
                    raw_image
                    * calibration_scale
                )

                #
                # Alias retained only for the existing scalar
                # bookkeeping below; this is NOT a complex alpha.
                #
                alpha = calibration_scale

            if not torch.isfinite(
                prediction_raw
            ).all():

                raise RuntimeError(
                    "Non-finite reconstruction "
                    f"{fname} "
                    f"slice={sample['slice_num']}"
                )

            target_slices.append(
                target
                .detach()
                .cpu()
                .numpy()
            )

            prediction_slices.append(
                prediction_raw
                .detach()
                .cpu()
                .numpy()
            )

            calibration_scale_values.append(
                float(
                    torch.abs(
                        alpha
                    ).item()
                )
            )

            sampling_values.append(
                float(
                    raw_mask
                    .float()
                    .mean()
                    .item()
                )
            )

            acs_values.append(
                int(
                    num_low
                )
            )

            del (
                masked_kspace,
                target,
                sens,
                zf,
                measured_scaled,
                sens_batched,
                rss_scaled,
                predicted_kspace,
                prediction_raw,
            )

        target_volume = np.stack(
            target_slices,
            axis=0,
        )

        prediction_volume = np.stack(
            prediction_slices,
            axis=0,
        )

        max_value_256 = float(
            np.max(
                target_volume
            )
        )

        metrics = (
            compute_volume_metrics(
                target=
                    target_volume,

                prediction=
                    prediction_volume,

                max_value=
                    max_value_256,
            )
        )

        unique_acs = set(
            acs_values
        )

        if (
            len(
                unique_acs
            )
            != 1
        ):

            raise RuntimeError(
                "ACS changed within "
                f"{fname}: "
                f"{sorted(unique_acs)}"
            )

        row = {
            "fname":
                fname,

            "method":
                "shamaei_e2e_varnet",

            "acceleration":
                args.acceleration,

            "grid":
                "256x256",

            "adaptation":
                "image_crop_256",

            "num_slices":
                len(
                    indices
                ),

            "sampling_fraction":
                float(
                    np.mean(
                        sampling_values
                    )
                ),

            "acs_lines":
                acs_values[
                    0
                ],

            "espirit_crop":
                ESPIRIT_CROP,

            "espirit_runtime_sec":
                espirit_runtime,

            "runtime_sec":
                model_runtime,

            "nmse":
                metrics[
                    "nmse"
                ],

            "psnr":
                metrics[
                    "psnr"
                ],

            "ssim":
                metrics[
                    "ssim"
                ],

            "max_value_256":
                max_value_256,

            "calibration_scale_mean":
                float(
                    np.mean(
                        calibration_scale_values
                    )
                ),

            "calibration_scale_std":
                float(
                    np.std(
                        calibration_scale_values
                    )
                ),

            "checkpoint_epoch":
                int(
                    checkpoint[
                        "epoch"
                    ]
                )
                + 1,

            "checkpoint_val_ssim":
                float(
                    checkpoint[
                        "best_val_calibrated_ssim"
                    ]
                ),

            "output_calibration":
                (
                    "zf_rss_l2_norm_match"
                ),
        }

        append_row(
            output_path,
            row,
        )

        print(
            f"[{volume_position:02d}/"
            f"{len(benchmark_fnames):02d}] "
            f"{fname} "
            f"| NMSE="
            f"{metrics['nmse']:.6f} "
            f"| PSNR="
            f"{metrics['psnr']:.3f} "
            f"| SSIM="
            f"{metrics['ssim']:.4f} "
            f"| zf_l2_scale="
            f"{np.mean(calibration_scale_values):.4f}",
            flush=True,
        )

    print()
    print(
        "saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
