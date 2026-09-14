import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(CODE_ROOT)
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

from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
)

from evaluation.metrics import (
    compute_volume_metrics,
)


ESPIRIT_CROP = 0.0


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/"
            "shamaei_fastmri256_"
            "overfit_r4.pt"
        ),
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
                "Unexpected 1-D mask "
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
            "Unexpected mask shape "
            f"{tuple(mask.shape)}"
        )

    return (
        mask[
            None,
            None,
            ...
        ]
    )


def ensure_complex(
    x,
):

    if torch.is_complex(
        x
    ):
        return x

    if (
        x.shape[-1] == 2
    ):
        return torch.view_as_complex(
            x.contiguous()
        )

    raise RuntimeError(
        "Cannot convert output k-space "
        f"to complex: shape={tuple(x.shape)}, "
        f"dtype={x.dtype}"
    )


def measurement_scale_factor(
    predicted_kspace,
    measured_kspace,
    mask,
):
    """
    Solve

        alpha* =
            argmin_alpha
            || M (alpha k_pred - y) ||_2^2

    using ONLY acquired k-space.

    Closed form:

        alpha =
          <k_pred, y>
          / <k_pred, k_pred>

    No target image is used.
    """

    predicted_kspace = (
        ensure_complex(
            predicted_kspace
        )
    )

    measured_kspace = (
        ensure_complex(
            measured_kspace
        )
    )

    acquired = (
        mask
        .expand(
            predicted_kspace.shape[0],
            predicted_kspace.shape[1],
            predicted_kspace.shape[2],
            predicted_kspace.shape[3],
        )
    )

    pred_acquired = (
        predicted_kspace[
            acquired
        ]
    )

    measured_acquired = (
        measured_kspace[
            acquired
        ]
    )

    denominator = (
        (
            pred_acquired.conj()
            * pred_acquired
        )
        .real
        .sum()
        .clamp_min(
            1e-12
        )
    )

    numerator = (
        (
            pred_acquired.conj()
            * measured_acquired
        )
        .sum()
    )

    alpha = (
        numerator
        / denominator
    )

    return alpha


def acquired_relative_error(
    predicted_kspace,
    measured_kspace,
    mask,
):

    predicted_kspace = (
        ensure_complex(
            predicted_kspace
        )
    )

    measured_kspace = (
        ensure_complex(
            measured_kspace
        )
    )

    acquired = (
        mask
        .expand(
            predicted_kspace.shape[0],
            predicted_kspace.shape[1],
            predicted_kspace.shape[2],
            predicted_kspace.shape[3],
        )
    )

    pred = (
        predicted_kspace[
            acquired
        ]
    )

    measured = (
        measured_kspace[
            acquired
        ]
    )

    numerator = (
        torch.linalg.vector_norm(
            pred
            - measured
        )
    )

    denominator = (
        torch.linalg.vector_norm(
            measured
        )
        .clamp_min(
            1e-12
        )
    )

    return float(
        (
            numerator
            / denominator
        ).item()
    )


def metrics(
    target,
    prediction,
):

    target_np = (
        target
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    prediction_np = (
        prediction
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    max_value = float(
        target.max()
        .item()
    )

    return compute_volume_metrics(
        target_np[
            None,
            ...
        ],
        prediction_np[
            None,
            ...
        ],
        max_value=max_value,
    )


def print_metrics(
    name,
    result,
):

    print(
        f"{name:<22} "
        f"NMSE={result['nmse']:.8f} "
        f"PSNR={result['psnr']:.4f} "
        f"SSIM={result['ssim']:.6f}"
    )


def main():

    args = parse_args()

    device = torch.device(
        args.device
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    acceleration = int(
        checkpoint[
            "adaptation"
        ][
            "acceleration"
        ]
    )

    dataset_index = int(
        checkpoint[
            "sample"
        ][
            "dataset_index"
        ]
    )

    expected_fname = (
        checkpoint[
            "sample"
        ][
            "fname"
        ]
    )

    expected_slice = int(
        checkpoint[
            "sample"
        ][
            "slice_num"
        ]
    )

    measurement_scale = (
        torch.tensor(
            float(
                checkpoint[
                    "measurement_scale"
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
    )

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "MEASUREMENT SCALE CALIBRATION GATE"
    )

    print(
        "=" * 90
    )

    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "device:",
        device,
    )

    print(
        "acceleration:",
        acceleration,
    )

    print(
        "saved measurement scale:",
        float(
            measurement_scale.item()
        ),
    )

    #
    # ----------------------------------------------------------
    # Exact same sample used by the overfit gate.
    # ----------------------------------------------------------
    #

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

    sample = dataset[
        dataset_index
    ]

    if (
        sample["fname"]
        != expected_fname
    ):
        raise RuntimeError(
            "Checkpoint/sample filename mismatch: "
            f"{sample['fname']} "
            f"!= {expected_fname}"
        )

    if (
        int(
            sample[
                "slice_num"
            ]
        )
        != expected_slice
    ):
        raise RuntimeError(
            "Checkpoint/sample slice mismatch."
        )

    print()
    print(
        "file:",
        sample["fname"],
    )

    print(
        "slice:",
        sample["slice_num"],
    )

    print(
        "dataset index:",
        dataset_index,
    )

    #
    # ----------------------------------------------------------
    # Rebuild exact fastMRI-256 problem.
    # ----------------------------------------------------------
    #

    (
        sample_256,
        masked_kspace,
        raw_mask,
        num_low,
    ) = build_fastmri_256_problem(
        sample,
        acceleration,
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
            dtype=torch.complex64,
        )
    )

    print()
    print(
        "kspace shape:",
        tuple(
            masked_kspace.shape
        ),
    )

    print(
        "target shape:",
        tuple(
            target.shape
        ),
    )

    print(
        "mask shape:",
        tuple(
            mask.shape
        ),
    )

    print(
        "ACS:",
        int(
            num_low
        ),
    )

    print(
        "sampling fraction:",
        float(
            mask.float()
            .mean()
            .item()
        ),
    )

    #
    # ----------------------------------------------------------
    # Reference ZF.
    # ----------------------------------------------------------
    #

    zf_raw = (
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

    zf_metrics = metrics(
        target,
        zf_raw,
    )

    #
    # ----------------------------------------------------------
    # Load trained Shamaei model.
    # ----------------------------------------------------------
    #

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

    #
    # Same normalized domain used during training.
    #
    measured_scaled = (
        masked_kspace
        / measurement_scale
    )[
        None,
        ...
    ]

    sensitivity_batched = (
        sens_maps[
            None,
            ...
        ]
    )

    with torch.no_grad():

        (
            rss_scaled,
            predicted_kspace,
        ) = model.reconstruct_rss(
            masked_kspace=
                measured_scaled,

            mask=
                mask,

            sensitivity_maps=
                sensitivity_batched,
        )

    predicted_kspace = (
        ensure_complex(
            predicted_kspace
        )
    )

    print()
    print(
        "predicted kspace shape:",
        tuple(
            predicted_kspace.shape
        ),
    )

    print(
        "predicted kspace dtype:",
        predicted_kspace.dtype,
    )

    print(
        "RSS shape:",
        tuple(
            rss_scaled.shape
        ),
    )

    #
    # ----------------------------------------------------------
    # Uncalibrated output.
    # ----------------------------------------------------------
    #

    prediction_uncalibrated = (
        rss_scaled[
            0
        ]
        * measurement_scale
    )

    uncalibrated_metrics = (
        metrics(
            target,
            prediction_uncalibrated,
        )
    )

    error_before = (
        acquired_relative_error(
            predicted_kspace=
                predicted_kspace,

            measured_kspace=
                measured_scaled,

            mask=
                mask,
        )
    )

    #
    # ----------------------------------------------------------
    # Measurement-only global complex calibration.
    # ----------------------------------------------------------
    #

    alpha = (
        measurement_scale_factor(
            predicted_kspace=
                predicted_kspace,

            measured_kspace=
                measured_scaled,

            mask=
                mask,
        )
    )

    predicted_kspace_calibrated = (
        alpha
        * predicted_kspace
    )

    error_after = (
        acquired_relative_error(
            predicted_kspace=
                predicted_kspace_calibrated,

            measured_kspace=
                measured_scaled,

            mask=
                mask,
        )
    )

    #
    # Multiplying every coil/k-space sample
    # by alpha multiplies RSS magnitude by |alpha|.
    #
    prediction_calibrated = (
        rss_scaled[
            0
        ]
        * torch.abs(
            alpha
        )
        * measurement_scale
    )

    calibrated_metrics = (
        metrics(
            target,
            prediction_calibrated,
        )
    )

    print()
    print(
        "=" * 90
    )

    print(
        "MEASUREMENT-DOMAIN SCALE"
    )

    print(
        "=" * 90
    )

    print(
        "alpha real:",
        float(
            alpha.real.item()
        ),
    )

    print(
        "alpha imag:",
        float(
            alpha.imag.item()
        ),
    )

    print(
        "|alpha|:",
        float(
            torch.abs(
                alpha
            ).item()
        ),
    )

    print(
        "acquired relative error "
        "before:",
        error_before,
    )

    print(
        "acquired relative error "
        "after:",
        error_after,
    )

    print()
    print(
        "target max:",
        float(
            target.max()
            .item()
        ),
    )

    print(
        "uncalibrated pred max:",
        float(
            prediction_uncalibrated
            .max()
            .item()
        ),
    )

    print(
        "calibrated pred max:",
        float(
            prediction_calibrated
            .max()
            .item()
        ),
    )

    print()
    print(
        "=" * 90
    )

    print(
        "RAW-SCALE METRICS"
    )

    print(
        "=" * 90
    )

    print_metrics(
        "ZF",
        zf_metrics,
    )

    print_metrics(
        "Shamaei uncal.",
        uncalibrated_metrics,
    )

    print_metrics(
        "Shamaei calibrated",
        calibrated_metrics,
    )

    #
    # ----------------------------------------------------------
    # Gate.
    # ----------------------------------------------------------
    #

    if not math.isfinite(
        float(
            torch.abs(
                alpha
            ).item()
        )
    ):
        raise RuntimeError(
            "Non-finite calibration scale."
        )

    if not (
        error_after
        < error_before
    ):
        raise RuntimeError(
            "Measurement calibration "
            "did not reduce acquired "
            "k-space error."
        )

    if not (
        calibrated_metrics[
            "nmse"
        ]
        <
        uncalibrated_metrics[
            "nmse"
        ]
    ):
        raise RuntimeError(
            "Calibration did not "
            "improve raw NMSE."
        )

    if not (
        calibrated_metrics[
            "nmse"
        ]
        <
        zf_metrics[
            "nmse"
        ]
    ):
        raise RuntimeError(
            "Overfit Shamaei model "
            "still does not beat ZF "
            "after calibration."
        )

    if not (
        calibrated_metrics[
            "ssim"
        ]
        >
        zf_metrics[
            "ssim"
        ]
    ):
        raise RuntimeError(
            "Overfit Shamaei model "
            "still does not beat ZF "
            "SSIM after calibration."
        )

    print()
    print(
        "=" * 90
    )

    print(
        "SHAMAEI MEASUREMENT "
        "SCALE CALIBRATION GATE PASSED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
