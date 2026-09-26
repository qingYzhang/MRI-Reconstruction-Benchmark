#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RENDER_SCRIPT = (
    PROJECT_ROOT
    / "code"
    / "scripts"
    / "render_fastmri256_reconstruction_figures.py"
)


spec = importlib.util.spec_from_file_location(
    "render_fig",
    RENDER_SCRIPT,
)

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
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
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=532,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


def evaluate(
    name,
    target,
    prediction,
):

    target_np = (
        m.tensor_to_numpy(
            target
        )
    )

    pred_np = (
        m.tensor_to_numpy(
            prediction
        )
    )

    metrics = (
        m.slice_metrics(
            target=
                target_np,

            prediction=
                pred_np,

            max_value=
                float(
                    target_np.max()
                ),
        )
    )

    norm_ratio = (
        np.linalg.norm(
            pred_np.ravel()
        )
        /
        np.linalg.norm(
            target_np.ravel()
        )
    )

    print(
        f"{name:20s}"
        f" scale-norm={norm_ratio:8.4f}"
        f" | NMSE={metrics['nmse']:9.5f}"
        f" | PSNR={metrics['psnr']:7.3f}"
        f" | SSIM={metrics['ssim']:7.4f}"
    )


@torch.no_grad()
def main():

    args = parse_args()

    device = torch.device(
        args.device
    )

    dataset = (
        m.prior_eval
        .FastMRIBrainDataset(
            root=(
                "extracted/"
                "multicoil_train"
            ),
            acquisition="AXT2",
            cache_dir="code/.cache",
        )
    )

    sample = (
        dataset[
            args.dataset_idx
        ]
    )

    print("=" * 100)
    print("SHAMAEI SCALE CALIBRATION DIAGNOSTIC")
    print("=" * 100)

    print(
        "file:",
        sample[
            "fname"
        ],
    )

    print(
        "slice:",
        sample[
            "slice_num"
        ],
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    (
        sample_256,
        masked_kspace,
        raw_mask,
        num_low,
    ) = (
        m.prior_eval
        .build_fastmri_256_problem(
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

    mask = (
        m.prior_eval
        .prepare_mask(
            raw_mask,
            width=256,
            device=device,
        )
    )

    sens_maps = (
        m.prior_eval
        .estimate_sens_espirit(
            masked_kspace,
            num_low_frequencies=
                num_low,
            crop=0.0,
        )
        .to(
            device,
            dtype=torch.complex64,
        )
    )

    (
        model,
        _,
    ) = (
        m.load_checkpoint_state(
            args.checkpoint,
            acceleration=
                args.acceleration,
            device=device,
        )
    )

    #
    # ---------------------------------------------------------
    # Measurement normalization.
    # ---------------------------------------------------------
    #
    zf = (
        m.prior_eval
        .zero_filled_rss(
            masked_kspace,
            (
                256,
                256,
            ),
        )
        .to(
            device
        )
    )

    measurement_scale = (
        zf.quantile(
            0.999
        )
    )

    measured_scaled = (
        masked_kspace
        / measurement_scale
    )

    #
    # Shamaei mask: [B,1,H,W].
    #
    mask_2d = (
        mask
        .squeeze()
        .bool()
    )

    if (
        mask_2d.ndim
        == 1
    ):

        mask_2d = (
            mask_2d[
                None,
                :
            ]
            .expand(
                256,
                256,
            )
        )

    model_mask = (
        mask_2d[
            None,
            None,
            ...
        ]
    )

    (
        rss_scaled,
        output_kspace,
    ) = (
        model.reconstruct_rss(
            measured_scaled[
                None,
                ...
            ],
            model_mask,
            sens_maps[
                None,
                ...
            ],
        )
    )

    rss_scaled = (
        rss_scaled[
            0
        ]
    )

    pred_kspace = (
        output_kspace[
            0
        ]
    )

    acquired = (
        mask_2d[
            None,
            ...
        ]
        .expand_as(
            pred_kspace
        )
    )

    kp = (
        pred_kspace[
            acquired
        ]
    )

    yy = (
        measured_scaled[
            acquired
        ]
    )

    kp_power = (
        torch.sum(
            torch.abs(
                kp
            ) ** 2
        )
    )

    y_power = (
        torch.sum(
            torch.abs(
                yy
            ) ** 2
        )
    )

    cross = (
        torch.sum(
            torch.conj(
                kp
            )
            * yy
        )
    )

    #
    # Existing calibration.
    #
    alpha_complex = (
        cross
        / kp_power
    )

    #
    # Measurement-only alternative #1:
    #
    # Match acquired k-space L2 norm.
    #
    scale_norm = (
        torch.sqrt(
            y_power
            / kp_power
        )
    )

    #
    # Measurement-only alternative #2:
    #
    # Least-squares scalar using magnitudes only.
    #
    scale_mag_ls = (
        torch.sum(
            torch.abs(
                kp
            )
            * torch.abs(
                yy
            )
        )
        / kp_power
    )

    #
    # Complex coherence:
    #
    # 1   -> complex prediction and measurement
    #        have matching phase structure.
    #
    # ~0  -> complex phase mismatch/cancellation.
    #
    coherence = (
        torch.abs(
            cross
        )
        /
        torch.sqrt(
            kp_power
            * y_power
        )
    )

    print()
    print("=" * 100)
    print("GLOBAL ACQUIRED-KSPACE DIAGNOSTICS")
    print("=" * 100)

    print(
        "measurement_scale:",
        measurement_scale.item(),
    )

    print(
        "complex alpha:",
        alpha_complex.item(),
    )

    print(
        "|complex alpha|:",
        torch.abs(
            alpha_complex
        ).item(),
    )

    print(
        "measurement norm scale:",
        scale_norm.item(),
    )

    print(
        "magnitude-LS scale:",
        scale_mag_ls.item(),
    )

    print(
        "complex coherence:",
        coherence.item(),
    )

    print()
    print("=" * 100)
    print("PER-COIL COMPLEX CALIBRATION")
    print("=" * 100)

    coil_alphas = []

    for coil in range(
        pred_kspace.shape[0]
    ):

        coil_mask = (
            mask_2d
        )

        p = (
            pred_kspace[
                coil
            ][
                coil_mask
            ]
        )

        y = (
            measured_scaled[
                coil
            ][
                coil_mask
            ]
        )

        denominator = (
            torch.sum(
                torch.abs(
                    p
                ) ** 2
            )
        )

        alpha_c = (
            torch.sum(
                torch.conj(
                    p
                )
                * y
            )
            / denominator
        )

        coil_alphas.append(
            alpha_c
        )

        phase_deg = (
            torch.angle(
                alpha_c
            )
            * 180.0
            / np.pi
        )

        print(
            f"coil {coil:02d}"
            f" | |alpha|="
            f"{torch.abs(alpha_c).item():10.6f}"
            f" | phase="
            f"{phase_deg.item():9.2f} deg"
        )

    #
    # ---------------------------------------------------------
    # Produce image-space reconstructions using only
    # measurement-derived scale choices.
    # ---------------------------------------------------------
    #
    recon_no_calibration = (
        rss_scaled
        * measurement_scale
    )

    recon_complex_alpha = (
        rss_scaled
        * torch.abs(
            alpha_complex
        )
        * measurement_scale
    )

    recon_norm_scale = (
        rss_scaled
        * scale_norm
        * measurement_scale
    )

    recon_mag_ls = (
        rss_scaled
        * scale_mag_ls
        * measurement_scale
    )

    print()
    print("=" * 100)
    print("TARGET USED BELOW FOR DIAGNOSTIC METRICS ONLY")
    print("=" * 100)

    evaluate(
        "no calibration",
        target,
        recon_no_calibration,
    )

    evaluate(
        "complex alpha",
        target,
        recon_complex_alpha,
    )

    evaluate(
        "kspace norm scale",
        target,
        recon_norm_scale,
    )

    evaluate(
        "magnitude LS",
        target,
        recon_mag_ls,
    )

    #
    # ---------------------------------------------------------
    # Oracle image scalar.
    #
    # DEBUG ONLY. This must never be used for benchmark.
    # ---------------------------------------------------------
    #
    raw_np = (
        m.tensor_to_numpy(
            recon_no_calibration
        )
        .astype(
            np.float64
        )
    )

    target_np = (
        m.tensor_to_numpy(
            target
        )
        .astype(
            np.float64
        )
    )

    oracle_beta = (
        np.sum(
            raw_np
            * target_np
        )
        /
        np.sum(
            raw_np ** 2
        )
    )

    oracle = (
        recon_no_calibration
        * float(
            oracle_beta
        )
    )

    print()
    print(
        "oracle image beta "
        "(DEBUG ONLY):",
        oracle_beta,
    )

    evaluate(
        "oracle image scale",
        target,
        oracle,
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        "If complex coherence is low while "
        "k-space norm or magnitude-LS scaling "
        "restores the image metrics, then the "
        "current global complex alpha is failing "
        "because of phase cancellation rather "
        "than poor anatomical reconstruction."
    )


if __name__ == "__main__":
    main()
