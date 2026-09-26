#!/usr/bin/env python3

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CODE_ROOT = (
    PROJECT_ROOT
    / "code"
)

PRIOR_EVAL_PATH = (
    CODE_ROOT
    / "scripts"
    / "eval_fastmri256_prior_free.py"
)

SHAMAEI_ROOT = (
    PROJECT_ROOT
    / "external"
    / "shamaei_original"
)

sys.path.insert(
    0,
    str(CODE_ROOT),
)

sys.path.insert(
    0,
    str(SHAMAEI_ROOT),
)


# ---------------------------------------------------------------------
# Load the EXACT frozen five-method runner as a module.
#
# This is deliberate: figures must use the same ZF / CG / LACS / NeRP /
# CAPS code path that produced the benchmark numbers.
# ---------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "fastmri256_frozen_eval",
    PRIOR_EVAL_PATH,
)

prior_eval = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    prior_eval
)


from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
)


METHODS = [
    "zf",
    "cg",
    "lacs",
    "nerp",
    "caps",
    "shamaei",
]

DISPLAY_NAMES = {
    "zf":
        "ZF",

    "cg":
        "CG",

    "lacs":
        "LACS",

    "nerp":
        "NeRP",

    "caps":
        "CAPS",

    "shamaei":
        "Shamaei",

    "target":
        "Target",
}


DIAGNOSTIC_FIELDS = [
    "acceleration",
    "fname",
    "dataset_idx",
    "slice_num",
    "slice_position",
    "num_slices_in_volume",
    "method",
    "nmse",
    "psnr",
    "ssim",
    "target_max",
    "target_p995",
    "prediction_min",
    "prediction_max",
    "prediction_mean",
    "prediction_p995",
    "prediction_target_norm_ratio",
    "sampling_fraction",
    "acs_lines",
    "shamaei_measurement_scale",
    "shamaei_zf_l2_scale",
    "figure_path",
    "array_path",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[
            4,
            8,
        ],
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default=(
            "outputs/"
            "shamaei_fastmri256_split.json"
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=(
            "extracted/"
            "multicoil_train"
        ),
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="AXT2",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--max_volumes",
        type=int,
        default=None,
        help=(
            "Smoke-test limit. "
            "Default: all frozen benchmark volumes."
        ),
    )

    parser.add_argument(
        "--all_slices",
        action="store_true",
        help=(
            "Render every evaluated slice. "
            "Default: middle evaluated slice "
            "from every volume."
        ),
    )

    parser.add_argument(
        "--save_arrays",
        action="store_true",
        help=(
            "Save target and all reconstruction "
            "arrays as compressed NPZ for later "
            "metric debugging."
        ),
    )

    parser.add_argument(
        "--annotate_metrics",
        action="store_true",
        help=(
            "Show slice-level metrics in figure "
            "titles. Default is method names only."
        ),
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

    return parser.parse_args()


def sync():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_fname(
    raw_sample,
):

    if hasattr(
        prior_eval,
        "get_fname",
    ):
        return prior_eval.get_fname(
            raw_sample
        )

    if hasattr(
        raw_sample,
        "fname",
    ):
        return Path(
            raw_sample.fname
        ).name

    if isinstance(
        raw_sample,
        tuple,
    ):
        return Path(
            raw_sample[0]
        ).name

    raise RuntimeError(
        "Cannot determine filename "
        "from raw dataset sample."
    )


def get_raw_samples(
    dataset,
):

    if hasattr(
        prior_eval,
        "get_raw_samples",
    ):
        return prior_eval.get_raw_samples(
            dataset
        )

    if hasattr(
        dataset,
        "raw_samples",
    ):
        return dataset.raw_samples

    if hasattr(
        dataset,
        "examples",
    ):
        return dataset.examples

    raise RuntimeError(
        "Dataset has neither "
        "raw_samples nor examples."
    )


def build_volume_indices(
    dataset,
):

    result = {}

    raw_samples = (
        get_raw_samples(
            dataset
        )
    )

    for idx, raw in enumerate(
        raw_samples
    ):

        fname = get_fname(
            raw
        )

        result.setdefault(
            fname,
            [],
        ).append(
            idx
        )

    return result


def load_benchmark_fnames(
    split_json,
):

    with open(
        split_json
    ) as f:
        split = json.load(
            f
        )

    if (
        "benchmark"
        not in split
    ):
        raise RuntimeError(
            "split JSON has no "
            "'benchmark' section."
        )

    benchmark = (
        split[
            "benchmark"
        ]
    )

    if isinstance(
        benchmark,
        dict,
    ) and (
        "fnames"
        in benchmark
    ):

        fnames = (
            benchmark[
                "fnames"
            ]
        )

    elif isinstance(
        benchmark,
        list,
    ):

        if (
            len(benchmark) > 0
            and isinstance(
                benchmark[0],
                str,
            )
        ):
            fnames = (
                benchmark
            )

        else:
            fnames = sorted(
                {
                    entry[
                        "fname"
                    ]
                    for entry
                    in benchmark
                }
            )

    else:
        raise RuntimeError(
            "Unsupported benchmark schema "
            "in split JSON."
        )

    fnames = sorted(
        fnames
    )

    if len(fnames) != 30:

        print(
            "WARNING: expected frozen "
            "30-volume benchmark, got",
            len(fnames),
        )

    return fnames


def load_checkpoint_state(
    checkpoint_path,
    acceleration,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected Shamaei "
            "checkpoint format."
        )

    checkpoint_acc = (
        checkpoint.get(
            "acceleration",
            None,
        )
    )

    if (
        checkpoint_acc
        is not None
        and int(
            checkpoint_acc
        )
        != int(
            acceleration
        )
    ):
        raise RuntimeError(
            "Checkpoint acceleration "
            f"R{checkpoint_acc} does not "
            f"match requested R{acceleration}."
        )

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
    ]

    state = None

    for key in candidate_keys:

        value = (
            checkpoint.get(
                key,
                None,
            )
        )

        if isinstance(
            value,
            dict,
        ):
            state = value
            print(
                "checkpoint state key:",
                key,
            )
            break

    if state is None:

        #
        # Allow raw state_dict checkpoint.
        #
        tensor_values = [
            v
            for v in
            checkpoint.values()
            if torch.is_tensor(
                v
            )
        ]

        if tensor_values:
            state = checkpoint
            print(
                "checkpoint appears to be "
                "a raw state_dict"
            )

    if state is None:
        raise RuntimeError(
            "Could not locate model state "
            "in checkpoint. Keys: "
            f"{list(checkpoint.keys())}"
        )

    #
    # DDP checkpoints may use "module.".
    #
    if (
        len(state) > 0
        and all(
            key.startswith(
                "module."
            )
            for key
            in state.keys()
        )
    ):

        state = {
            key[
                len(
                    "module."
                ):
            ]:
                value
            for key, value
            in state.items()
        }

    model = (
        ShamaeiE2EVarNetFastMRI(
            num_layers=
                12,

            regularizer_num_filters=
                18,

            regularizer_num_pull_layers=
                4,

            regularizer_dropout=
                0.0,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    if (
        parameter_count
        != 29452068
    ):
        raise RuntimeError(
            "Unexpected Shamaei "
            "parameter count: "
            f"{parameter_count}"
        )

    print(
        "Shamaei parameter count:",
        parameter_count,
    )

    metadata = {
        "epoch":
            checkpoint.get(
                "epoch",
                "",
            ),

        "best_val_ssim":
            checkpoint.get(
                "best_val_calibrated_ssim",
                checkpoint.get(
                    "best_val_ssim",
                    checkpoint.get(
                        "best",
                        "",
                    ),
                ),
            ),
    }

    return (
        model,
        metadata,
    )


def tensor_to_numpy(
    x,
):

    if torch.is_tensor(
        x
    ):

        x = (
            x
            .detach()
            .cpu()
            .numpy()
        )

    x = np.asarray(
        x
    )

    if np.iscomplexobj(
        x
    ):
        x = np.abs(
            x
        )

    return (
        x.astype(
            np.float32,
            copy=False,
        )
    )


def slice_metrics(
    target,
    prediction,
    max_value,
):

    target = tensor_to_numpy(
        target
    )

    prediction = tensor_to_numpy(
        prediction
    )

    if (
        target.shape
        != prediction.shape
    ):
        raise RuntimeError(
            "Metric shape mismatch: "
            f"target={target.shape}, "
            f"prediction={prediction.shape}"
        )

    denominator = (
        np.sum(
            target ** 2
        )
    )

    if denominator <= 0:
        nmse = float(
            "nan"
        )
    else:
        nmse = float(
            np.sum(
                (
                    target
                    - prediction
                ) ** 2
            )
            / denominator
        )

    mse = float(
        np.mean(
            (
                target
                - prediction
            ) ** 2
        )
    )

    if mse <= 0:
        psnr = float(
            "inf"
        )
    else:
        psnr = float(
            20.0
            * np.log10(
                float(
                    max_value
                )
            )
            - 10.0
            * np.log10(
                mse
            )
        )

    data_range = float(
        max_value
    )

    if data_range <= 0:
        data_range = float(
            np.max(
                target
            )
            - np.min(
                target
            )
        )

    if data_range <= 0:
        ssim = float(
            "nan"
        )
    else:
        ssim = float(
            structural_similarity(
                target,
                prediction,
                data_range=
                    data_range,
            )
        )

    return {
        "nmse":
            nmse,

        "psnr":
            psnr,

        "ssim":
            ssim,
    }


def make_acquired_mask(
    mask,
    reference,
):

    acquired = (
        mask
        .squeeze()
        .bool()
    )

    #
    # Usually mask becomes [H,W].
    # Expand across coils.
    #
    while (
        acquired.ndim
        < reference.ndim
    ):
        acquired = (
            acquired
            .unsqueeze(
                0
            )
        )

    try:
        acquired = (
            acquired
            .expand(
                reference.shape
            )
        )

    except RuntimeError as exc:
        raise RuntimeError(
            "Could not broadcast "
            f"mask {tuple(mask.shape)} "
            "to predicted k-space "
            f"{tuple(reference.shape)}"
        ) from exc

    return acquired


@torch.no_grad()
def run_shamaei(
    model,
    masked_kspace,
    mask,
    sens_maps,
):

    """
    Corrected Shamaei fastMRI-256 inference.

    Global amplitude is calibrated using:

        s = ||ZF(y)||_2 / ||x_raw||_2

    where both quantities are derived from the acquired
    measurement. No target information is used.
    """

    height = int(
        masked_kspace.shape[-2]
    )

    width = int(
        masked_kspace.shape[-1]
    )

    #
    # Zero-filled RSS directly from acquired raw k-space.
    #
    zf = (
        prior_eval.zero_filled_rss(
            masked_kspace,
            (
                height,
                width,
            ),
        )
        .to(
            masked_kspace.device,
            dtype=torch.float32,
        )
    )

    #
    # Same measurement normalization used during
    # corrected Shamaei training/evaluation.
    #
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
    )

    #
    # Common benchmark mask -> [B,1,H,W].
    #
    mask_2d = (
        mask
        .squeeze()
        .to(
            masked_kspace.device
        )
        .bool()
    )

    if mask_2d.ndim == 1:

        if (
            mask_2d.numel()
            != width
        ):
            raise RuntimeError(
                "Unexpected Shamaei "
                "1-D mask shape: "
                f"{tuple(mask_2d.shape)}"
            )

        mask_2d = (
            mask_2d[
                None,
                :
            ]
            .expand(
                height,
                width,
            )
        )

    if (
        tuple(
            mask_2d.shape
        )
        != (
            height,
            width,
        )
    ):
        raise RuntimeError(
            "Unexpected Shamaei "
            "2-D mask shape: "
            f"{tuple(mask_2d.shape)}"
        )

    model_mask = (
        mask_2d[
            None,
            None,
            ...
        ]
    )

    measured_batch = (
        measured_scaled[
            None,
            ...
        ]
    )

    sens_batch = (
        sens_maps[
            None,
            ...
        ]
    )

    (
        rss_scaled,
        _predicted_kspace,
    ) = (
        model.reconstruct_rss(
            measured_batch,
            model_mask,
            sens_batch,
        )
    )

    if (
        rss_scaled.ndim != 3
        or rss_scaled.shape[0] != 1
    ):
        raise RuntimeError(
            "Unexpected Shamaei RSS "
            f"shape: {tuple(rss_scaled.shape)}"
        )

    #
    # Restore raw adapted fastMRI units first.
    #
    raw_image = (
        rss_scaled[
            0
        ]
        * measurement_scale
    )

    #
    # Corrected target-free amplitude calibration.
    #
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

    reconstruction = (
        raw_image
        * calibration_scale
    )

    if (
        tuple(
            reconstruction.shape
        )
        != (
            height,
            width,
        )
    ):
        raise RuntimeError(
            "Unexpected corrected "
            "Shamaei image shape: "
            f"{tuple(reconstruction.shape)}"
        )

    if not torch.isfinite(
        reconstruction
    ).all():
        raise RuntimeError(
            "Corrected Shamaei "
            "reconstruction is non-finite."
        )

    return (
        reconstruction.float(),
        {
            "measurement_scale":
                float(
                    measurement_scale.item()
                ),

            "zf_l2_scale":
                float(
                    calibration_scale.item()
                ),
        },
    )


def save_figure(
    images,
    metrics,
    out_path,
    fname,
    acceleration,
    slice_num,
    annotate_metrics,
):

    order = [
        "zf",
        "cg",
        "lacs",
        "nerp",
        "caps",
        "shamaei",
        "target",
    ]

    target = (
        images[
            "target"
        ]
    )

    #
    # SAME display window for every method.
    # Important for visual comparison.
    #
    vmax = float(
        np.quantile(
            target,
            0.995,
        )
    )

    if (
        not np.isfinite(
            vmax
        )
        or vmax <= 0
    ):
        vmax = float(
            np.max(
                target
            )
        )

    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        len(order),
        figsize=(
            21,
            3.45,
        ),
    )

    for ax, method in zip(
        axes,
        order,
    ):

        ax.imshow(
            images[
                method
            ],
            cmap="gray",
            vmin=0.0,
            vmax=vmax,
        )

        ax.axis(
            "off"
        )

        if (
            method != "target"
            and annotate_metrics
        ):

            m = (
                metrics[
                    method
                ]
            )

            title = (
                f"{DISPLAY_NAMES[method]}\n"
                f"N {m['nmse']:.3f} | "
                f"P {m['psnr']:.1f} | "
                f"S {m['ssim']:.3f}"
            )

        else:

            title = (
                DISPLAY_NAMES[
                    method
                ]
            )

        ax.set_title(
            title,
            fontsize=10,
        )

    fig.suptitle(
        (
            f"{fname}   "
            f"R={acceleration}   "
            f"slice={slice_num}"
        ),
        fontsize=11,
    )

    plt.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.01,
        top=0.82,
        wspace=0.02,
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(
        fig
    )


def prediction_stats(
    target,
    prediction,
):

    target_norm = float(
        np.linalg.norm(
            target.ravel()
        )
    )

    pred_norm = float(
        np.linalg.norm(
            prediction.ravel()
        )
    )

    norm_ratio = (
        pred_norm
        / max(
            target_norm,
            1e-12,
        )
    )

    return {
        "prediction_min":
            float(
                np.min(
                    prediction
                )
            ),

        "prediction_max":
            float(
                np.max(
                    prediction
                )
            ),

        "prediction_mean":
            float(
                np.mean(
                    prediction
                )
            ),

        "prediction_p995":
            float(
                np.quantile(
                    prediction,
                    0.995,
                )
            ),

        "prediction_target_norm_ratio":
            float(
                norm_ratio
            ),
    }


def main():

    args = parse_args()

    device = torch.device(
        args.device
        if (
            args.device
            != "cuda"
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    if (
        device.type
        != "cuda"
    ):
        print(
            "WARNING: CUDA is not active. "
            "CAPS/NeRP will be very slow."
        )

    output_dir = Path(
        args.output_dir
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    arrays_dir = (
        output_dir
        / "arrays"
    )

    diagnostics_path = (
        output_dir
        / "diagnostics.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_fnames = (
        load_benchmark_fnames(
            args.split_json
        )
    )

    if (
        args.max_volumes
        is not None
    ):
        benchmark_fnames = (
            benchmark_fnames[
                :
                args.max_volumes
            ]
        )

    if (
        args.num_shards < 1
        or args.shard_id < 0
        or args.shard_id >= args.num_shards
    ):
        raise ValueError(
            "Invalid --num_shards/--shard_id."
        )

    benchmark_fnames = [
        fname
        for position, fname
        in enumerate(
            benchmark_fnames
        )
        if (
            position
            % args.num_shards
            == args.shard_id
        )
    ]

    dataset = (
        prior_eval.FastMRIBrainDataset(
            root=
                args.data_root,

            acquisition=
                args.acquisition,

            cache_dir=
                "code/.cache",
        )
    )

    volume_indices = (
        build_volume_indices(
            dataset
        )
    )

    missing = [
        fname
        for fname in
        benchmark_fnames
        if fname
        not in
        volume_indices
    ]

    if missing:
        raise RuntimeError(
            "Frozen benchmark volumes "
            "missing from dataset: "
            f"{missing}"
        )

    (
        shamaei_model,
        checkpoint_metadata,
    ) = (
        load_checkpoint_state(
            args.checkpoint,
            args.acceleration,
            device,
        )
    )

    caps_session = (
        prior_eval.CAPSSession(
            acceleration=
                args.acceleration,

            device=
                device,
        )
    )

    diagnostics_rows = []

    print(
        "=" * 100
    )

    print(
        "FASTMRI-256 RECONSTRUCTION "
        "FIGURE GENERATION"
    )

    print(
        "=" * 100
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "device:",
        device,
    )

    print(
        "volumes:",
        len(
            benchmark_fnames
        ),
    )

    print(
        "slice mode:",
        (
            "ALL"
            if args.all_slices
            else "MIDDLE"
        ),
    )

    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "checkpoint epoch:",
        checkpoint_metadata[
            "epoch"
        ],
    )

    print(
        "checkpoint best val SSIM:",
        checkpoint_metadata[
            "best_val_ssim"
        ],
    )

    print(
        "output:",
        output_dir,
    )

    print(
        "=" * 100
    )

    for volume_position, fname in enumerate(
        benchmark_fnames,
        start=1,
    ):

        indices = (
            volume_indices[
                fname
            ]
        )

        if (
            args.all_slices
        ):

            selected_indices = (
                list(
                    indices
                )
            )

        else:

            middle_position = (
                len(indices)
                // 2
            )

            selected_indices = [
                indices[
                    middle_position
                ]
            ]

        print()

        print(
            f"[{volume_position}/"
            f"{len(benchmark_fnames)}] "
            f"{fname}"
        )

        print(
            "  volume slices:",
            len(indices),
        )

        print(
            "  figures:",
            len(
                selected_indices
            ),
        )

        for dataset_idx in (
            selected_indices
        ):

            sample = (
                dataset[
                    dataset_idx
                ]
            )

            slice_num = int(
                sample[
                    "slice_num"
                ]
            )

            slice_position = (
                indices.index(
                    dataset_idx
                )
            )

            volume_stem = (
                Path(
                    fname
                ).stem
            )

            figure_path = (
                figures_dir
                / volume_stem
                / (
                    f"slice_"
                    f"{slice_num:03d}.png"
                )
            )

            array_path = (
                arrays_dir
                / volume_stem
                / (
                    f"slice_"
                    f"{slice_num:03d}.npz"
                )
            )

            if (
                args.resume
                and figure_path.exists()
                and (
                    not args.save_arrays
                    or array_path.exists()
                )
            ):

                print(
                    "  SKIP existing:",
                    figure_path,
                )

                continue

            print(
                "  reconstruct "
                f"dataset_idx={dataset_idx} "
                f"slice_num={slice_num}"
            )

            (
                sample_256,
                masked_kspace,
                raw_mask,
                num_low,
            ) = (
                prior_eval
                .build_fastmri_256_problem(
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

            target = (
                sample_256[
                    "target"
                ]
                .to(
                    device,
                    dtype=
                        torch.float32,
                )
            )

            mask = (
                prior_eval.prepare_mask(
                    raw_mask,
                    width=
                        prior_eval
                        .FASTMRI_256_SIZE,
                    device=
                        device,
                )
            )

            sampling_fraction = float(
                mask.float()
                .mean()
                .item()
            )

            #
            # IMPORTANT:
            # Same acquired-data ESPIRiT
            # used by formal benchmark.
            #
            sens_maps = (
                prior_eval
                .estimate_sens_espirit(
                    masked_kspace,
                    num_low_frequencies=
                        num_low,
                    crop=
                        prior_eval
                        .ESPIRIT_CROP,
                )
                .to(
                    device,
                    dtype=
                        torch.complex64,
                )
            )

            if not torch.isfinite(
                sens_maps
            ).all():
                raise RuntimeError(
                    "Non-finite ESPIRiT maps."
                )

            #
            # ---------------------------------------------------------
            # ZF
            # ---------------------------------------------------------
            #
            zf = (
                prior_eval
                .zero_filled_rss(
                    masked_kspace,
                    (
                        prior_eval
                        .FASTMRI_256_SIZE,

                        prior_eval
                        .FASTMRI_256_SIZE,
                    ),
                )
                .to(
                    device
                )
            )

            #
            # ---------------------------------------------------------
            # CG
            # ---------------------------------------------------------
            #
            cg = (
                prior_eval
                .cg_sense(
                    measured_kspace=
                        masked_kspace,

                    sens_maps=
                        sens_maps,

                    mask=
                        mask,

                    num_iters=
                        prior_eval
                        .CG_ITERS,

                    lambda_reg=
                        prior_eval
                        .CG_LAMBDA,
                )
                .abs()
            )

            #
            # ---------------------------------------------------------
            # LACS
            # ---------------------------------------------------------
            #
            lacs_result = (
                prior_eval
                .reconstruct_lacs(
                    measured_kspace=
                        masked_kspace,

                    sens_maps=
                        sens_maps,

                    mask=
                        mask,

                    prior=
                        None,

                    wavelet_levels=
                        prior_eval
                        .LACS_LEVELS,

                    verbose=
                        False,
                )
            )

            lacs = (
                lacs_result[
                    "reconstruction"
                ]
                .abs()
            )

            #
            # ---------------------------------------------------------
            # NeRP
            # ---------------------------------------------------------
            #
            nerp_result = (
                prior_eval
                .reconstruct_nerp(
                    measured_kspace=
                        masked_kspace,

                    sens_maps=
                        sens_maps,

                    mask=
                        mask,

                    prior=
                        None,

                    reconstruction_iters=
                        prior_eval
                        .NERP_ITERS,

                    reconstruction_learning_rate=
                        prior_eval
                        .NERP_LR,

                    seed=
                        0,

                    verbose=
                        False,
                )
            )

            nerp = (
                nerp_result[
                    "reconstruction"
                ]
                .abs()
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            #
            # ---------------------------------------------------------
            # CAPS
            # ---------------------------------------------------------
            #
            caps_seed = (
                prior_eval
                .make_caps_sample_seed(
                    fname=
                        fname,

                    slice_num=
                        slice_num,

                    acceleration=
                        args.acceleration,
                )
            )

            (
                caps,
                _caps_runtime,
                caps_scale,
            ) = (
                caps_session
                .reconstruct(
                    masked_kspace=
                        masked_kspace,

                    sens_maps=
                        sens_maps,

                    raw_mask=
                        raw_mask,

                    sample_seed=
                        caps_seed,
                )
            )

            #
            # ---------------------------------------------------------
            # Shamaei non-enhanced E2E-VarNet
            # ---------------------------------------------------------
            #
            (
                shamaei,
                shamaei_diag,
            ) = run_shamaei(
                model=
                    shamaei_model,

                masked_kspace=
                    masked_kspace,

                mask=
                    mask,

                sens_maps=
                    sens_maps,
            )

            #
            # Convert everything to the SAME
            # raw adapted-target units.
            #
            images = {
                "zf":
                    tensor_to_numpy(
                        zf
                    ),

                "cg":
                    tensor_to_numpy(
                        cg
                    ),

                "lacs":
                    tensor_to_numpy(
                        lacs
                    ),

                "nerp":
                    tensor_to_numpy(
                        nerp
                    ),

                "caps":
                    tensor_to_numpy(
                        caps
                    ),

                "shamaei":
                    tensor_to_numpy(
                        shamaei
                    ),

                "target":
                    tensor_to_numpy(
                        target
                    ),
            }

            expected_shape = (
                prior_eval
                .FASTMRI_256_SIZE,
                prior_eval
                .FASTMRI_256_SIZE,
            )

            for key, image in (
                images.items()
            ):

                if (
                    tuple(
                        image.shape
                    )
                    != expected_shape
                ):
                    raise RuntimeError(
                        f"{key} shape "
                        f"{image.shape}, "
                        "expected "
                        f"{expected_shape}"
                    )

                if not np.isfinite(
                    image
                ).all():
                    raise RuntimeError(
                        f"{key} has "
                        "non-finite pixels."
                    )

            target_np = (
                images[
                    "target"
                ]
            )

            target_max = float(
                np.max(
                    target_np
                )
            )

            target_p995 = float(
                np.quantile(
                    target_np,
                    0.995,
                )
            )

            #
            # NOTE:
            # Formal volume metric uses one common
            # max_value_256 for the WHOLE volume.
            #
            # Here slice diagnostics use target_max
            # only to locate suspicious cases.
            # These diagnostics do NOT replace the
            # formal benchmark metrics.
            #
            metric_values = {}

            for method in METHODS:

                metric_values[
                    method
                ] = (
                    slice_metrics(
                        target=
                            target_np,

                        prediction=
                            images[
                                method
                            ],

                        max_value=
                            target_max,
                    )
                )

            save_figure(
                images=
                    images,

                metrics=
                    metric_values,

                out_path=
                    figure_path,

                fname=
                    fname,

                acceleration=
                    args.acceleration,

                slice_num=
                    slice_num,

                annotate_metrics=
                    args.annotate_metrics,
            )

            if (
                args.save_arrays
            ):

                array_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.savez_compressed(
                    array_path,

                    zf=
                        images[
                            "zf"
                        ],

                    cg=
                        images[
                            "cg"
                        ],

                    lacs=
                        images[
                            "lacs"
                        ],

                    nerp=
                        images[
                            "nerp"
                        ],

                    caps=
                        images[
                            "caps"
                        ],

                    shamaei=
                        images[
                            "shamaei"
                        ],

                    target=
                        images[
                            "target"
                        ],
                )

            for method in METHODS:

                stats = (
                    prediction_stats(
                        target=
                            target_np,

                        prediction=
                            images[
                                method
                            ],
                    )
                )

                diag = {
                    "acceleration":
                        args.acceleration,

                    "fname":
                        fname,

                    "dataset_idx":
                        dataset_idx,

                    "slice_num":
                        slice_num,

                    "slice_position":
                        slice_position,

                    "num_slices_in_volume":
                        len(indices),

                    "method":
                        method,

                    "nmse":
                        metric_values[
                            method
                        ][
                            "nmse"
                        ],

                    "psnr":
                        metric_values[
                            method
                        ][
                            "psnr"
                        ],

                    "ssim":
                        metric_values[
                            method
                        ][
                            "ssim"
                        ],

                    "target_max":
                        target_max,

                    "target_p995":
                        target_p995,

                    **stats,

                    "sampling_fraction":
                        sampling_fraction,

                    "acs_lines":
                        int(
                            num_low
                        ),

                    "shamaei_measurement_scale":
                        (
                            shamaei_diag[
                                "measurement_scale"
                            ]
                            if method
                            == "shamaei"
                            else ""
                        ),

                    "shamaei_zf_l2_scale":
                        (
                            shamaei_diag[
                                "zf_l2_scale"
                            ]
                            if method
                            == "shamaei"
                            else ""
                        ),

                    "figure_path":
                        str(
                            figure_path
                        ),

                    "array_path":
                        (
                            str(
                                array_path
                            )
                            if args.save_arrays
                            else ""
                        ),
                }

                diagnostics_rows.append(
                    diag
                )

            #
            # Write incrementally so a long run
            # remains debuggable if interrupted.
            #
            diagnostics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with diagnostics_path.open(
                "w",
                newline="",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=
                        DIAGNOSTIC_FIELDS,
                )

                writer.writeheader()

                writer.writerows(
                    diagnostics_rows
                )

            print(
                "    saved:",
                figure_path,
            )

            print(
                "    norm ratios:",
                "ZF="
                f"{prediction_stats(target_np, images['zf'])['prediction_target_norm_ratio']:.3f}",
                "CG="
                f"{prediction_stats(target_np, images['cg'])['prediction_target_norm_ratio']:.3f}",
                "LACS="
                f"{prediction_stats(target_np, images['lacs'])['prediction_target_norm_ratio']:.3f}",
                "NeRP="
                f"{prediction_stats(target_np, images['nerp'])['prediction_target_norm_ratio']:.3f}",
                "CAPS="
                f"{prediction_stats(target_np, images['caps'])['prediction_target_norm_ratio']:.3f}",
                "Shamaei="
                f"{prediction_stats(target_np, images['shamaei'])['prediction_target_norm_ratio']:.3f}",
            )

            print(
                "    Shamaei:",
                "measurement_scale="
                f"{shamaei_diag['measurement_scale']:.6g}",
                "zf_l2_scale="
                f"{shamaei_diag['zf_l2_scale']:.6g}",
            )

            del zf
            del cg
            del lacs
            del lacs_result
            del nerp
            del nerp_result
            del caps
            del shamaei
            del sens_maps
            del masked_kspace
            del target

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

    print()
    print(
        "=" * 100
    )

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "diagnostics:",
        diagnostics_path,
    )

    print(
        "diagnostic rows:",
        len(
            diagnostics_rows
        ),
    )

    print(
        "figures:",
        figures_dir,
    )


if __name__ == "__main__":
    main()
