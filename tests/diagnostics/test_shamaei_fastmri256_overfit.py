import argparse
import json
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

PROJECT_ROOT = (
    CODE_ROOT.parent
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

#
# Exact loss implementation used by
# the Shamaei source repository.
#
from src.models.components.losses import (
    SSIMLoss,
)


ESPIRIT_CROP = 0.0

SOURCE_SEED = 12345

DEFAULT_STEPS = 200

DEFAULT_LR = 1e-3


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        default=4,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LR,
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


def get_raw_samples(
    dataset,
):

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


def get_fname(
    raw_sample,
):

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
        "Cannot parse raw sample."
    )


def build_volume_index(
    dataset,
):

    volume_indices = {}

    for idx, raw_sample in enumerate(
        get_raw_samples(
            dataset
        )
    ):

        fname = get_fname(
            raw_sample
        )

        volume_indices.setdefault(
            fname,
            [],
        ).append(
            idx
        )

    return volume_indices


def make_batched_mask(
    raw_mask,
    device,
):

    mask = prepare_mask(
        raw_mask,
        width=FASTMRI_256_SIZE,
        device=device,
    )

    #
    # Reduce whatever broadcast layout
    # prepare_mask returned to an explicit
    # 2-D Cartesian mask.
    #
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
            raise ValueError(
                "Unexpected 1-D mask "
                f"shape: {tuple(mask.shape)}"
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
        raise ValueError(
            "Unexpected mask "
            f"shape: {tuple(mask.shape)}"
        )

    #
    # DIRECT VarNet expects:
    #
    # [B, 1, H, W]
    #
    return (
        mask[
            None,
            None,
            ...,
        ]
    )


def normalize_for_source_ssim(
    image,
    eps=1e-8,
):
    """
    Paper convention:

        image / abs(image).max()

    image shape:
        [B,H,W]
    """

    scale = (
        image
        .abs()
        .amax(
            dim=(-2, -1),
            keepdim=True,
        )
        .clamp_min(
            eps
        )
    )

    return (
        image
        / scale
    )


def source_ssim_loss(
    criterion,
    prediction,
    target,
):
    """
    prediction / target:
        [B,H,W]

    Repository SSIMLoss expects:
        [B,C,H,W]
    """

    prediction = (
        normalize_for_source_ssim(
            prediction
        )
        .unsqueeze(1)
    )

    target = (
        normalize_for_source_ssim(
            target
        )
        .unsqueeze(1)
    )

    loss = criterion(
        prediction,
        target,
    )

    if loss.ndim != 0:
        loss = loss.mean()

    return loss


def compute_metrics(
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
        target.max().item()
    )

    #
    # One-slice "volume".
    #
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


def gradients_are_finite(
    model,
):

    count = 0
    squared_norm = 0.0

    for parameter in (
        model.parameters()
    ):

        if parameter.grad is None:
            continue

        count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            return (
                False,
                count,
                float("nan"),
            )

        grad_norm = float(
            parameter
            .grad
            .detach()
            .float()
            .norm()
            .item()
        )

        squared_norm += (
            grad_norm
            ** 2
        )

    return (
        True,
        count,
        math.sqrt(
            squared_norm
        ),
    )


@torch.no_grad()
def evaluate_model(
    model,
    measured_kspace,
    mask,
    sensitivity_maps,
    measurement_scale,
    target_raw,
    criterion,
):

    model.eval()

    rss_scaled, _ = (
        model.reconstruct_rss(
            masked_kspace=
                measured_kspace,

            mask=
                mask,

            sensitivity_maps=
                sensitivity_maps,
        )
    )

    if not torch.isfinite(
        rss_scaled
    ).all():
        raise RuntimeError(
            "Non-finite reconstruction."
        )

    target_scaled = (
        target_raw[
            None,
            ...
        ]
        / measurement_scale
    )

    loss = source_ssim_loss(
        criterion=
            criterion,

        prediction=
            rss_scaled,

        target=
            target_scaled,
    )

    prediction_raw = (
        rss_scaled[0]
        * measurement_scale
    )

    metrics = compute_metrics(
        target=
            target_raw,

        prediction=
            prediction_raw,
    )

    return (
        float(
            loss.item()
        ),
        prediction_raw,
        metrics,
    )


def main():

    args = parse_args()

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    torch.manual_seed(
        SOURCE_SEED
    )

    np.random.seed(
        SOURCE_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SOURCE_SEED
        )

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "SINGLE-SLICE OVERFIT GATE"
    )

    print(
        "=" * 90
    )

    print(
        "device:",
        device,
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "steps:",
        args.steps,
    )

    print(
        "learning rate:",
        args.learning_rate,
    )

    print(
        "seed:",
        SOURCE_SEED,
    )

    #
    # ----------------------------------------------------------
    # Frozen split.
    # ----------------------------------------------------------
    #

    split_path = Path(
        args.split
    )

    with split_path.open() as f:
        split = json.load(
            f
        )

    train_fnames = set(
        split[
            "train"
        ][
            "fnames"
        ]
    )

    benchmark_fnames = set(
        split[
            "benchmark"
        ][
            "fnames"
        ]
    )

    if (
        train_fnames
        & benchmark_fnames
    ):
        raise RuntimeError(
            "Train/benchmark leakage."
        )

    #
    # ----------------------------------------------------------
    # Dataset.
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

    volume_indices = (
        build_volume_index(
            dataset
        )
    )

    available_train = sorted(
        train_fnames
        & set(
            volume_indices
        )
    )

    if not available_train:
        raise RuntimeError(
            "No training volumes found."
        )

    #
    # Deterministically choose the first
    # TRAIN volume and its middle slice.
    #
    fname = (
        available_train[0]
    )

    indices = (
        volume_indices[
            fname
        ]
    )

    dataset_index = (
        indices[
            len(indices)
            // 2
        ]
    )

    sample = dataset[
        dataset_index
    ]

    if (
        sample["fname"]
        in benchmark_fnames
    ):
        raise RuntimeError(
            "Selected overfit sample "
            "belongs to benchmark cohort."
        )

    print()
    print(
        "training file:",
        sample["fname"],
    )

    print(
        "dataset index:",
        dataset_index,
    )

    print(
        "slice number:",
        sample["slice_num"],
    )

    print(
        "benchmark member:",
        sample["fname"]
        in benchmark_fnames,
    )

    #
    # ----------------------------------------------------------
    # Exact frozen fastMRI-256 acquisition.
    # ----------------------------------------------------------
    #

    (
        sample_256,
        masked_kspace,
        raw_mask,
        num_low,
    ) = build_fastmri_256_problem(
        sample,
        args.acceleration,
    )

    masked_kspace = (
        masked_kspace
        .to(
            device,
            dtype=torch.complex64,
        )
    )

    target_raw = (
        sample_256[
            "target"
        ]
        .to(
            device,
            dtype=torch.float32,
        )
    )

    mask = make_batched_mask(
        raw_mask=
            raw_mask,

        device=
            device,
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
            target_raw.shape
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
    # Common fastMRI benchmark sensitivity policy.
    #
    # IMPORTANT:
    # Sensitivity is estimated ONCE from acquired
    # data only. No target information is used.
    # ----------------------------------------------------------
    #

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
        .detach()
    )

    if not torch.isfinite(
        sens_maps
    ).all():
        raise RuntimeError(
            "Non-finite sensitivity maps."
        )

    print(
        "sensitivity shape:",
        tuple(
            sens_maps.shape
        ),
    )

    #
    # Add batch dimension:
    #
    # [C,H,W]
    # ->
    # [1,C,H,W]
    #
    sens_maps = (
        sens_maps[
            None,
            ...
        ]
    )

    #
    # ----------------------------------------------------------
    # Measurement-derived normalization.
    #
    # We must NOT use target intensity to scale
    # the measured input.
    #
    # Use the 99.9th percentile of the
    # zero-filled RSS reconstruction.
    # ----------------------------------------------------------
    #

    with torch.no_grad():

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

        measurement_scale = (
            torch.quantile(
                zf_raw.flatten(),
                0.999,
            )
            .clamp_min(
                1e-8
            )
        )

    print(
        "measurement scale:",
        float(
            measurement_scale.item()
        ),
    )

    print(
        "target max:",
        float(
            target_raw.max()
            .item()
        ),
    )

    print(
        "scale / target max:",
        float(
            (
                measurement_scale
                /
                target_raw.max()
            ).item()
        ),
    )

    #
    # Input normalization is entirely
    # measurement-derived.
    #
    measured_scaled = (
        masked_kspace
        / measurement_scale
    )[
        None,
        ...
    ]

    target_scaled = (
        target_raw
        / measurement_scale
    )[
        None,
        ...
    ]

    #
    # ----------------------------------------------------------
    # Exact source architecture.
    # ----------------------------------------------------------
    #

    model = (
        ShamaeiE2EVarNetFastMRI(
            num_layers=12,
            regularizer_num_filters=18,
            regularizer_num_pull_layers=4,
            regularizer_dropout=0.0,
        )
        .to(
            device
        )
    )

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "number of cascades:",
        len(
            model
            .model
            .layers_list
        ),
    )

    print(
        "parameter count:",
        num_parameters,
    )

    criterion = (
        SSIMLoss()
        .to(
            device
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    #
    # ----------------------------------------------------------
    # Initial reconstruction.
    # ----------------------------------------------------------
    #

    (
        initial_loss,
        initial_prediction,
        initial_metrics,
    ) = evaluate_model(
        model=
            model,

        measured_kspace=
            measured_scaled,

        mask=
            mask,

        sensitivity_maps=
            sens_maps,

        measurement_scale=
            measurement_scale,

        target_raw=
            target_raw,

        criterion=
            criterion,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "INITIAL"
    )

    print(
        "=" * 90
    )

    print(
        "source SSIM loss:",
        initial_loss,
    )

    print(
        "NMSE:",
        initial_metrics[
            "nmse"
        ],
    )

    print(
        "PSNR:",
        initial_metrics[
            "psnr"
        ],
    )

    print(
        "SSIM:",
        initial_metrics[
            "ssim"
        ],
    )

    #
    # ----------------------------------------------------------
    # One-slice overfit.
    # ----------------------------------------------------------
    #

    first_grad_norm = None
    first_grad_count = None

    print()
    print(
        "=" * 90
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 90
    )

    for step in range(
        1,
        args.steps + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        rss_scaled, _ = (
            model.reconstruct_rss(
                masked_kspace=
                    measured_scaled,

                mask=
                    mask,

                sensitivity_maps=
                    sens_maps,
            )
        )

        loss = source_ssim_loss(
            criterion=
                criterion,

            prediction=
                rss_scaled,

            target=
                target_scaled,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss "
                f"at step {step}."
            )

        loss.backward()

        (
            finite_grads,
            grad_count,
            grad_norm,
        ) = gradients_are_finite(
            model
        )

        if not finite_grads:
            raise RuntimeError(
                f"Non-finite gradients "
                f"at step {step}."
            )

        if step == 1:

            first_grad_norm = (
                grad_norm
            )

            first_grad_count = (
                grad_count
            )

        optimizer.step()

        if (
            step == 1
            or step % 10 == 0
            or step == args.steps
        ):

            with torch.no_grad():

                prediction_raw = (
                    rss_scaled[0]
                    * measurement_scale
                )

                numerator = (
                    (
                        prediction_raw
                        - target_raw
                    )
                    .pow(2)
                    .sum()
                )

                denominator = (
                    target_raw
                    .pow(2)
                    .sum()
                    .clamp_min(
                        1e-12
                    )
                )

                current_nmse = float(
                    (
                        numerator
                        / denominator
                    ).item()
                )

            print(
                f"step "
                f"{step:04d}/"
                f"{args.steps} "
                f"| source_loss="
                f"{float(loss.item()):.6f} "
                f"| raw_nmse="
                f"{current_nmse:.6f} "
                f"| grad_norm="
                f"{grad_norm:.6e}"
            )

    #
    # ----------------------------------------------------------
    # Final reconstruction AFTER final update.
    # ----------------------------------------------------------
    #

    (
        final_loss,
        final_prediction,
        final_metrics,
    ) = evaluate_model(
        model=
            model,

        measured_kspace=
            measured_scaled,

        mask=
            mask,

        sensitivity_maps=
            sens_maps,

        measurement_scale=
            measurement_scale,

        target_raw=
            target_raw,

        criterion=
            criterion,
    )

    final_learning_rates = [
        float(
            layer
            .learning_rate
            .detach()
            .cpu()
            .item()
        )
        for layer
        in (
            model
            .model
            .layers_list
        )
    ]

    print()
    print(
        "=" * 90
    )

    print(
        "FINAL"
    )

    print(
        "=" * 90
    )

    print(
        "initial source loss:",
        initial_loss,
    )

    print(
        "final source loss:",
        final_loss,
    )

    print(
        "initial NMSE:",
        initial_metrics[
            "nmse"
        ],
    )

    print(
        "final NMSE:",
        final_metrics[
            "nmse"
        ],
    )

    print(
        "initial PSNR:",
        initial_metrics[
            "psnr"
        ],
    )

    print(
        "final PSNR:",
        final_metrics[
            "psnr"
        ],
    )

    print(
        "initial SSIM:",
        initial_metrics[
            "ssim"
        ],
    )

    print(
        "final SSIM:",
        final_metrics[
            "ssim"
        ],
    )

    print(
        "first-step grad count:",
        first_grad_count,
    )

    print(
        "first-step grad norm:",
        first_grad_norm,
    )

    print(
        "final DC learning rates:",
        final_learning_rates,
    )

    #
    # ----------------------------------------------------------
    # Save checkpoint.
    # ----------------------------------------------------------
    #

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "architecture": {
            "num_layers":
                12,
            "regularizer_num_filters":
                18,
            "regularizer_num_pull_layers":
                4,
            "regularizer_dropout":
                0.0,
        },

        "training": {
            "steps":
                args.steps,
            "learning_rate":
                args.learning_rate,
            "optimizer":
                "Adam",
            "loss":
                (
                    "source SSIMLoss after "
                    "per-image absolute-max "
                    "normalization"
                ),
            "seed":
                SOURCE_SEED,
        },

        "adaptation": {
            "grid":
                "fastMRI-256",
            "acceleration":
                args.acceleration,
            "sensitivity":
                (
                    "shared ESPIRiT "
                    "crop=0.0"
                ),
            "input_normalization":
                (
                    "measurement-derived "
                    "ZF RSS q=0.999"
                ),
        },

        "sample": {
            "fname":
                sample["fname"],
            "slice_num":
                int(
                    sample[
                        "slice_num"
                    ]
                ),
            "dataset_index":
                dataset_index,
        },

        "measurement_scale":
            float(
                measurement_scale
                .item()
            ),

        "initial_metrics":
            initial_metrics,

        "final_metrics":
            final_metrics,

        "initial_source_loss":
            initial_loss,

        "final_source_loss":
            final_loss,
    }

    torch.save(
        checkpoint,
        output_path,
    )

    print()
    print(
        "checkpoint saved:",
        output_path,
    )

    #
    # ----------------------------------------------------------
    # Gate checks.
    # ----------------------------------------------------------
    #

    if (
        first_grad_count is None
        or first_grad_count == 0
    ):
        raise RuntimeError(
            "No trainable gradients."
        )

    if (
        first_grad_norm is None
        or not math.isfinite(
            first_grad_norm
        )
        or first_grad_norm <= 0.0
    ):
        raise RuntimeError(
            "Invalid first-step "
            "gradient norm."
        )

    if not (
        final_loss
        < initial_loss
    ):
        raise RuntimeError(
            "Source SSIM loss did "
            "not improve."
        )

    if not (
        final_metrics[
            "ssim"
        ]
        >
        initial_metrics[
            "ssim"
        ]
    ):
        raise RuntimeError(
            "Raw-scale SSIM did "
            "not improve."
        )

    print()
    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "OVERFIT GATE PASSED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
