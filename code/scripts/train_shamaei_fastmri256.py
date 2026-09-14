import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from skimage.metrics import (
    structural_similarity,
)

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torch.utils.data.distributed import (
    DistributedSampler,
)


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
)

from data.fastmri_256 import (
    FASTMRI_256_SIZE,
    build_fastmri_256_problem,
)

from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
)

from src.models.components.losses import (
    SSIMLoss,
)


SOURCE_SEED = 12345

NUM_CASCADES = 12

REGULARIZER_FILTERS = 18
REGULARIZER_POOLS = 4
REGULARIZER_DROPOUT = 0.0

DEFAULT_LR = 1e-3

DEFAULT_MAX_EPOCHS = 200

DEFAULT_ACCUM_STEPS = 8

DEFAULT_EARLY_STOP_PATIENCE = 10

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
        "--manifest",
        type=str,
        default=(
            "outputs/"
            "shamaei_fastmri256_"
            "training_manifest.json"
        ),
    )

    parser.add_argument(
        "--cache_root",
        type=str,
        default=(
            "cache/"
            "shamaei_fastmri256"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=
            DEFAULT_MAX_EPOCHS,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=
            DEFAULT_LR,
    )

    parser.add_argument(
        "--accum_steps",
        type=int,
        default=
            DEFAULT_ACCUM_STEPS,
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=
            DEFAULT_EARLY_STOP_PATIENCE,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--val_limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--expected_world_size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def init_distributed():

    if "RANK" not in os.environ:

        raise RuntimeError(
            "This script must be launched "
            "with torchrun."
        )

    rank = int(
        os.environ["RANK"]
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    torch.cuda.set_device(
        local_rank
    )

    dist.init_process_group(
        backend="nccl",
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    return (
        rank,
        local_rank,
        world_size,
        device,
    )


def seed_everything(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


def sha256_file(
    path,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def cache_path(
    cache_root,
    acceleration,
    split,
    fname,
    slice_num,
):

    return (
        Path(cache_root)
        / f"r{acceleration}"
        / split
        / (
            f"{Path(fname).stem}_"
            f"slice"
            f"{int(slice_num):03d}.pt"
        )
    )


class ShamaeiManifestDataset(
    Dataset
):

    def __init__(
        self,
        manifest,
        split,
        acceleration,
        cache_root,
        limit=None,
    ):

        self.split = split

        self.acceleration = (
            acceleration
        )

        self.cache_root = Path(
            cache_root
        )

        self.entries = list(
            manifest[
                split
            ][
                "samples"
            ]
        )

        if limit is not None:

            self.entries = (
                self.entries[
                    :limit
                ]
            )

        if split == "train":

            data_root = (
                "extracted/"
                "multicoil_train"
            )

        elif split == "validation":

            data_root = (
                "extracted/"
                "multicoil_val"
            )

        else:

            raise ValueError(
                split
            )

        self.dataset = (
            FastMRIBrainDataset(
                root=
                    data_root,

                acquisition=
                    "AXT2",

                cache_dir=
                    "code/.cache",
            )
        )

    def __len__(
        self
    ):

        return len(
            self.entries
        )

    def __getitem__(
        self,
        position,
    ):

        entry = (
            self.entries[
                position
            ]
        )

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

        sample = (
            self.dataset[
                idx
            ]
        )

        if (
            sample[
                "fname"
            ]
            != fname
        ):

            raise RuntimeError(
                "Manifest fname mismatch."
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
                "Manifest slice mismatch."
            )

        (
            sample_256,
            masked_kspace,
            raw_mask,
            num_low,
        ) = (
            build_fastmri_256_problem(
                sample,
                self.acceleration,
            )
        )

        cpath = cache_path(
            cache_root=
                self.cache_root,

            acceleration=
                self.acceleration,

            split=
                self.split,

            fname=
                fname,

            slice_num=
                slice_num,
        )

        if not cpath.exists():

            raise FileNotFoundError(
                cpath
            )

        cache = torch.load(
            cpath,
            map_location="cpu",
            weights_only=False,
        )

        if (
            cache[
                "fname"
            ]
            != fname
        ):

            raise RuntimeError(
                "Cache fname mismatch."
            )

        if (
            int(
                cache[
                    "slice_num"
                ]
            )
            != slice_num
        ):

            raise RuntimeError(
                "Cache slice mismatch."
            )

        if (
            int(
                cache[
                    "acceleration"
                ]
            )
            != self.acceleration
        ):

            raise RuntimeError(
                "Cache R mismatch."
            )

        if (
            int(
                cache[
                    "num_low_frequencies"
                ]
            )
            != int(
                num_low
            )
        ):

            raise RuntimeError(
                "Cache ACS mismatch."
            )

        sens_ri = cache[
            "sens_ri_fp16"
        ]

        if (
            sens_ri.dtype
            != torch.float16
        ):

            raise RuntimeError(
                "Sensitivity cache "
                "is not FP16."
            )

        #
        # raw_mask from fastMRI generally
        # collapses to a 1-D PE mask.
        #
        mask = (
            raw_mask
            .squeeze()
            .bool()
        )

        if (
            mask.ndim
            not in [1, 2]
        ):

            raise RuntimeError(
                "Unexpected mask shape: "
                f"{tuple(mask.shape)}"
            )

        return {
            "masked_kspace":
                masked_kspace,

            "target":
                sample_256[
                    "target"
                ].float(),

            "mask":
                mask,

            "sens_ri_fp16":
                sens_ri,

            "measurement_scale":
                float(
                    cache[
                        "measurement_scale"
                    ]
                ),

            "fname":
                fname,

            "slice_num":
                slice_num,

            "num_low":
                int(
                    num_low
                ),
        }


class TrainableShamaei(
    nn.Module
):

    """
    Wrapper so DDP sees the complete
    Shamaei reconstruction as its forward().
    """

    def __init__(
        self
    ):

        super().__init__()

        self.core = (
            ShamaeiE2EVarNetFastMRI(
                num_layers=
                    NUM_CASCADES,

                regularizer_num_filters=
                    REGULARIZER_FILTERS,

                regularizer_num_pull_layers=
                    REGULARIZER_POOLS,

                regularizer_dropout=
                    REGULARIZER_DROPOUT,
            )
        )

    def forward(
        self,
        masked_kspace,
        mask,
        sensitivity_maps,
    ):

        return (
            self.core
            .reconstruct_rss(
                masked_kspace=
                    masked_kspace,

                mask=
                    mask,

                sensitivity_maps=
                    sensitivity_maps,
            )
        )


def make_full_mask(
    mask,
    device,
):

    mask = (
        mask
        .to(
            device,
            non_blocking=True,
        )
        .bool()
        .squeeze()
    )

    if mask.ndim == 1:

        if (
            mask.numel()
            != FASTMRI_256_SIZE
        ):

            raise RuntimeError(
                "Unexpected 1-D "
                f"mask: {mask.shape}"
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
            "Unexpected mask "
            f"shape: {mask.shape}"
        )

    return (
        mask[
            None,
            None,
            ...
        ]
    )


def move_sample(
    item,
    device,
):

    masked = (
        item[
            "masked_kspace"
        ]
        .to(
            device,
            dtype=torch.complex64,
            non_blocking=True,
        )
    )

    target_raw = (
        item[
            "target"
        ]
        .to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
    )

    sens_ri = (
        item[
            "sens_ri_fp16"
        ]
        .to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        .contiguous()
    )

    sens = (
        torch.view_as_complex(
            sens_ri
        )
    )

    scale = torch.tensor(
        float(
            item[
                "measurement_scale"
            ]
        ),
        device=device,
        dtype=torch.float32,
    )

    scale = (
        scale
        .clamp_min(
            1e-8
        )
    )

    mask = make_full_mask(
        item[
            "mask"
        ],
        device,
    )

    #
    # B=1 because coil count is
    # variable across fastMRI volumes.
    #
    measured_scaled = (
        masked
        / scale
    )[
        None,
        ...
    ]

    sensitivity_maps = (
        sens[
            None,
            ...
        ]
    )

    target_scaled = (
        target_raw
        / scale
    )[
        None,
        ...
    ]

    return (
        measured_scaled,
        target_scaled,
        target_raw,
        mask,
        sensitivity_maps,
        scale,
    )


def normalize_absmax(
    image,
    eps=1e-8,
):

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

    #
    # Same convention used in the
    # successful overfit gate:
    # prediction and reference are
    # independently max-normalized.
    #
    prediction = (
        normalize_absmax(
            prediction
        )
        .unsqueeze(1)
    )

    target = (
        normalize_absmax(
            target
        )
        .unsqueeze(1)
    )

    loss = criterion(
        prediction,
        target,
    )

    if loss.ndim != 0:

        loss = (
            loss.mean()
        )

    return loss


def acquired_scale_alpha(
    predicted_kspace,
    measured_kspace,
    mask,
):

    if not torch.is_complex(
        predicted_kspace
    ):

        raise RuntimeError(
            "Predicted k-space "
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


def calibrated_metrics(
    rss_scaled,
    predicted_kspace,
    measured_scaled,
    mask,
    measurement_scale,
    target_raw,
):

    alpha = (
        acquired_scale_alpha(
            predicted_kspace=
                predicted_kspace,

            measured_kspace=
                measured_scaled,

            mask=
                mask,
        )
    )

    prediction_raw = (
        rss_scaled[
            0
        ]
        * torch.abs(
            alpha
        )
        * measurement_scale
    )

    denominator = (
        target_raw
        .pow(2)
        .sum()
        .clamp_min(
            1e-12
        )
    )

    nmse = (
        (
            prediction_raw
            - target_raw
        )
        .pow(2)
        .sum()
        /
        denominator
    )

    mse = (
        (
            prediction_raw
            - target_raw
        )
        .pow(2)
        .mean()
        .clamp_min(
            1e-20
        )
    )

    max_value = (
        target_raw
        .max()
        .clamp_min(
            1e-12
        )
    )

    psnr = (
        20.0
        * torch.log10(
            max_value
        )
        -
        10.0
        * torch.log10(
            mse
        )
    )

    #
    # Same raw-scale convention as the
    # fastMRI-256 benchmark:
    # normalize both by target maximum,
    # do NOT independently normalize pred.
    #
    target_unit = (
        target_raw
        / max_value
    )

    prediction_unit = (
        prediction_raw
        / max_value
    )

    target_np = (
        target_unit
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    prediction_np = (
        prediction_unit
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    ssim = float(
        structural_similarity(
            target_np,
            prediction_np,
            data_range=1.0,
            win_size=7,
        )
    )

    return {
        "nmse":
            float(
                nmse.item()
            ),

        "psnr":
            float(
                psnr.item()
            ),

        "ssim":
            ssim,

        "alpha_abs":
            float(
                torch.abs(
                    alpha
                ).item()
            ),
    }


def reduce_stats(
    values,
    device,
):

    tensor = torch.tensor(
        values,
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    return (
        tensor
        .cpu()
        .tolist()
    )


def atomic_torch_save(
    obj,
    path,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def save_checkpoint(
    path,
    ddp_model,
    optimizer,
    scheduler,
    epoch,
    best_val_ssim,
    bad_epochs,
    args,
    manifest_hash,
    world_size,
):

    model = (
        ddp_model
        .module
        .core
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "epoch":
            int(
                epoch
            ),

        "best_val_calibrated_ssim":
            float(
                best_val_ssim
            ),

        "bad_epochs":
            int(
                bad_epochs
            ),

        "architecture": {
            "num_layers":
                NUM_CASCADES,

            "regularizer_num_filters":
                REGULARIZER_FILTERS,

            "regularizer_num_pull_layers":
                REGULARIZER_POOLS,

            "regularizer_dropout":
                REGULARIZER_DROPOUT,
        },

        "training": {
            "seed":
                SOURCE_SEED,

            "optimizer":
                "Adam",

            "learning_rate_initial":
                args.learning_rate,

            "weight_decay":
                0.0,

            "loss":
                (
                    "Shamaei SSIMLoss "
                    "with independent "
                    "prediction/reference "
                    "absolute-max "
                    "normalization"
                ),

            "max_epochs":
                args.max_epochs,

            "gradient_accumulation":
                args.accum_steps,

            "world_size":
                world_size,

            "microbatch_per_rank":
                1,

            "nominal_effective_batch":
                (
                    world_size
                    * args.accum_steps
                ),

            "scheduler":
                (
                    "ReduceLROnPlateau("
                    "mode=max,"
                    "factor=0.1,"
                    "patience=10)"
                ),

            "checkpoint_monitor":
                (
                    "mean_validation_"
                    "calibrated_raw_ssim"
                ),

            "early_stop_patience":
                args.early_stop_patience,
        },

        "adaptation": {
            "grid":
                "256x256",

            "grid_method":
                "image_crop_256",

            "sensitivity":
                "ESPIRiT_crop0",

            "sensitivity_cache":
                "real_imag_fp16",

            "measurement_normalization":
                "ZF_RSS_q0.999",

            "output_calibration":
                (
                    "acquired_kspace_"
                    "complex_least_squares"
                ),

            "acceleration":
                args.acceleration,
        },

        "manifest":
            args.manifest,

        "manifest_sha256":
            manifest_hash,
    }

    atomic_torch_save(
        checkpoint,
        path,
    )


def append_history(
    path,
    row,
):

    path = Path(
        path
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


def train_one_epoch(
    ddp_model,
    loader,
    sampler,
    criterion,
    optimizer,
    epoch,
    accum_steps,
    rank,
    device,
):

    ddp_model.train()

    sampler.set_epoch(
        epoch
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_microsteps = len(
        loader
    )

    local_loss_sum = (
        torch.zeros(
            1,
            device=device,
            dtype=torch.float64,
        )
    )

    local_count = 0

    optimizer_steps = 0

    first_grad_norm = None

    for micro_idx, item in enumerate(
        loader
    ):

        (
            measured_scaled,
            target_scaled,
            _,
            mask,
            sens,
            _,
        ) = move_sample(
            item,
            device,
        )

        group_start = (
            micro_idx
            // accum_steps
            * accum_steps
        )

        group_size = min(
            accum_steps,
            total_microsteps
            - group_start,
        )

        group_position = (
            micro_idx
            - group_start
            + 1
        )

        sync_now = (
            group_position
            == group_size
        )

        sync_context = (
            contextlib.nullcontext()
            if sync_now
            else ddp_model.no_sync()
        )

        with sync_context:

            (
                rss_scaled,
                _,
            ) = ddp_model(
                measured_scaled,
                mask,
                sens,
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
                    f"Non-finite train loss "
                    f"at epoch={epoch}, "
                    f"micro={micro_idx}"
                )

            (
                loss
                / float(
                    group_size
                )
            ).backward()

        local_loss_sum += (
            loss.detach()
            .double()
        )

        local_count += 1

        if sync_now:

            if (
                first_grad_norm
                is None
            ):

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        ddp_model
                        .parameters(),
                        max_norm=
                            float(
                                "inf"
                            ),
                    )
                )

                first_grad_norm = float(
                    grad_norm.item()
                )

                if not math.isfinite(
                    first_grad_norm
                ):

                    raise RuntimeError(
                        "Non-finite "
                        "gradient norm."
                    )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        if (
            rank == 0
            and (
                micro_idx == 0
                or (
                    micro_idx + 1
                ) % 50 == 0
                or (
                    micro_idx + 1
                )
                == total_microsteps
            )
        ):

            print(
                f"  train "
                f"{micro_idx + 1:4d}/"
                f"{total_microsteps:4d} "
                f"| source_loss="
                f"{float(loss.item()):.6f}",
                flush=True,
            )

    stats = torch.tensor(
        [
            float(
                local_loss_sum.item()
            ),
            float(
                local_count
            ),
        ],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        stats,
        op=dist.ReduceOp.SUM,
    )

    train_loss = float(
        (
            stats[0]
            / stats[1]
        ).item()
    )

    return {
        "source_loss":
            train_loss,

        "optimizer_steps_per_rank":
            optimizer_steps,

        "first_grad_norm":
            first_grad_norm,
    }


@torch.no_grad()
def validate(
    ddp_model,
    loader,
    criterion,
    rank,
    device,
):

    ddp_model.eval()

    #
    # local sums:
    #
    # source loss
    # calibrated NMSE
    # calibrated PSNR
    # calibrated SSIM
    # |alpha|
    # count
    #
    sums = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    total_steps = len(
        loader
    )

    for val_idx, item in enumerate(
        loader
    ):

        (
            measured_scaled,
            target_scaled,
            target_raw,
            mask,
            sens,
            measurement_scale,
        ) = move_sample(
            item,
            device,
        )

        (
            rss_scaled,
            predicted_kspace,
        ) = ddp_model(
            measured_scaled,
            mask,
            sens,
        )

        source_loss = (
            source_ssim_loss(
                criterion=
                    criterion,

                prediction=
                    rss_scaled,

                target=
                    target_scaled,
            )
        )

        metrics = (
            calibrated_metrics(
                rss_scaled=
                    rss_scaled,

                predicted_kspace=
                    predicted_kspace,

                measured_scaled=
                    measured_scaled,

                mask=
                    mask,

                measurement_scale=
                    measurement_scale,

                target_raw=
                    target_raw,
            )
        )

        values = [
            float(
                source_loss.item()
            ),
            metrics[
                "nmse"
            ],
            metrics[
                "psnr"
            ],
            metrics[
                "ssim"
            ],
            metrics[
                "alpha_abs"
            ],
        ]

        if not all(
            math.isfinite(x)
            for x in values
        ):

            raise RuntimeError(
                "Non-finite validation "
                "metric."
            )

        sums[0] += values[0]
        sums[1] += values[1]
        sums[2] += values[2]
        sums[3] += values[3]
        sums[4] += values[4]
        sums[5] += 1.0

        if (
            rank == 0
            and (
                val_idx == 0
                or (
                    val_idx + 1
                ) % 50 == 0
                or (
                    val_idx + 1
                )
                == total_steps
            )
        ):

            print(
                f"  val   "
                f"{val_idx + 1:4d}/"
                f"{total_steps:4d}",
                flush=True,
            )

    reduced = reduce_stats(
        sums,
        device,
    )

    count = (
        reduced[5]
    )

    if count <= 0:
        raise RuntimeError(
            "No validation samples."
        )

    return {
        "source_loss":
            reduced[0]
            / count,

        "calibrated_nmse":
            reduced[1]
            / count,

        "calibrated_psnr":
            reduced[2]
            / count,

        "calibrated_ssim":
            reduced[3]
            / count,

        "alpha_abs_mean":
            reduced[4]
            / count,

        "num_samples":
            int(
                count
            ),
    }


def main():

    args = parse_args()

    (
        rank,
        local_rank,
        world_size,
        device,
    ) = init_distributed()

    if (
        world_size
        != args.expected_world_size
    ):

        raise RuntimeError(
            f"Expected "
            f"{args.expected_world_size} "
            f"processes, got "
            f"{world_size}."
        )

    seed_everything(
        SOURCE_SEED
    )

    manifest_path = Path(
        args.manifest
    )

    with manifest_path.open() as f:

        manifest = json.load(
            f
        )

    manifest_hash = sha256_file(
        manifest_path
    )

    output_dir = Path(
        args.output_dir
    )

    if rank == 0:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    dist.barrier()

    train_dataset = (
        ShamaeiManifestDataset(
            manifest=
                manifest,

            split=
                "train",

            acceleration=
                args.acceleration,

            cache_root=
                args.cache_root,

            limit=
                args.train_limit,
        )
    )

    val_dataset = (
        ShamaeiManifestDataset(
            manifest=
                manifest,

            split=
                "validation",

            acceleration=
                args.acceleration,

            cache_root=
                args.cache_root,

            limit=
                args.val_limit,
        )
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=
                world_size,

            rank=
                rank,

            shuffle=True,

            seed=
                SOURCE_SEED,

            drop_last=True,
        )
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=
                world_size,

            rank=
                rank,

            shuffle=False,

            drop_last=False,
        )
    )

    loader_kwargs = {
        "batch_size":
            None,

        "num_workers":
            args.num_workers,

        "pin_memory":
            True,
    }

    if args.num_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    train_loader = (
        DataLoader(
            train_dataset,
            sampler=
                train_sampler,
            **loader_kwargs,
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            sampler=
                val_sampler,
            **loader_kwargs,
        )
    )

    module = (
        TrainableShamaei()
        .to(
            device
        )
    )

    parameter_count = sum(
        p.numel()
        for p in module.parameters()
    )

    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=
            args.learning_rate,
        weight_decay=
            0.0,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=10,
        )
    )

    criterion = (
        SSIMLoss()
        .to(
            device
        )
    )

    start_epoch = 0

    best_val_ssim = (
        -float(
            "inf"
        )
    )

    bad_epochs = 0

    if args.resume is not None:

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        if (
            checkpoint[
                "manifest_sha256"
            ]
            != manifest_hash
        ):

            raise RuntimeError(
                "Manifest changed since "
                "checkpoint."
            )

        if (
            int(
                checkpoint[
                    "adaptation"
                ][
                    "acceleration"
                ]
            )
            != args.acceleration
        ):

            raise RuntimeError(
                "Resume acceleration "
                "mismatch."
            )

        module.core.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_val_ssim = float(
            checkpoint[
                "best_val_calibrated_ssim"
            ]
        )

        bad_epochs = int(
            checkpoint[
                "bad_epochs"
            ]
        )

    ddp_model = DDP(
        module,
        device_ids=[
            local_rank
        ],
        output_device=
            local_rank,
        broadcast_buffers=
            False,
        find_unused_parameters=
            False,
    )

    if rank == 0:

        print(
            "=" * 100
        )

        print(
            "SHAMAEI FASTMRI-256 "
            "FORMAL DDP TRAINING"
        )

        print(
            "=" * 100
        )

        print(
            "acceleration:",
            args.acceleration,
        )

        print(
            "world size:",
            world_size,
        )

        print(
            "parameter count:",
            parameter_count,
        )

        print(
            "train manifest samples:",
            len(
                train_dataset
            ),
        )

        print(
            "validation manifest samples:",
            len(
                val_dataset
            ),
        )

        print(
            "train microsteps/rank:",
            len(
                train_loader
            ),
        )

        print(
            "validation steps/rank:",
            len(
                val_loader
            ),
        )

        print(
            "microbatch/rank:",
            1,
        )

        print(
            "gradient accumulation:",
            args.accum_steps,
        )

        print(
            "nominal effective batch:",
            (
                world_size
                * args.accum_steps
            ),
        )

        print(
            "learning rate:",
            args.learning_rate,
        )

        print(
            "max epochs:",
            args.max_epochs,
        )

        print(
            "early-stop patience:",
            args.early_stop_patience,
        )

        print(
            "manifest sha256:",
            manifest_hash,
        )

        print(
            "resume:",
            args.resume,
        )

        print(
            "=" * 100,
            flush=True,
        )

    history_path = (
        output_dir
        / "history.csv"
    )

    for epoch in range(
        start_epoch,
        args.max_epochs,
    ):

        epoch_start = (
            time.time()
        )

        if rank == 0:

            print()
            print(
                "=" * 100
            )

            print(
                f"EPOCH "
                f"{epoch + 1}/"
                f"{args.max_epochs}"
            )

            print(
                "=" * 100,
                flush=True,
            )

        train_result = (
            train_one_epoch(
                ddp_model=
                    ddp_model,

                loader=
                    train_loader,

                sampler=
                    train_sampler,

                criterion=
                    criterion,

                optimizer=
                    optimizer,

                epoch=
                    epoch,

                accum_steps=
                    args.accum_steps,

                rank=
                    rank,

                device=
                    device,
            )
        )

        val_result = validate(
            ddp_model=
                ddp_model,

            loader=
                val_loader,

            criterion=
                criterion,

            rank=
                rank,

            device=
                device,
        )

        val_ssim = float(
            val_result[
                "calibrated_ssim"
            ]
        )

        improved = (
            val_ssim
            >
            best_val_ssim
        )

        if improved:

            best_val_ssim = (
                val_ssim
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

        scheduler.step(
            val_ssim
        )

        current_lr = float(
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        if rank == 0:

            print()
            print(
                f"epoch={epoch + 1:03d} "
                f"| train_source_loss="
                f"{train_result['source_loss']:.6f} "
                f"| val_source_loss="
                f"{val_result['source_loss']:.6f} "
                f"| val_cal_NMSE="
                f"{val_result['calibrated_nmse']:.6f} "
                f"| val_cal_PSNR="
                f"{val_result['calibrated_psnr']:.3f} "
                f"| val_cal_SSIM="
                f"{val_result['calibrated_ssim']:.6f} "
                f"| mean_|alpha|="
                f"{val_result['alpha_abs_mean']:.6f} "
                f"| lr="
                f"{current_lr:.3e} "
                f"| best="
                f"{best_val_ssim:.6f} "
                f"| bad_epochs="
                f"{bad_epochs} "
                f"| time="
                f"{elapsed / 60:.1f}m",
                flush=True,
            )

            row = {
                "epoch":
                    epoch + 1,

                "train_source_loss":
                    train_result[
                        "source_loss"
                    ],

                "val_source_loss":
                    val_result[
                        "source_loss"
                    ],

                "val_calibrated_nmse":
                    val_result[
                        "calibrated_nmse"
                    ],

                "val_calibrated_psnr":
                    val_result[
                        "calibrated_psnr"
                    ],

                "val_calibrated_ssim":
                    val_result[
                        "calibrated_ssim"
                    ],

                "val_alpha_abs_mean":
                    val_result[
                        "alpha_abs_mean"
                    ],

                "learning_rate":
                    current_lr,

                "best_val_calibrated_ssim":
                    best_val_ssim,

                "bad_epochs":
                    bad_epochs,

                "epoch_seconds":
                    elapsed,

                "optimizer_steps_per_rank":
                    train_result[
                        "optimizer_steps_per_rank"
                    ],
            }

            append_history(
                history_path,
                row,
            )

            save_checkpoint(
                path=
                    output_dir
                    / "last.pt",

                ddp_model=
                    ddp_model,

                optimizer=
                    optimizer,

                scheduler=
                    scheduler,

                epoch=
                    epoch,

                best_val_ssim=
                    best_val_ssim,

                bad_epochs=
                    bad_epochs,

                args=
                    args,

                manifest_hash=
                    manifest_hash,

                world_size=
                    world_size,
            )

            if improved:

                save_checkpoint(
                    path=
                        output_dir
                        / "best.pt",

                    ddp_model=
                        ddp_model,

                    optimizer=
                        optimizer,

                    scheduler=
                        scheduler,

                    epoch=
                        epoch,

                    best_val_ssim=
                        best_val_ssim,

                    bad_epochs=
                        bad_epochs,

                    args=
                        args,

                    manifest_hash=
                        manifest_hash,

                    world_size=
                        world_size,
                )

                print(
                    "BEST CHECKPOINT UPDATED",
                    flush=True,
                )

        dist.barrier()

        if (
            bad_epochs
            >= args.early_stop_patience
        ):

            if rank == 0:

                print()
                print(
                    "EARLY STOP: "
                    f"validation calibrated "
                    f"SSIM did not improve "
                    f"for {bad_epochs} epochs."
                )

            break

    if rank == 0:

        print()
        print(
            "=" * 100
        )

        print(
            "SHAMAEI TRAINING FINISHED"
        )

        print(
            "best validation "
            "calibrated SSIM:",
            best_val_ssim,
        )

        print(
            "best checkpoint:",
            output_dir
            / "best.pt",
        )

        print(
            "last checkpoint:",
            output_dir
            / "last.pt",
        )

        print(
            "=" * 100
        )

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
