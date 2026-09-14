import argparse
import copy
import csv
import hashlib
import gc
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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

LAPS_ROOT = (
    PROJECT_ROOT
    / "external"
    / "laps_original"
)


#
# Patched vendored diffusers first.
#
sys.path.insert(
    0,
    str(
        LAPS_ROOT
        / "submodules"
        / "diffusers"
        / "src"
    ),
)

sys.path.insert(
    0,
    str(
        LAPS_ROOT
        / "src"
    ),
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

from evaluation.metrics import (
    compute_volume_metrics,
)

from methods.lacs import (
    reconstruct_lacs,
)

from methods.nerp import (
    reconstruct_nerp,
)

from operators.sense import (
    prepare_mask,
    cg_sense,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)

from laps.configs import (
    recon as recon_configs,
)

from laps.recon.linops import (
    CartesianSenseLinop,
)

from laps.recon.reconstructor import (
    StableDiffusionReconstructor,
)


ALL_METHODS = [
    "zf",
    "cg",
    "lacs",
    "nerp",
    "caps",
]

CG_ITERS = 50
CG_LAMBDA = 0.01

NERP_ITERS = 1000
NERP_LR = 1e-5

LACS_LEVELS = 2

CAPS_BASE_SEED = 0

ESPIRIT_CROP = 0.0

TUNING_VOLUMES = 30


FIELDNAMES = [
    "fname",
    "method",
    "acceleration",
    "grid",
    "adaptation",
    "num_slices",
    "sampling_fraction",
    "acs_lines",
    "espirit_crop",
    "espirit_runtime_sec",
    "runtime_sec",
    "nmse",
    "psnr",
    "ssim",
    "max_value_256",
    "cg_iters",
    "cg_lambda",
    "lacs_wavelet_levels",
    "nerp_iters",
    "nerp_lr",
    "caps_seed_mode",
    "caps_base_seed",
    "caps_n_avgs",
    "caps_prior_start_timestep",
    "caps_output_dc_iters",
    "caps_lambda_ldm",
    "caps_scale_mean",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        choices=[
            "smoke",
            "tuning30",
        ],
        default="tuning30",
    )

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
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
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
        "--seed",
        type=int,
        default=20260904,
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
        "--limit_volumes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def make_caps_sample_seed(
    fname,
    slice_num,
    acceleration,
):
    """
    Deterministic per-slice CAPS seed.

    This makes stochastic diffusion sampling
    invariant to Slurm sharding / execution order.
    """

    key = (
        f"{CAPS_BASE_SEED}|"
        f"{fname}|"
        f"{int(slice_num)}|"
        f"R{int(acceleration)}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**31)
    )

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


def make_laps_mask(
    raw_mask,
    device,
):

    mask = (
        raw_mask
        .squeeze()
        .to(device)
    )

    if mask.ndim == 1:

        if (
            mask.numel()
            != FASTMRI_256_SIZE
        ):
            raise ValueError(
                "Unexpected 1-D mask: "
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
        raise ValueError(
            "Unexpected CAPS mask: "
            f"{tuple(mask.shape)}"
        )

    return mask.to(
        torch.complex64
    )


def make_caps_params(
    acceleration,
    device,
):

    params = copy.deepcopy(
        recon_configs
        .sd_caps_medvae_4
    )

    params.device = device

    params.im_size = (
        FASTMRI_256_SIZE,
        FASTMRI_256_SIZE,
    )

    params.debug = False

    #
    # Official recon_test.py settings
    # for 1-D sampling.
    #
    if acceleration >= 7:

        params.output_dc_config = {
            "n_iters":
                5,

            "threshold":
                1e-5,

            "lambda_l2":
                5e-4,

            "lambda_ldm":
                0.03,

            "lambda_l2_from_data":
                False,

            "avg_before_dc":
                False,
        }

    else:

        params.output_dc_config = {
            "n_iters":
                10,

            "threshold":
                1e-5,

            "lambda_l2":
                5e-4,

            "lambda_ldm":
                0.025,

            "lambda_l2_from_data":
                False,

            "avg_before_dc":
                False,
        }

    return params


class CAPSSession:
    """
    Keep one official CAPS diffusion model
    loaded per process.

    Only the MRI forward model is replaced
    from slice to slice.
    """

    def __init__(
        self,
        acceleration,
        device,
    ):

        self.acceleration = (
            acceleration
        )

        self.device = device

        self.params = (
            make_caps_params(
                acceleration,
                device,
            )
        )

        self.reconstructor = None


    def reconstruct(
        self,
        masked_kspace,
        sens_maps,
        raw_mask,
        sample_seed,
    ):

        mask_laps = (
            make_laps_mask(
                raw_mask,
                self.device,
            )
        )

        A = (
            CartesianSenseLinop(
                mps=
                    sens_maps,

                mask=
                    mask_laps,

                ishape=(
                    FASTMRI_256_SIZE,
                    FASTMRI_256_SIZE,
                ),
            )
            .to(
                self.device
            )
        )

        scale = (
            A.H(
                masked_kspace
            )
            .abs()
            .quantile(
                0.999
            )
        )

        if (
            not torch.isfinite(
                scale
            )
            or scale.item() <= 0
        ):
            raise RuntimeError(
                "Invalid CAPS scale: "
                f"{scale}"
            )

        #
        # Load model only once.
        #
        if (
            self.reconstructor
            is None
        ):

            print(
                "\n"
                "Loading CAPS model "
                "for this process..."
            )

            self.reconstructor = (
                StableDiffusionReconstructor(
                    forward_model=
                        A,

                    params=
                        self.params,

                    load_model_to_cpu=
                        False,
                )
            )

        else:

            #
            # Same pattern used by official
            # evaluation when forward model
            # changes between samples.
            #
            self.reconstructor.forward_model = (
                A
            )

            self.reconstructor.dc_args[
                "forward_model"
            ] = A

        #
        # Deterministic benchmark:
        # same fixed stochastic draw set
        # for each slice.
        #
        #
        # Deterministic per-slice random
        # stream. Different slices receive
        # different streams, while results
        # remain invariant to sharding.
        #
        torch.manual_seed(
            sample_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                sample_seed
            )

        sync()

        start = time.time()

        output = (
            self.reconstructor
            .reconstruct(
                (
                    masked_kspace
                    / scale
                )[
                    None,
                    ...
                ]
            )
        )

        sync()

        runtime = (
            time.time()
            - start
        )

        recon_complex = (
            output.recon[
                0
            ]
            .detach()
            .to(
                self.device
            )
        )

        sens_mask = (
            torch.linalg.vector_norm(
                sens_maps,
                dim=0,
            )
            > 0.5
        )

        recon_complex = (
            recon_complex
            * sens_mask
        )

        #
        # Restore original fastMRI
        # intensity scale.
        #
        recon_raw = (
            recon_complex.abs()
            * scale
        )

        return (
            recon_raw,
            runtime,
            float(
                scale.item()
            ),
        )


def load_existing_rows(
    output_path,
):

    if (
        not output_path.exists()
    ):
        return []

    with output_path.open() as f:
        return list(
            csv.DictReader(f)
        )


def save_rows_atomic(
    output_path,
    rows,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    os.replace(
        tmp,
        output_path,
    )


def build_volume_indices(
    dataset,
):

    volume_indices = (
        defaultdict(list)
    )

    raw_samples = (
        get_raw_samples(
            dataset
        )
    )

    for idx, raw_sample in enumerate(
        raw_samples
    ):

        fname = get_fname(
            raw_sample
        )

        volume_indices[
            fname
        ].append(
            idx
        )

    return volume_indices


def select_cohort(
    dataset,
    volume_indices,
    args,
):

    all_fnames = sorted(
        volume_indices.keys()
    )

    if (
        args.subset
        == "smoke"
    ):

        smoke_sample = (
            dataset[532]
        )

        fname = smoke_sample[
            "fname"
        ]

        indices = (
            volume_indices[
                fname
            ]
        )

        center_idx = min(
            indices,
            key=lambda idx:
                abs(
                    int(
                        dataset[idx][
                            "slice_num"
                        ]
                    )
                    - 8
                ),
        )

        #
        # Smoke evaluates only the
        # previously frozen central slice.
        #
        volume_indices = dict(
            volume_indices
        )

        volume_indices[
            fname
        ] = [
            center_idx
        ]

        selected = [
            fname
        ]

        return (
            selected,
            volume_indices,
        )

    rng = random.Random(
        args.seed
    )

    selected = rng.sample(
        all_fnames,
        min(
            TUNING_VOLUMES,
            len(
                all_fnames
            ),
        ),
    )

    selected = sorted(
        selected
    )

    return (
        selected,
        volume_indices,
    )


def run_timed(
    func,
):

    sync()

    start = time.time()

    result = func()

    sync()

    return (
        result,
        time.time()
        - start,
    )


def main():

    args = parse_args()

    if (
        args.shard_id < 0
        or
        args.shard_id
        >= args.num_shards
    ):
        raise ValueError(
            "Invalid shard_id."
        )

    device = torch.device(
        args.device
    )

    methods = list(
        dict.fromkeys(
            args.methods
        )
    )

    print(
        "=" * 100
    )

    print(
        "FASTMRI-256 PRIOR-FREE "
        "BENCHMARK"
    )

    print(
        "=" * 100
    )

    print(
        "subset:",
        args.subset,
    )

    print(
        "data root:",
        args.data_root,
    )

    print(
        "acquisition:",
        args.acquisition,
    )

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "methods:",
        methods,
    )

    print(
        "device:",
        device,
    )

    print(
        "grid:",
        "256x256",
    )

    print(
        "adaptation:",
        "image_crop_256",
    )

    print(
        "tuning seed:",
        args.seed,
    )

    print(
        "shard:",
        f"{args.shard_id}/"
        f"{args.num_shards}",
    )

    dataset = (
        FastMRIBrainDataset(
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

    (
        full_selected_fnames,
        volume_indices,
    ) = select_cohort(
        dataset,
        volume_indices,
        args,
    )

    print(
        "Full cohort size:",
        len(
            full_selected_fnames
        ),
    )

    #
    # Debug gate while preserving the
    # beginning of the exact tuning30
    # cohort.
    #
    if (
        args.limit_volumes
        is not None
    ):
        full_selected_fnames = (
            full_selected_fnames[
                :
                args.limit_volumes
            ]
        )

    selected_fnames = (
        full_selected_fnames[
            args.shard_id
            ::
            args.num_shards
        ]
    )

    print(
        "Volumes in this shard:",
        len(
            selected_fnames
        ),
    )

    for fname in (
        selected_fnames
    ):
        print(
            " ",
            fname,
        )

    output_path = Path(
        args.output
    )

    cohort_path = (
        output_path.with_name(
            output_path.stem
            + "_cohort.txt"
        )
    )

    cohort_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with cohort_path.open(
        "w"
    ) as f:

        for fname in (
            selected_fnames
        ):
            f.write(
                fname
                + "\n"
            )

    if (
        args.resume
    ):
        rows = (
            load_existing_rows(
                output_path
            )
        )
    else:
        rows = []

    done = {
        (
            row[
                "fname"
            ],
            row[
                "method"
            ],
        )
        for row in rows
    }

    caps_session = (
        CAPSSession(
            acceleration=
                args.acceleration,

            device=
                device,
        )
        if "caps"
        in methods
        else None
    )

    for volume_number, fname in enumerate(
        selected_fnames,
        start=1,
    ):

        pending = [
            method
            for method in methods
            if (
                fname,
                method,
            )
            not in done
        ]

        if not pending:

            print(
                f"\n"
                f"[{volume_number}/"
                f"{len(selected_fnames)}] "
                f"{fname} "
                "already complete."
            )

            continue

        print()
        print(
            "=" * 100
        )

        print(
            f"[{volume_number}/"
            f"{len(selected_fnames)}] "
            f"{fname}"
        )

        print(
            "pending:",
            pending,
        )

        print(
            "=" * 100
        )

        indices = (
            volume_indices[
                fname
            ]
        )

        target_slices = []

        pred_slices = {
            method: []
            for method in pending
        }

        runtime_by_method = {
            method: 0.0
            for method in pending
        }

        caps_scales = []

        sampling_fractions = []

        acs_values = []

        espirit_runtime = 0.0

        need_sens = any(
            method in {
                "cg",
                "lacs",
                "nerp",
                "caps",
            }
            for method in pending
        )

        print(
            "slices evaluated:",
            len(
                indices
            ),
        )

        for slice_pos, idx in enumerate(
            indices,
            start=1,
        ):

            sample = dataset[
                idx
            ]

            print(
                f"  slice "
                f"{slice_pos}/"
                f"{len(indices)} "
                f"| dataset idx={idx} "
                f"| slice_num="
                f"{sample['slice_num']}"
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

            target_slices.append(
                target
                .detach()
                .cpu()
                .numpy()
            )

            mask = prepare_mask(
                raw_mask,
                width=
                    FASTMRI_256_SIZE,
                device=
                    device,
            )

            sampling_fractions.append(
                float(
                    mask.float()
                    .mean()
                    .item()
                )
            )

            acs_values.append(
                int(
                    num_low
                )
            )

            sens_maps = None

            if need_sens:

                sync()

                sens_start = (
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

                espirit_runtime += (
                    time.time()
                    - sens_start
                )

                if not torch.isfinite(
                    sens_maps
                ).all():

                    raise RuntimeError(
                        "Non-finite "
                        "sensitivity maps."
                    )

            if "zf" in pending:

                (
                    zf,
                    elapsed,
                ) = run_timed(
                    lambda:
                        zero_filled_rss(
                            masked_kspace,
                            (
                                FASTMRI_256_SIZE,
                                FASTMRI_256_SIZE,
                            ),
                        )
                        .to(
                            device
                        )
                )

                pred_slices[
                    "zf"
                ].append(
                    zf
                    .detach()
                    .cpu()
                    .numpy()
                )

                runtime_by_method[
                    "zf"
                ] += elapsed

                del zf

            if "cg" in pending:

                (
                    cg,
                    elapsed,
                ) = run_timed(
                    lambda:
                        cg_sense(
                            measured_kspace=
                                masked_kspace,
                            sens_maps=
                                sens_maps,
                            mask=
                                mask,
                            num_iters=
                                CG_ITERS,
                            lambda_reg=
                                CG_LAMBDA,
                        )
                )

                cg = (
                    cg.abs()
                )

                pred_slices[
                    "cg"
                ].append(
                    cg
                    .detach()
                    .cpu()
                    .numpy()
                )

                runtime_by_method[
                    "cg"
                ] += elapsed

                del cg

            if "lacs" in pending:

                (
                    lacs_result,
                    elapsed,
                ) = run_timed(
                    lambda:
                        reconstruct_lacs(
                            measured_kspace=
                                masked_kspace,
                            sens_maps=
                                sens_maps,
                            mask=
                                mask,
                            prior=
                                None,
                            wavelet_levels=
                                LACS_LEVELS,
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

                pred_slices[
                    "lacs"
                ].append(
                    lacs
                    .detach()
                    .cpu()
                    .numpy()
                )

                runtime_by_method[
                    "lacs"
                ] += elapsed

                del lacs
                del lacs_result

            if "nerp" in pending:

                (
                    nerp_result,
                    elapsed,
                ) = run_timed(
                    lambda:
                        reconstruct_nerp(
                            measured_kspace=
                                masked_kspace,
                            sens_maps=
                                sens_maps,
                            mask=
                                mask,
                            prior=
                                None,
                            reconstruction_iters=
                                NERP_ITERS,
                            reconstruction_learning_rate=
                                NERP_LR,
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

                pred_slices[
                    "nerp"
                ].append(
                    nerp
                    .detach()
                    .cpu()
                    .numpy()
                )

                runtime_by_method[
                    "nerp"
                ] += elapsed

                del nerp
                del nerp_result

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if "caps" in pending:

                caps_sample_seed = (
                    make_caps_sample_seed(
                        fname=
                            fname,

                        slice_num=
                            sample["slice_num"],

                        acceleration=
                            args.acceleration,
                    )
                )

                (
                    caps,
                    elapsed,
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
                            caps_sample_seed,
                    )
                )
                pred_slices[
                    "caps"
                ].append(
                    caps
                    .detach()
                    .cpu()
                    .numpy()
                )

                runtime_by_method[
                    "caps"
                ] += elapsed

                caps_scales.append(
                    caps_scale
                )

                del caps

            del masked_kspace
            del target
            del mask

            if (
                sens_maps
                is not None
            ):
                del sens_maps

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        #
        # ----------------------------------
        # Volume-level evaluation.
        # ----------------------------------
        #
        target_volume = (
            np.stack(
                target_slices,
                axis=0,
            )
        )

        #
        # One common data range for all
        # methods in this adapted volume.
        #
        max_value_256 = float(
            np.max(
                target_volume
            )
        )

        sampling_fraction = (
            float(
                np.mean(
                    sampling_fractions
                )
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
                "ACS count changed "
                "within volume: "
                f"{sorted(unique_acs)}"
            )

        acs_lines = (
            acs_values[0]
        )

        for method in pending:

            prediction_volume = (
                np.stack(
                    pred_slices[
                        method
                    ],
                    axis=0,
                )
            )

            metric_values = (
                compute_volume_metrics(
                    target=
                        target_volume,
                    prediction=
                        prediction_volume,
                    max_value=
                        max_value_256,
                )
            )

            caps_output_dc_iters = ""
            caps_lambda_ldm = ""
            caps_n_avgs = ""
            caps_tp = ""
            caps_seed_mode = ""
            caps_base_seed = ""
            caps_scale_mean = ""

            if method == "caps":

                caps_output_dc_iters = (
                    caps_session
                    .params
                    .output_dc_config[
                        "n_iters"
                    ]
                )

                caps_lambda_ldm = (
                    caps_session
                    .params
                    .output_dc_config[
                        "lambda_ldm"
                    ]
                )

                caps_n_avgs = (
                    caps_session
                    .params
                    .n_avgs
                )

                caps_tp = (
                    caps_session
                    .params
                    .prior_start_timestep
                )

                caps_seed_mode = (
                    "sha256_fname_slice_acceleration"
                )

                caps_base_seed = (
                    CAPS_BASE_SEED
                )
                
                caps_scale_mean = (
                    float(
                        np.mean(
                            caps_scales
                        )
                    )
                )

            row = {
                "fname":
                    fname,

                "method":
                    method,

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
                    sampling_fraction,

                "acs_lines":
                    acs_lines,

                "espirit_crop":
                    ESPIRIT_CROP,

                "espirit_runtime_sec":
                    espirit_runtime,

                "runtime_sec":
                    runtime_by_method[
                        method
                    ],

                "nmse":
                    metric_values[
                        "nmse"
                    ],

                "psnr":
                    metric_values[
                        "psnr"
                    ],

                "ssim":
                    metric_values[
                        "ssim"
                    ],

                "max_value_256":
                    max_value_256,

                "cg_iters":
                    (
                        CG_ITERS
                        if method
                        == "cg"
                        else ""
                    ),

                "cg_lambda":
                    (
                        CG_LAMBDA
                        if method
                        == "cg"
                        else ""
                    ),

                "lacs_wavelet_levels":
                    (
                        LACS_LEVELS
                        if method
                        == "lacs"
                        else ""
                    ),

                "nerp_iters":
                    (
                        NERP_ITERS
                        if method
                        == "nerp"
                        else ""
                    ),

                "nerp_lr":
                    (
                        NERP_LR
                        if method
                        == "nerp"
                        else ""
                    ),

                "caps_seed_mode":
                    caps_seed_mode,

                "caps_base_seed":
                    caps_base_seed,

                "caps_n_avgs":
                    caps_n_avgs,

                "caps_prior_start_timestep":
                    caps_tp,

                "caps_output_dc_iters":
                    caps_output_dc_iters,

                "caps_lambda_ldm":
                    caps_lambda_ldm,

                "caps_scale_mean":
                    caps_scale_mean,
            }

            rows.append(
                row
            )

            done.add(
                (
                    fname,
                    method,
                )
            )

            save_rows_atomic(
                output_path,
                rows,
            )

            print(
                f"\n"
                f"{method.upper():5s} "
                f"| NMSE="
                f"{metric_values['nmse']:.6f} "
                f"| PSNR="
                f"{metric_values['psnr']:.3f} "
                f"| SSIM="
                f"{metric_values['ssim']:.4f} "
                f"| runtime="
                f"{runtime_by_method[method]:.2f}s"
            )

        print(
            "\ncheckpoint saved:",
            output_path,
        )

    #
    # ----------------------------------
    # Final shard summary.
    # ----------------------------------
    #
    print()
    print(
        "=" * 100
    )

    print(
        "FINAL SHARD SUMMARY"
    )

    print(
        "=" * 100
    )

    selected_set = set(
        selected_fnames
    )

    for method in methods:

        method_rows = [
            row
            for row in rows
            if (
                row[
                    "fname"
                ]
                in selected_set
                and
                row[
                    "method"
                ]
                == method
            )
        ]

        if not method_rows:
            continue

        print(
            f"\n"
            f"{method.upper()} "
            f"({len(method_rows)} volumes)"
        )

        for metric in [
            "nmse",
            "psnr",
            "ssim",
        ]:

            values = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row
                    in method_rows
                ],
                dtype=np.float64,
            )

            print(
                f"  "
                f"{metric.upper():5s}: "
                f"{values.mean():.6f} "
                f"+/- "
                f"{values.std():.6f}"
            )

    print(
        "\nSaved:",
        output_path,
    )

    print(
        "Cohort:",
        cohort_path,
    )


if __name__ == "__main__":
    main()
