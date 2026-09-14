import copy
import gc
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

LAPS_ROOT = (
    PROJECT_ROOT
    / "external"
    / "laps_original"
)

#
# Make sure our patched vendored diffusers
# is used before site-packages.
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
)

from data.fastmri_256 import (
    FASTMRI_256_SIZE,
    build_fastmri_256_problem,
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

from laps.utils import (
    clear_cache,
)


DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 0

DATASET_INDEX = 532


def make_mask_2d(
    raw_mask,
    size,
    device,
):
    """
    Convert fastMRI broadcast mask to the
    2-D Cartesian mask expected by LAPS.
    """

    mask = (
        raw_mask
        .squeeze()
        .to(device)
    )

    if mask.ndim == 1:

        if mask.numel() != size:
            raise ValueError(
                "Unexpected 1-D mask size: "
                f"{tuple(mask.shape)}"
            )

        mask = (
            mask[None, :]
            .expand(
                size,
                size,
            )
        )

    elif mask.ndim == 2:

        if tuple(mask.shape) != (
            size,
            size,
        ):
            raise ValueError(
                "Unexpected 2-D mask shape: "
                f"{tuple(mask.shape)}"
            )

    else:
        raise ValueError(
            "Could not convert raw mask "
            f"with shape {tuple(raw_mask.shape)}"
        )

    return mask.to(
        torch.complex64
    )


def nmse(
    target,
    prediction,
):
    return (
        torch.sum(
            (
                target
                - prediction
            ) ** 2
        )
        /
        torch.sum(
            target ** 2
        )
    ).item()


def make_caps_params(
    acceleration,
):
    """
    Start from the official released CAPS
    configuration and only apply the same
    acceleration-dependent output-DC override
    used by official recon_test.py.
    """

    params = copy.deepcopy(
        recon_configs.sd_caps_medvae_4
    )

    params.device = DEVICE

    params.im_size = (
        FASTMRI_256_SIZE,
        FASTMRI_256_SIZE,
    )

    params.debug = False

    #
    # Official 1-D recon_test settings.
    #
    if acceleration >= 7:

        params.output_dc_config = {
            "n_iters": 5,
            "threshold": 1e-5,
            "lambda_l2": 5e-4,
            "lambda_ldm": 0.03,
            "lambda_l2_from_data": False,
            "avg_before_dc": False,
        }

    else:

        params.output_dc_config = {
            "n_iters": 10,
            "threshold": 1e-5,
            "lambda_l2": 5e-4,
            "lambda_ldm": 0.025,
            "lambda_l2_from_data": False,
            "avg_before_dc": False,
        }

    return params


def run_one(
    dataset,
    acceleration,
):
    print()
    print("=" * 100)
    print(
        f"CAPS FASTMRI-256 R={acceleration}"
    )
    print("=" * 100)

    sample = dataset[
        DATASET_INDEX
    ]

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
            DEVICE,
            dtype=torch.complex64,
        )
    )

    target = (
        sample_256[
            "target"
        ]
        .to(
            DEVICE,
            dtype=torch.float32,
        )
    )

    mask = make_mask_2d(
        raw_mask=
            raw_mask,

        size=
            FASTMRI_256_SIZE,

        device=
            DEVICE,
    )

    assert tuple(
        masked_kspace.shape
    ) == (
        16,
        256,
        256,
    )

    assert tuple(
        mask.shape
    ) == (
        256,
        256,
    )

    print(
        "file:",
        sample["fname"],
    )

    print(
        "slice:",
        sample["slice_num"],
    )

    print(
        "kspace:",
        tuple(
            masked_kspace.shape
        ),
    )

    print(
        "mask:",
        tuple(
            mask.shape
        ),
    )

    print(
        "sampling fraction:",
        mask.real.float()
        .mean()
        .item(),
    )

    print(
        "ACS lines:",
        num_low,
    )

    #
    # ----------------------------------
    # ESPIRiT on the SAME acquired data.
    # ----------------------------------
    #
    start = time.time()

    mps = (
        estimate_sens_espirit(
            masked_kspace,
            num_low_frequencies=
                num_low,
            crop=
                0.0,
        )
        .to(
            DEVICE,
            dtype=torch.complex64,
        )
    )

    espirit_runtime = (
        time.time()
        - start
    )

    print(
        "ESPIRiT runtime:",
        espirit_runtime,
    )

    print(
        "mps:",
        tuple(
            mps.shape
        ),
    )

    print(
        "finite mps:",
        torch.isfinite(
            mps
        ).all().item(),
    )

    assert tuple(
        mps.shape
    ) == (
        16,
        256,
        256,
    )

    assert torch.isfinite(
        mps
    ).all()

    #
    # ----------------------------------
    # Official LAPS/CAPS forward model.
    # ----------------------------------
    #
    A = CartesianSenseLinop(
        mps=mps,
        mask=mask,
        ishape=(
            FASTMRI_256_SIZE,
            FASTMRI_256_SIZE,
        ),
    )

    A = A.to(
        DEVICE
    )

    ahb = (
        A.H(
            masked_kspace
        )
        .abs()
    )

    scale = (
        ahb.quantile(
            0.999
        )
    )

    if (
        not torch.isfinite(scale)
        or scale.item() <= 0
    ):
        raise RuntimeError(
            f"Invalid LAPS normalization "
            f"scale: {scale}"
        )

    #
    # Match official recon_test normalization:
    # k-space is divided by the 99.9th
    # percentile of |A^H b|.
    #
    kspace_norm = (
        masked_kspace
        / scale
    )

    #
    # Evaluation target normalization from
    # official recon_test.
    #
    target_norm = (
        target
        / target.abs().max()
    )

    print(
        "AHb .999 scale:",
        scale.item(),
    )

    print(
        "normalized kspace finite:",
        torch.isfinite(
            kspace_norm
        ).all().item(),
    )

    #
    # Sensitivity support mask used by
    # official recon_test postprocessing.
    #
    sens_mask = (
        torch.linalg.vector_norm(
            mps,
            dim=0,
        )
        > 0.5
    )

    print(
        "sens support fraction:",
        sens_mask.float()
        .mean()
        .item(),
    )

    #
    # ----------------------------------
    # Official CAPS configuration.
    # ----------------------------------
    #
    params = make_caps_params(
        acceleration
    )

    print()
    print("--- CAPS CONFIG ---")

    print(
        "model:",
        params.model_name_or_path,
    )

    print(
        "DDIM steps:",
        params.num_inference_steps,
    )

    print(
        "dc type:",
        params.dc_type,
    )

    print(
        "latent nopt:",
        params.opt_params[
            "latent"
        ][
            "n_iters"
        ],
    )

    print(
        "latent lr:",
        params.opt_params[
            "latent"
        ][
            "lr"
        ],
    )

    print(
        "navg:",
        params.n_avgs,
    )

    print(
        "start with prior:",
        params.start_with_prior,
    )

    print(
        "start with CG:",
        params.start_with_cg,
    )

    print(
        "prior start timestep:",
        params.prior_start_timestep,
    )

    print(
        "output DC:",
        params.output_dc,
    )

    print(
        "output DC config:",
        params.output_dc_config,
    )

    assert (
        params.start_with_prior
        is False
    )

    assert (
        params.start_with_cg
        is True
    )

    assert (
        params.prior_start_timestep
        == 200
    )

    #
    # ----------------------------------
    # Instantiate official reconstructor.
    # ----------------------------------
    #
    print()
    print(
        "Instantiating CAPS..."
    )

    reconstructor = (
        StableDiffusionReconstructor(
            forward_model=A,
            params=params,
            load_model_to_cpu=False,
        )
    )

    #
    # Re-seed immediately before sampling.
    #
    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )
        torch.cuda.synchronize()

    print(
        "Starting CAPS reconstruction..."
    )

    start = time.time()

    #
    # IMPORTANT:
    #
    # No target and no fake prior are passed
    # into reconstruction.
    #
    # CAPS creates its own CG initialization
    # from measurements internally.
    #
    output = (
        reconstructor.reconstruct(
            kspace_norm[
                None,
                ...
            ]
        )
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime = (
        time.time()
        - start
    )

    recon = (
        output.recon[
            0
        ]
        .detach()
        .to(
            DEVICE
        )
    )

    recon = (
        recon
        * sens_mask
    )

    #
    # ----------------------------------
    # Two evaluation conventions.
    # ----------------------------------
    #
    # 1. Official-LAPS-style normalized
    #    representation used in this gate.
    #
    recon_mag_norm = (
        recon.abs()
    )

    result_nmse_normalized = (
        nmse(
            target_norm,
            recon_mag_norm,
        )
    )

    #
    # 2. Convert the reconstruction back
    #    to the original fastMRI image scale.
    #
    # We fed:
    #
    #     y_norm = y_raw / scale
    #
    # to a linear MRI forward model, so the
    # reconstructed image is in units
    #
    #     x_norm ~= x_raw / scale.
    #
    # Therefore:
    #
    #     x_raw = x_norm * scale
    #
    # This is the metric convention that
    # must be used for the unified
    # fastMRI-256 benchmark.
    #
    recon_mag_raw = (
        recon_mag_norm
        * scale
    )

    result_nmse_raw = (
        nmse(
            target,
            recon_mag_raw,
        )
    )

    target_max = (
        target.abs()
        .max()
    )

    scale_ratio = (
        scale
        / target_max
    )

    print()
    print("--- RESULT ---")

    print(
        "reconstruction:",
        tuple(
            recon.shape
        ),
    )

    print(
        "dtype:",
        recon.dtype,
    )

    print(
        "finite:",
        torch.isfinite(
            recon
        ).all().item(),
    )

    print(
        "max abs:",
        recon.abs()
        .max()
        .item(),
    )

    print(
        "runtime sec:",
        runtime,
    )

    print(
        "target max:",
        target_max.item(),
    )

    print(
        "AHb scale / target max:",
        scale_ratio.item(),
    )

    print(
        "normalized-space NMSE:",
        result_nmse_normalized,
    )

    print(
        "raw-scale NMSE:",
        result_nmse_raw,
    )

    print(
        "extra outputs:",
        output.extra_outputs,
    )

    assert tuple(
        recon.shape
    ) == (
        256,
        256,
    )

    assert torch.isfinite(
        recon
    ).all()

    assert torch.isfinite(
        torch.tensor(
            result_nmse_normalized
        )
    )

    assert torch.isfinite(
        torch.tensor(
            result_nmse_raw
        )
    )

    assert torch.isfinite(
        recon_mag_raw
    ).all()

    del reconstructor
    del output
    del A
    del mps

    clear_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    print()
    print(
        f"CAPS FASTMRI-256 "
        f"R={acceleration} PASSED"
    )


def main():
    print("=" * 100)
    print(
        "OFFICIAL CAPS × FASTMRI-256 "
        "SINGLE-SLICE GATE"
    )
    print("=" * 100)

    print(
        "device:",
        DEVICE,
    )

    dataset = (
        FastMRIBrainDataset(
            root=
                "extracted/"
                "multicoil_train",

            acquisition=
                "AXT2",

            cache_dir=
                "code/.cache",
        )
    )

    sample = dataset[
        DATASET_INDEX
    ]

    print(
        "dataset index:",
        DATASET_INDEX,
    )

    print(
        "file:",
        sample["fname"],
    )

    print(
        "slice:",
        sample["slice_num"],
    )

    assert (
        sample["fname"]
        ==
        "file_brain_AXT2_200_2000080.h5"
    )

    assert (
        int(
            sample["slice_num"]
        )
        == 8
    )

    for acceleration in [
        4,
        8,
    ]:
        run_one(
            dataset,
            acceleration,
        )

    print()
    print("=" * 100)
    print(
        "CAPS FASTMRI-256 "
        "SINGLE-SLICE GATE PASSED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
