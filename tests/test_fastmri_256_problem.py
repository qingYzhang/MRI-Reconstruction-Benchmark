import sys
from pathlib import Path

import torch

from fastmri.data.transforms import (
    center_crop,
)


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
    apply_cartesian_mask,
    zero_filled_rss,
)

from operators.sense import (
    fft2c,
    ifft2c,
    prepare_mask,
    cg_sense,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ADAPTED_SIZE = 256


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


def main():

    dataset = FastMRIBrainDataset(
        root=
            "extracted/multicoil_train",

        acquisition=
            "AXT2",

        cache_dir=
            "code/.cache",
    )

    #
    # Same frozen smoke slice used
    # throughout the benchmark.
    #
    sample = dataset[532]

    full_kspace_native = (
        sample["kspace"]
        .to(DEVICE)
    )

    target_native = (
        sample["target"]
        .to(DEVICE)
    )

    print("=" * 80)
    print("FASTMRI 256 ADAPTATION TEST")
    print("=" * 80)

    print(
        "file:",
        sample["fname"],
    )

    print(
        "slice:",
        sample["slice_num"],
    )

    print(
        "native kspace:",
        tuple(
            full_kspace_native.shape
        ),
    )

    print(
        "native target:",
        tuple(
            target_native.shape
        ),
    )

    #
    # ----------------------------------
    # Native k-space -> coil images.
    # ----------------------------------
    #
    coil_images_native = (
        ifft2c(
            full_kspace_native
        )
    )

    #
    # IMPORTANT:
    #
    # Crop in IMAGE domain, not
    # directly in k-space.
    #
    coil_images_256 = (
        center_crop(
            coil_images_native,
            (
                ADAPTED_SIZE,
                ADAPTED_SIZE,
            ),
        )
    )

    #
    # Construct a self-consistent
    # 256x256 fully sampled acquisition.
    #
    full_kspace_256 = (
        fft2c(
            coil_images_256
        )
    )

    recovered_coils = (
        ifft2c(
            full_kspace_256
        )
    )

    fft_roundtrip_error = (
        torch.linalg.norm(
            recovered_coils
            - coil_images_256
        )
        /
        torch.linalg.norm(
            coil_images_256
        )
    ).item()

    #
    # fastMRI target is RSS.
    #
    rss_256 = torch.sqrt(
        torch.sum(
            torch.abs(
                coil_images_256
            ) ** 2,
            dim=0,
        )
    )

    target_256 = (
        center_crop(
            target_native,
            (
                ADAPTED_SIZE,
                ADAPTED_SIZE,
            ),
        )
    )

    target_rss_error = (
        torch.linalg.norm(
            rss_256
            - target_256
        )
        /
        torch.linalg.norm(
            target_256
        )
    ).item()

    print()
    print(
        "adapted coil images:",
        tuple(
            coil_images_256.shape
        ),
    )

    print(
        "adapted full kspace:",
        tuple(
            full_kspace_256.shape
        ),
    )

    print(
        "adapted target:",
        tuple(
            target_256.shape
        ),
    )

    print(
        "FFT roundtrip rel error:",
        fft_roundtrip_error,
    )

    print(
        "RSS vs target crop rel error:",
        target_rss_error,
    )

    assert (
        tuple(
            full_kspace_256.shape[-2:]
        )
        ==
        (
            ADAPTED_SIZE,
            ADAPTED_SIZE,
        )
    )

    assert (
        fft_roundtrip_error
        < 1e-5
    )

    #
    # Build a sample compatible with
    # our existing deterministic mask
    # function.
    #
    sample_256 = dict(
        sample
    )

    sample_256[
        "kspace"
    ] = (
        full_kspace_256
        .detach()
        .cpu()
    )

    sample_256[
        "target"
    ] = (
        target_256
        .detach()
        .cpu()
    )

    #
    # The synthetic grid contains no
    # encoded readout padding.
    #
    sample_256[
        "padding_left"
    ] = 0

    sample_256[
        "padding_right"
    ] = ADAPTED_SIZE

    sample_256[
        "encoding_size"
    ] = (
        ADAPTED_SIZE,
        ADAPTED_SIZE,
    )

    sample_256[
        "recon_size"
    ] = (
        ADAPTED_SIZE,
        ADAPTED_SIZE,
    )

    for acceleration in [
        4,
        8,
    ]:

        print()
        print("=" * 80)
        print(
            f"R={acceleration}"
        )
        print("=" * 80)

        (
            masked_kspace,
            raw_mask,
            num_low,
        ) = apply_cartesian_mask(
            sample_256,
            acceleration,
        )

        masked_kspace = (
            masked_kspace
            .to(DEVICE)
        )

        mask = prepare_mask(
            raw_mask,
            width=
                masked_kspace.shape[-1],
            device=
                DEVICE,
        )

        sampling_fraction = (
            mask
            .float()
            .mean()
            .item()
        )

        print(
            "masked kspace:",
            tuple(
                masked_kspace.shape
            ),
        )

        print(
            "sampling fraction:",
            sampling_fraction,
        )

        print(
            "ACS lines:",
            num_low,
        )

        sens_maps = (
            estimate_sens_espirit(
                masked_kspace,
                num_low_frequencies=
                    num_low,
                crop=
                    0.0,
            )
        )

        print(
            "sens maps:",
            tuple(
                sens_maps.shape
            ),
        )

        print(
            "finite sens:",
            torch.isfinite(
                sens_maps
            ).all().item(),
        )

        zf = (
            zero_filled_rss(
                masked_kspace,
                (
                    ADAPTED_SIZE,
                    ADAPTED_SIZE,
                ),
            )
            .to(DEVICE)
        )

        cg = cg_sense(
            measured_kspace=
                masked_kspace,

            sens_maps=
                sens_maps,

            mask=
                mask,

            num_iters=
                50,

            lambda_reg=
                0.01,
        )

        cg = torch.abs(
            cg
        )

        print(
            "ZF NMSE:",
            nmse(
                target_256,
                zf,
            ),
        )

        print(
            "CG NMSE:",
            nmse(
                target_256,
                cg,
            ),
        )

        assert torch.isfinite(
            sens_maps
        ).all()

        assert torch.isfinite(
            cg
        ).all()

    print()
    print(
        "FASTMRI 256 PROBLEM "
        "CONSTRUCTION PASSED"
    )


if __name__ == "__main__":
    main()
