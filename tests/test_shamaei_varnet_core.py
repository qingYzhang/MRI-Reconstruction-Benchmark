import sys
from pathlib import Path

import torch


CODE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(CODE_ROOT)
)


from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
    fastmri_fft2,
    fastmri_ifft2,
    complex_to_direct,
    direct_to_complex,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 80)
    print("SHAMAEI E2E-VARNET CORE UNIT TEST")
    print("=" * 80)

    print(
        "device:",
        DEVICE
    )

    torch.manual_seed(
        0
    )

    H = 64
    W = 64

    #
    # ----------------------------------------------------------
    # Verify our DIRECT-format centered FFT adapter first.
    # ----------------------------------------------------------
    #

    z = torch.randn(
        1,
        1,
        H,
        W,
        device=DEVICE,
    ) + 1j * torch.randn(
        1,
        1,
        H,
        W,
        device=DEVICE,
    )

    z = z.to(
        torch.complex64
    )

    z_ri = complex_to_direct(
        z
    )

    recovered = direct_to_complex(
        fastmri_ifft2(
            fastmri_fft2(
                z_ri
            )
        )
    )

    fft_error = (
        torch.linalg.vector_norm(
            recovered - z
        )
        /
        torch.linalg.vector_norm(
            z
        )
    ).item()

    print(
        "FFT inverse relative error:",
        fft_error
    )

    #
    # ----------------------------------------------------------
    # Synthetic MRI.
    # ----------------------------------------------------------
    #

    axis = torch.linspace(
        -1.0,
        1.0,
        H,
        device=DEVICE,
    )

    y, x = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    image = torch.exp(
        -(
            x ** 2 / 0.18
            +
            y ** 2 / 0.30
        )
    )

    image = image.to(
        torch.complex64
    )

    image = image[
        None,
        None,
        ...,
    ]

    #
    # Single-coil normalized sensitivity.
    #
    sens = torch.ones(
        1,
        1,
        H,
        W,
        dtype=torch.complex64,
        device=DEVICE,
    )

    coil_image = (
        image * sens
    )

    full_kspace = (
        direct_to_complex(
            fastmri_fft2(
                complex_to_direct(
                    coil_image
                )
            )
        )
    )

    #
    # R≈4 Cartesian mask + ACS.
    #
    mask = torch.zeros(
        1,
        1,
        H,
        W,
        dtype=torch.bool,
        device=DEVICE,
    )

    mask[
        ...,
        :,
        ::4
    ] = True

    center = W // 2

    mask[
        ...,
        :,
        center - 4:
        center + 4,
    ] = True

    masked_kspace = (
        full_kspace
        * mask
    )

    print(
        "sampling fraction:",
        mask.float()
        .mean()
        .item()
    )

    #
    # ----------------------------------------------------------
    # Source-faithful architecture.
    # ----------------------------------------------------------
    #

    model = (
        ShamaeiE2EVarNetFastMRI(
            num_layers=12,
            regularizer_num_filters=18,
            regularizer_num_pull_layers=4,
            regularizer_dropout=0.0,
        )
        .to(DEVICE)
    )

    print(
        "number of cascades:",
        len(
            model
            .model
            .layers_list
        )
    )

    learning_rates = [
        float(
            layer
            .learning_rate
            .detach()
            .cpu()
            .item()
        )
        for layer
        in model
        .model
        .layers_list
    ]

    print(
        "initial DC learning rates:",
        learning_rates
    )

    #
    # Forward.
    #
    rss, output_kspace = (
        model.reconstruct_rss(
            masked_kspace=
                masked_kspace,

            mask=
                mask,

            sensitivity_maps=
                sens,
        )
    )

    print(
        "output kspace shape:",
        tuple(
            output_kspace.shape
        )
    )

    print(
        "RSS shape:",
        tuple(
            rss.shape
        )
    )

    print(
        "finite kspace:",
        torch.isfinite(
            output_kspace
        ).all().item()
    )

    print(
        "finite RSS:",
        torch.isfinite(
            rss
        ).all().item()
    )

    #
    # ----------------------------------------------------------
    # Backward path.
    #
    # Functionality test only.
    # ----------------------------------------------------------
    #

    loss = torch.mean(
        (
            rss
            - torch.abs(
                image[:, 0]
            )
        ) ** 2
    )

    loss.backward()

    finite_grads = all(
        (
            parameter.grad
            is None
        )
        or torch.isfinite(
            parameter.grad
        ).all().item()
        for parameter
        in model.parameters()
    )

    grad_count = sum(
        parameter.grad
        is not None
        for parameter
        in model.parameters()
    )

    print(
        "test loss:",
        loss.item()
    )

    print(
        "parameters with gradient:",
        grad_count
    )

    print(
        "finite gradients:",
        finite_grads
    )

    print(
        "forward passed:",
        (
            output_kspace.shape
            == masked_kspace.shape
            and rss.shape
            == (1, H, W)
        )
    )


if __name__ == "__main__":
    main()
