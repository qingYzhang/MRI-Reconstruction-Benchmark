import torch

from methods.lacs.wavelab import (
    wavelab_d4_forward,
)


@torch.no_grad()
def compute_official_lacs_weights(
    reconstruction: torch.Tensor,
    prior: torch.Tensor,
    wavelet_levels: int = 2,
    epsilon1: float = 0.1,
):
    """
    Port of the weight update in
    follow_up_application.m.

    Original source logic:

        reconstructed_image = abs(W' * X)

        diff_in_wav =
            abs(
                W * (
                    reconstructed_image - im0
                )
            )

        map_diff =
            diff_in_wav /
            (diff_in_wav + 1)

        W1(map_diff > epsilon1) = 1

        W1(map_diff <= epsilon1) =
            1 / (1 + abs(X0))

        W2 =
            1 / (
                1
                + abs(
                    reconstructed_image
                    - im0
                )
            )

        gamma = mean(W2(:))

    Inputs are assumed to already be
    on the same intensity scale.
    """

    if reconstruction.shape != prior.shape:
        raise ValueError(
            f"reconstruction shape "
            f"{reconstruction.shape} "
            f"!= prior shape "
            f"{prior.shape}"
        )

    #
    # Original code works from magnitude
    # reconstructed image.
    #
    reconstructed_image = torch.abs(
        reconstruction
    )

    prior_image = (
        torch.abs(prior)
        if torch.is_complex(prior)
        else prior
    )

    #
    # X0 = W * im0
    #
    x0, _ = wavelab_d4_forward(
        prior_image,
        levels=wavelet_levels,
    )

    inverse_wav_baseline = (
        1.0
        /
        (
            1.0
            + torch.abs(x0)
        )
    )

    #
    # diff_in_wav =
    # abs(W * (reconstructed_image-im0))
    #
    diff_in_wav, _ = (
        wavelab_d4_forward(
            reconstructed_image
            - prior_image,
            levels=wavelet_levels,
        )
    )

    diff_in_wav = torch.abs(
        diff_in_wav
    )

    map_diff = (
        diff_in_wav
        /
        (
            diff_in_wav
            + 1.0
        )
    )

    w1 = torch.ones_like(
        map_diff
    )

    similar = (
        map_diff
        <= epsilon1
    )

    w1[similar] = (
        inverse_wav_baseline[
            similar
        ]
    )

    #
    # W2 image-domain weight.
    #
    w2 = (
        1.0
        /
        (
            1.0
            + torch.abs(
                reconstructed_image
                - prior_image
            )
        )
    )

    gamma = torch.mean(
        w2
    )

    diagnostics = {
        "w1_mean":
            float(
                w1.mean().item()
            ),

        "w1_min":
            float(
                w1.min().item()
            ),

        "w1_max":
            float(
                w1.max().item()
            ),

        "w2_mean":
            float(
                w2.mean().item()
            ),

        "w2_min":
            float(
                w2.min().item()
            ),

        "w2_max":
            float(
                w2.max().item()
            ),

        "different_wavelet_fraction":
            float(
                (
                    map_diff
                    > epsilon1
                )
                .float()
                .mean()
                .item()
            ),

        "gamma":
            float(
                gamma.item()
            ),
    }

    return (
        w1,
        w2,
        diagnostics,
    )
