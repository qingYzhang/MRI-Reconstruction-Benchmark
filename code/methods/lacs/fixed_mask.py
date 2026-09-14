import torch

from methods.lacs.official_fista import (
    official_fista_reconstruct,
    estimate_official_scale,
)

from methods.lacs.official_weights import (
    compute_official_lacs_weights,
)


@torch.no_grad()
def lacs_fixed_mask(
    measured_kspace,
    sens_maps,
    mask,
    x_prior,
    wavelet_levels=2,
    outer_rounds=3,
    epsilon1=0.1,
    weight_mode="adaptive",
    verbose=False,
):
    """
    Fixed-mask reconstruction-side adaptation
    of the original LACS algorithm.

    Original LACS:
        round 1:
            W1 = I
            W2 = 0

        reconstruct
        update W1/W2

        round 2:
            weighted reconstruction

        update W1/W2

        round 3:
            weighted reconstruction

    Here all rounds use the same fixed
    fastMRI sampling mask so that LACS can be
    compared fairly with other reconstruction
    methods.

    weight_mode:
        "adaptive"
            official W1/W2 updates

        "fixed"
            first round uses W1=I,W2=0;
            later rounds use W1=I,W2=I

        "no_prior"
            all rounds use W1=I,W2=0
    """

    if outer_rounds < 1:
        raise ValueError(
            "outer_rounds must be >= 1"
        )

    if weight_mode not in {
        "adaptive",
        "fixed",
        "no_prior",
    }:
        raise ValueError(
            f"Unknown weight_mode: "
            f"{weight_mode}"
        )

    device = measured_kspace.device

    image_shape = (
        measured_kspace.shape[-2],
        measured_kspace.shape[-1],
    )

    if tuple(x_prior.shape) != image_shape:
        raise ValueError(
            f"Prior shape "
            f"{x_prior.shape} "
            f"!= image shape "
            f"{image_shape}"
        )

    x_prior = x_prior.to(
        device=device
    )

    #
    # One common intensity scale.
    #
    # This is important because original
    # epsilon1=0.1 assumes baseline/follow-up
    # images are on a comparable normalized
    # gray-level scale.
    #
    scale = estimate_official_scale(
        measured_kspace,
        sens_maps,
        mask,
    )

    prior_normalized = (
        x_prior / scale
    )

    real_dtype = (
        measured_kspace.real.dtype
    )

    #
    # Official initialization.
    #
    w1 = torch.ones(
        image_shape,
        dtype=real_dtype,
        device=device,
    )

    w2 = torch.zeros(
        image_shape,
        dtype=real_dtype,
        device=device,
    )

    round_results = []

    reconstruction = None

    for round_idx in range(
        outer_rounds
    ):

        if verbose:

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"LACS OUTER ROUND "
                f"{round_idx + 1}/"
                f"{outer_rounds}"
            )

            print(
                "=" * 80
            )

            print(
                "W1 mean:",
                w1.mean().item()
            )

            print(
                "W2 mean:",
                w2.mean().item()
            )

        reconstruction, history = (
            official_fista_reconstruct(
                measured_kspace=
                    measured_kspace,

                sens_maps=
                    sens_maps,

                mask=
                    mask,

                x_prior=
                    x_prior,

                w1=
                    w1,

                w2=
                    w2,

                wavelet_levels=
                    wavelet_levels,

                num_iters=
                    20,

                L=
                    100.0,

                beta=
                    0.8,

                beta_prior=
                    0.8,

                beta_wavelet=
                    0.8,

                lambda_prior_initial=
                    30.0,

                lambda_wavelet_initial=
                    5.0,

                lambda_prior_floor=
                    0.158,

                lambda_wavelet_floor=
                    0.158,

                mu_initial=
                    0.1,

                mu_floor=
                    0.025,

                use_hard_dc=
                    True,

                verbose=
                    False,
            )
        )

        round_entry = {
            "round":
                round_idx + 1,

            "w1_mean_before":
                float(
                    w1.mean().item()
                ),

            "w2_mean_before":
                float(
                    w2.mean().item()
                ),

            "fista_history":
                history,

            "weight_diagnostics_after":
                None,
        }

        #
        # No need to update weights
        # after the final reconstruction.
        #
        if (
            round_idx
            < outer_rounds - 1
        ):

            if weight_mode == "adaptive":

                reconstruction_normalized = (
                    reconstruction
                    / scale
                )

                (
                    w1,
                    w2,
                    diagnostics,
                ) = (
                    compute_official_lacs_weights(
                        reconstruction=
                            reconstruction_normalized,

                        prior=
                            prior_normalized,

                        wavelet_levels=
                            wavelet_levels,

                        epsilon1=
                            epsilon1,
                    )
                )

                round_entry[
                    "weight_diagnostics_after"
                ] = diagnostics

            elif weight_mode == "fixed":

                w1 = torch.ones(
                    image_shape,
                    dtype=real_dtype,
                    device=device,
                )

                w2 = torch.ones(
                    image_shape,
                    dtype=real_dtype,
                    device=device,
                )

                round_entry[
                    "weight_diagnostics_after"
                ] = {
                    "w1_mean": 1.0,
                    "w2_mean": 1.0,
                    "gamma": 1.0,
                }

            elif weight_mode == "no_prior":

                w1 = torch.ones(
                    image_shape,
                    dtype=real_dtype,
                    device=device,
                )

                w2 = torch.zeros(
                    image_shape,
                    dtype=real_dtype,
                    device=device,
                )

                round_entry[
                    "weight_diagnostics_after"
                ] = {
                    "w1_mean": 1.0,
                    "w2_mean": 0.0,
                    "gamma": 0.0,
                }

        round_results.append(
            round_entry
        )

    return {
        "reconstruction":
            reconstruction,

        "rounds":
            round_results,

        "scale":
            float(scale.item()),
    }
