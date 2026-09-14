import math

import torch

from operators.sense import (
    fft2c,
    ifft2c,
    sense_forward,
    sense_adjoint,
)

from methods.lacs.wavelab import (
    wavelab_d4_forward,
    wavelab_d4_inverse,
)


def complex_soft_threshold(
    x: torch.Tensor,
    threshold,
    eps: float = 1e-12,
):
    """
    Complex soft threshold corresponding to
    the MATLAB SoftThresh behavior.
    """

    magnitude = torch.abs(
        x
    )

    shrink = torch.clamp(
        magnitude
        - threshold,
        min=0.0,
    )

    return (
        shrink
        /
        torch.clamp(
            magnitude,
            min=eps,
        )
        * x
    )


@torch.no_grad()
def estimate_official_scale(
    measured_kspace,
    sens_maps,
    mask,
    eps: float = 1e-8,
):
    """
    fastMRI adaptation only.

    The original demo images are already
    intensity-normalized.

    fastMRI raw data are not, so normalize
    from acquired measurements only.

    No fully sampled target is used.
    """

    adjoint = sense_adjoint(
        measured_kspace,
        sens_maps,
        mask,
    )

    magnitude = torch.abs(
        adjoint
    ).reshape(-1)

    scale = torch.quantile(
        magnitude,
        0.99,
    )

    return torch.clamp(
        scale,
        min=eps,
    )


@torch.no_grad()
def relative_data_residual(
    image,
    measured_kspace,
    sens_maps,
    mask,
):
    predicted = sense_forward(
        image,
        sens_maps,
        mask,
    )

    numerator = torch.linalg.norm(
        predicted
        - measured_kspace
    )

    denominator = torch.linalg.norm(
        measured_kspace
    )

    return (
        numerator
        /
        (
            denominator
            + 1e-12
        )
    )


@torch.no_grad()
def source_style_multicoil_dc(
    image,
    measured_kspace,
    sens_maps,
    mask,
    eps: float = 1e-8,
):
    """
    Multi-coil analogue of the original
    MATLAB hard Fourier data-consistency step:

        temp_Y = F*(W'*X)*F'
        temp_Y(mask) = Y(mask)
        X = W*(F'*temp_Y*F)

    The original method is single-coil, so
    hard replacement is exact there.

    Here:

        1. synthesize all coil k-space
        2. replace acquired samples
        3. inverse FFT coils
        4. SENSE-combine back to one image

    This is the closest direct multi-coil
    POCS analogue, but it is not mathematically
    identical to an exact constrained
    multi-coil SENSE projection.
    """

    coil_images = (
        sens_maps
        * image.unsqueeze(0)
    )

    predicted_kspace = fft2c(
        coil_images
    )

    mask_bool = mask.to(
        dtype=torch.bool,
        device=image.device,
    )

    kspace_dc = torch.where(
        mask_bool,
        measured_kspace,
        predicted_kspace,
    )

    coil_images_dc = ifft2c(
        kspace_dc
    )

    numerator = torch.sum(
        torch.conj(sens_maps)
        * coil_images_dc,
        dim=0,
    )

    denominator = torch.sum(
        torch.abs(sens_maps) ** 2,
        dim=0,
    )

    image_dc = (
        numerator
        /
        torch.clamp(
            denominator,
            min=eps,
        )
    )

    return image_dc


@torch.no_grad()
def official_fista_reconstruct(
    measured_kspace,
    sens_maps,
    mask,
    x_prior=None,
    w1=None,
    w2=None,
    wavelet_levels=2,

    num_iters=20,

    L=100.0,

    beta=0.8,
    beta_prior=0.8,
    beta_wavelet=0.8,

    lambda_prior_initial=30.0,
    lambda_wavelet_initial=5.0,

    lambda_prior_floor=0.158,
    lambda_wavelet_floor=0.158,

    mu_initial=0.1,
    mu_floor=0.025,

    use_hard_dc=True,
    verbose=False,
):
    """
    Source-faithful port of
    FISTA_based_solver.m, adapted to
    multi-coil fastMRI SENSE physics.

    IMPORTANT naming:

        original MATLAB "lambda"
            -> prior-difference term

        original MATLAB "lambda2"
            -> weighted wavelet term

    Original optimization variable X is
    wavelet coefficients:

        X = Psi x

        x = Psi^H X

    This function preserves:

        - coefficient-domain optimization
        - Nesterov/FISTA momentum
        - smoothed weighted penalties
        - lambda / mu continuation schedules
        - final soft threshold
        - source-style hard DC step

    Multi-coil SENSE and rectangular 2-D
    wavelets are intentional adaptations.
    """

    device = measured_kspace.device

    #
    # ------------------------------------------------
    # Normalize fastMRI intensity.
    # ------------------------------------------------
    #

    scale = estimate_official_scale(
        measured_kspace,
        sens_maps,
        mask,
    )

    y = (
        measured_kspace
        / scale
    )

    #
    # ------------------------------------------------
    # Prior.
    # ------------------------------------------------
    #

    image_shape = (
        measured_kspace.shape[-2],
        measured_kspace.shape[-1],
    )

    if x_prior is None:

        prior = torch.zeros(
            image_shape,
            dtype=measured_kspace.dtype,
            device=device,
        )

    else:

        if tuple(
            x_prior.shape
        ) != image_shape:

            raise ValueError(
                f"x_prior shape "
                f"{x_prior.shape} "
                f"!= image shape "
                f"{image_shape}"
            )

        prior = (
            x_prior
            .to(device)
            / scale
        )

    x0, wavelet_meta = (
        wavelab_d4_forward(
            prior,
            levels=wavelet_levels,
        )
    )

    #
    # ------------------------------------------------
    # Initial weights.
    #
    # Original first LACS iteration:
    #
    # W1 = ones
    # W2 = zeros
    # ------------------------------------------------
    #

    if w1 is None:

        w1 = torch.ones(
            image_shape,
            dtype=measured_kspace.real.dtype,
            device=device,
        )

    else:

        w1 = w1.to(
            device=device,
            dtype=measured_kspace.real.dtype,
        )

    if w2 is None:

        w2 = torch.zeros(
            image_shape,
            dtype=measured_kspace.real.dtype,
            device=device,
        )

    else:

        w2 = w2.to(
            device=device,
            dtype=measured_kspace.real.dtype,
        )

    if tuple(w1.shape) != image_shape:
        raise ValueError(
            f"W1 shape {w1.shape} "
            f"!= image shape {image_shape}"
        )

    if tuple(w2.shape) != image_shape:
        raise ValueError(
            f"W2 shape {w2.shape} "
            f"!= image shape {image_shape}"
        )

    #
    # ------------------------------------------------
    # MATLAB:
    #
    # X = W*(F'*Y*F)
    #
    # Multi-coil analogue:
    #
    # X = Psi A^H y
    # ------------------------------------------------
    #

    image_init = sense_adjoint(
        y,
        sens_maps,
        mask,
    )

    X, wavelet_meta = (
        wavelab_d4_forward(
            image_init,
            levels=wavelet_levels,
        )
    )

    X_k_m1 = X.clone()

    t_k = 1.0
    t_k_m1 = 1.0

    lambda_prior = float(
        lambda_prior_initial
    )

    lambda_wavelet = float(
        lambda_wavelet_initial
    )

    mu = float(
        mu_initial
    )

    history = []

    for iteration in range(
        num_iters
    ):

        #
        # ------------------------------------------------
        # Original:
        #
        # Z = X +
        #     ((t_{k-1}-1)/t_k)
        #     * (X-X_{k-1})
        # ------------------------------------------------
        #

        momentum = (
            (t_k_m1 - 1.0)
            / t_k
        )

        Z = (
            X
            + momentum
            * (
                X
                - X_k_m1
            )
        )

        #
        # ------------------------------------------------
        # Prior regularizer.
        #
        # MATLAB:
        #
        # temp_val1 =
        #   W2 .* (W'*(Z-X0))
        # ------------------------------------------------
        #

        temp_val1 = (
            w2
            * wavelab_d4_inverse(
                Z - x0,
                wavelet_meta,
            )
        )

        big_lambda = (
            complex_soft_threshold(
                temp_val1,
                threshold=(
                    lambda_prior
                    * mu
                ),
            )
        )

        #
        # ------------------------------------------------
        # Wavelet regularizer.
        #
        # MATLAB:
        #
        # temp_val2 = W1 .* Z
        # ------------------------------------------------
        #

        temp_val2 = (
            w1 * Z
        )

        big_lambda2 = (
            complex_soft_threshold(
                temp_val2,
                threshold=(
                    lambda_wavelet
                    * mu
                ),
            )
        )

        #
        # ------------------------------------------------
        # Data fidelity gradient.
        #
        # IMPORTANT:
        #
        # Original source uses X here,
        # not Z.
        #
        # Multi-coil analogue:
        #
        # Psi A^H(A Psi^H X - y)
        # ------------------------------------------------
        #

        image_x = (
            wavelab_d4_inverse(
                X,
                wavelet_meta,
            )
        )

        residual = (
            sense_forward(
                image_x,
                sens_maps,
                mask,
            )
            - y
        )

        data_gradient_image = (
            sense_adjoint(
                residual,
                sens_maps,
                mask,
            )
        )

        data_gradient, _ = (
            wavelab_d4_forward(
                data_gradient_image,
                levels=wavelet_levels,
            )
        )

        #
        # ------------------------------------------------
        # Exact source-code ordering.
        #
        # MATLAB:
        #
        # W2 .* (
        #   W*(temp_val1-big_lambda)
        # )
        #
        # Note this is the literal source
        # implementation.
        # ------------------------------------------------
        #

        prior_gradient_coeff, _ = (
            wavelab_d4_forward(
                temp_val1
                - big_lambda,
                levels=wavelet_levels,
            )
        )

        prior_gradient = (
            w2
            * prior_gradient_coeff
        )

        wavelet_gradient = (
            w1
            * (
                temp_val2
                - big_lambda2
            )
        )

        #
        # ------------------------------------------------
        # U update.
        # ------------------------------------------------
        #

        U = (
            Z
            - (
                1.0 / L
            )
            * (
                data_gradient
                +
                (
                    1.0 / mu
                )
                * (
                    prior_gradient
                    + wavelet_gradient
                )
            )
        )

        #
        # ------------------------------------------------
        # Original additional soft threshold:
        #
        # threshold = mu / L
        # ------------------------------------------------
        #

        X_candidate = (
            complex_soft_threshold(
                U,
                threshold=(
                    mu / L
                ),
            )
        )

        candidate_image = (
            wavelab_d4_inverse(
                X_candidate,
                wavelet_meta,
            )
        )

        residual_before_dc = (
            relative_data_residual(
                candidate_image,
                y,
                sens_maps,
                mask,
            )
        )

        #
        # ------------------------------------------------
        # Source-style hard DC.
        # ------------------------------------------------
        #

        if use_hard_dc:

            image_dc = (
                source_style_multicoil_dc(
                    candidate_image,
                    y,
                    sens_maps,
                    mask,
                )
            )

        else:

            image_dc = candidate_image

        residual_after_dc = (
            relative_data_residual(
                image_dc,
                y,
                sens_maps,
                mask,
            )
        )

        X_new, wavelet_meta = (
            wavelab_d4_forward(
                image_dc,
                levels=wavelet_levels,
            )
        )

        #
        # ------------------------------------------------
        # FISTA state update.
        # ------------------------------------------------
        #

        X_k_m1 = X

        X = X_new

        t_k_m1 = t_k

        t_k = (
            1.0
            + math.sqrt(
                4.0
                * t_k
                * t_k
                + 1.0
            )
        ) / 2.0

        history.append(
            {
                "iteration":
                    iteration + 1,

                "lambda_prior":
                    lambda_prior,

                "lambda_wavelet":
                    lambda_wavelet,

                "mu":
                    mu,

                "momentum":
                    momentum,

                "data_residual_before_dc":
                    float(
                        residual_before_dc.item()
                    ),

                "data_residual_after_dc":
                    float(
                        residual_after_dc.item()
                    ),

                "finite":
                    bool(
                        torch.isfinite(
                            image_dc
                        )
                        .all()
                        .item()
                    ),
            }
        )

        if verbose:

            print(
                f"iter "
                f"{iteration + 1:02d} | "
                f"lambda_prior="
                f"{lambda_prior:.6f} | "
                f"lambda_wavelet="
                f"{lambda_wavelet:.6f} | "
                f"mu="
                f"{mu:.6f} | "
                f"res_before="
                f"{residual_before_dc.item():.6e} | "
                f"res_after="
                f"{residual_after_dc.item():.6e}"
            )

        #
        # ------------------------------------------------
        # Original continuation schedule.
        # ------------------------------------------------
        #

        lambda_prior = max(
            beta_prior
            * lambda_prior,
            lambda_prior_floor,
        )

        mu = max(
            beta * mu,
            mu_floor,
        )

        lambda_wavelet = max(
            beta_wavelet
            * lambda_wavelet,
            lambda_wavelet_floor,
        )

    final_image = (
        wavelab_d4_inverse(
            X,
            wavelet_meta,
        )
        * scale
    )

    return (
        final_image,
        history,
    )
