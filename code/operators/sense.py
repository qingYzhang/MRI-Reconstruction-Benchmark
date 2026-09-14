import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered orthonormal 2D FFT.

    x:
        complex tensor [..., H, W]
    """

    x = torch.fft.ifftshift(
        x,
        dim=(-2, -1),
    )

    x = torch.fft.fft2(
        x,
        dim=(-2, -1),
        norm="ortho",
    )

    x = torch.fft.fftshift(
        x,
        dim=(-2, -1),
    )

    return x


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered orthonormal 2D inverse FFT.

    x:
        complex tensor [..., H, W]
    """

    x = torch.fft.ifftshift(
        x,
        dim=(-2, -1),
    )

    x = torch.fft.ifft2(
        x,
        dim=(-2, -1),
        norm="ortho",
    )

    x = torch.fft.fftshift(
        x,
        dim=(-2, -1),
    )

    return x


def prepare_mask(
    mask: torch.Tensor,
    width: int,
    device=None,
):
    """
    Convert fastMRI mask

        [1, 1, W, 1]

    into

        [1, 1, W]

    suitable for broadcasting over
        [coils, H, W].
    """

    mask = mask.squeeze()

    if mask.ndim != 1:
        raise ValueError(
            f"Expected 1D mask after squeeze, "
            f"got {mask.shape}"
        )

    if mask.shape[0] != width:
        raise ValueError(
            f"Mask width {mask.shape[0]} "
            f"!= k-space width {width}"
        )

    mask = mask.to(
        device=device,
        dtype=torch.bool,
    )

    return mask.view(
        1,
        1,
        width,
    )


def estimate_sens_acs(
    full_kspace: torch.Tensor,
    num_low_frequencies: int,
    eps: float = 1e-8,
):
    """
    Estimate coil sensitivity maps from the
    fully sampled ACS region.

    Parameters
    ----------
    full_kspace:
        complex [C, H, W]

    num_low_frequencies:
        number of central fully-sampled
        phase-encoding lines.

    Returns
    -------
    sens_maps:
        complex [C, H, W]

    Notes
    -----
    This is NOT full ESPIRiT.

    It is a simple ACS-based estimate:
        ACS -> IFFT -> coil images -> normalize by RSS
    """

    if not torch.is_complex(full_kspace):
        raise TypeError(
            "full_kspace must be complex"
        )

    if full_kspace.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], "
            f"got {full_kspace.shape}"
        )

    _, _, width = full_kspace.shape

    nlow = int(num_low_frequencies)

    if nlow <= 0 or nlow > width:
        raise ValueError(
            f"Invalid num_low_frequencies={nlow}"
        )

    left = (width - nlow + 1) // 2
    right = left + nlow

    acs = torch.zeros_like(
        full_kspace
    )

    acs[..., left:right] = (
        full_kspace[..., left:right]
    )

    coil_images = ifft2c(
        acs
    )

    rss = torch.sqrt(
        torch.sum(
            torch.abs(coil_images) ** 2,
            dim=0,
            keepdim=True,
        )
    )

    sens_maps = (
        coil_images
        / torch.clamp(
            rss,
            min=eps,
        )
    )

    return sens_maps


def sense_forward(
    image: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
):
    """
    A(x) = M F S x

    image:
        complex [H,W]

    sens_maps:
        complex [C,H,W]

    mask:
        bool/float, broadcastable to [C,H,W]

    returns:
        complex [C,H,W]
    """

    if image.ndim != 2:
        raise ValueError(
            f"image must be [H,W], "
            f"got {image.shape}"
        )

    coil_images = (
        sens_maps
        * image.unsqueeze(0)
    )

    kspace = fft2c(
        coil_images
    )

    return kspace * mask


def sense_adjoint(
    kspace: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
):
    """
    A^H(y) = S^H F^-1 M y

    kspace:
        complex [C,H,W]

    sens_maps:
        complex [C,H,W]

    returns:
        complex [H,W]
    """

    kspace = (
        kspace
        * mask
    )

    coil_images = ifft2c(
        kspace
    )

    image = torch.sum(
        torch.conj(sens_maps)
        * coil_images,
        dim=0,
    )

    return image


def sense_normal(
    image: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
    lambda_reg: float = 0.0,
):
    """
    (A^H A + lambda I)x
    """

    result = sense_adjoint(
        sense_forward(
            image,
            sens_maps,
            mask,
        ),
        sens_maps,
        mask,
    )

    if lambda_reg != 0:
        result = (
            result
            + lambda_reg * image
        )

    return result


def complex_inner(
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    Real inner product for complex CG.
    """

    return torch.real(
        torch.sum(
            torch.conj(x) * y
        )
    )


@torch.no_grad()
def cg_sense(
    measured_kspace: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
    num_iters: int = 30,
    lambda_reg: float = 1e-4,
    tol: float = 1e-8,
):
    """
    Solve:

        (A^H A + lambda I)x = A^H y

    using conjugate gradient.

    Returns
    -------
    image:
        complex [H,W]
    """

    b = sense_adjoint(
        measured_kspace,
        sens_maps,
        mask,
    )

    x = torch.zeros_like(
        b
    )

    r = b.clone()
    p = r.clone()

    rsold = complex_inner(
        r,
        r,
    )

    initial_residual = torch.sqrt(
        rsold
    )

    for iteration in range(num_iters):

        Ap = sense_normal(
            p,
            sens_maps,
            mask,
            lambda_reg=lambda_reg,
        )

        denom = complex_inner(
            p,
            Ap,
        )

        if torch.abs(denom) < 1e-20:
            break

        alpha = (
            rsold / denom
        )

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = complex_inner(
            r,
            r,
        )

        residual = torch.sqrt(
            rsnew
        )

        relative_residual = (
            residual
            / (initial_residual + 1e-20)
        )

        if relative_residual < tol:
            break

        beta = (
            rsnew / rsold
        )

        p = (
            r + beta * p
        )

        rsold = rsnew

    return x

@torch.no_grad()
def cg_normal_solve(
    rhs: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
    diag_reg: float,
    num_iters: int = 30,
    tol: float = 1e-8,
    x0=None,
):
    """
    Solve

        (A^H A + diag_reg I) x = rhs

    using complex conjugate gradient.

    This is used by LACS/ADMM and can
    later be reused by other methods.
    """

    if x0 is None:
        x = torch.zeros_like(
            rhs
        )
    else:
        x = x0.clone()

    r = (
        rhs
        - sense_normal(
            x,
            sens_maps,
            mask,
            lambda_reg=diag_reg,
        )
    )

    p = r.clone()

    rsold = complex_inner(
        r,
        r,
    )

    initial_residual = torch.sqrt(
        torch.clamp(
            rsold,
            min=0.0,
        )
    )

    for _ in range(num_iters):

        Ap = sense_normal(
            p,
            sens_maps,
            mask,
            lambda_reg=diag_reg,
        )

        denom = complex_inner(
            p,
            Ap,
        )

        if (
            torch.abs(denom)
            < 1e-20
        ):
            break

        alpha = (
            rsold
            / denom
        )

        x = (
            x
            + alpha * p
        )

        r = (
            r
            - alpha * Ap
        )

        rsnew = complex_inner(
            r,
            r,
        )

        residual = torch.sqrt(
            torch.clamp(
                rsnew,
                min=0.0,
            )
        )

        relative_residual = (
            residual
            /
            (
                initial_residual
                + 1e-20
            )
        )

        if (
            relative_residual
            < tol
        ):
            break

        beta = (
            rsnew
            / rsold
        )

        p = (
            r
            + beta * p
        )

        rsold = rsnew

    return x