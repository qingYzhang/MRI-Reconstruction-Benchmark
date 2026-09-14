import torch


#
# Original Wavelab MakeONFilter(
#     'Daubechies',
#     4,
# )
#
# NOTE:
# This is a 4-tap Daubechies filter,
# equivalent to PyWavelets "db2",
# NOT PyWavelets "db4".
#
D4_QMF = (
    0.482962913145,
    0.836516303738,
    0.224143868042,
    -0.129409522551,
)


def _filter_tensor(
    x: torch.Tensor,
):
    dtype = (
        x.real.dtype
        if torch.is_complex(x)
        else x.dtype
    )

    return torch.tensor(
        D4_QMF,
        dtype=dtype,
        device=x.device,
    )


def _mirror_filter(
    f: torch.Tensor,
):
    """
    Wavelab MirrorFilt:

        h(t) = (-1)^(t-1) f(t)

    MATLAB indexing is 1-based, so in
    Python 0-based indexing the signs are:

        + - + -
    """

    signs = torch.ones_like(f)

    signs[1::2] = -1.0

    return f * signs


def _aconv(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Wavelab aconv.

    Periodic convolution with the
    time-reverse of f.

    Equivalent index form:

        y[t] =
            sum_r f[r] x[(t+r) mod n]
    """

    y = torch.zeros_like(x)

    for r in range(
        f.numel()
    ):

        y = (
            y
            + f[r]
            * torch.roll(
                x,
                shifts=-r,
                dims=dim,
            )
        )

    return y


def _iconv(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Wavelab iconv.

    Periodic convolution with f.

    Equivalent index form:

        y[t] =
            sum_r f[r] x[(t-r) mod n]
    """

    y = torch.zeros_like(x)

    for r in range(
        f.numel()
    ):

        y = (
            y
            + f[r]
            * torch.roll(
                x,
                shifts=r,
                dims=dim,
            )
        )

    return y


def _lshift(
    x: torch.Tensor,
    dim: int,
):
    """
    Circular left shift by one.
    """

    return torch.roll(
        x,
        shifts=-1,
        dims=dim,
    )


def _rshift(
    x: torch.Tensor,
    dim: int,
):
    """
    Circular right shift by one.
    """

    return torch.roll(
        x,
        shifts=1,
        dims=dim,
    )


def _take_even(
    x: torch.Tensor,
    dim: int,
):
    """
    MATLAB:

        x(1:2:n-1)

    corresponds to Python indices:

        0, 2, 4, ...
    """

    index = [
        slice(None)
    ] * x.ndim

    index[dim] = slice(
        0,
        None,
        2,
    )

    return x[
        tuple(index)
    ]


def _upsample_even(
    x: torch.Tensor,
    dim: int,
):
    """
    Insert zeros between samples:

        [a,b,c]
        ->
        [a,0,b,0,c,0]
    """

    shape = list(
        x.shape
    )

    shape[dim] *= 2

    y = torch.zeros(
        shape,
        dtype=x.dtype,
        device=x.device,
    )

    index = [
        slice(None)
    ] * x.ndim

    index[dim] = slice(
        0,
        None,
        2,
    )

    y[
        tuple(index)
    ] = x

    return y


def _down_dyad_lo(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Port of DownDyadLo.m.
    """

    filtered = _aconv(
        x,
        f,
        dim,
    )

    return _take_even(
        filtered,
        dim,
    )


def _down_dyad_hi(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Port of DownDyadHi.m.
    """

    high_filter = (
        _mirror_filter(f)
    )

    shifted = _lshift(
        x,
        dim,
    )

    filtered = _iconv(
        shifted,
        high_filter,
        dim,
    )

    return _take_even(
        filtered,
        dim,
    )


def _up_dyad_lo(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Port of UpDyadLo.m.
    """

    upsampled = (
        _upsample_even(
            x,
            dim,
        )
    )

    return _iconv(
        upsampled,
        f,
        dim,
    )


def _up_dyad_hi(
    x: torch.Tensor,
    f: torch.Tensor,
    dim: int,
):
    """
    Port of UpDyadHi.m.
    """

    high_filter = (
        _mirror_filter(f)
    )

    upsampled = (
        _upsample_even(
            x,
            dim,
        )
    )

    shifted = _rshift(
        upsampled,
        dim,
    )

    return _aconv(
        shifted,
        high_filter,
        dim,
    )


def max_common_levels(
    height: int,
    width: int,
):
    """
    Maximum number of dyadic decompositions
    possible without padding.

    Example:

        768 x 396

        768 -> 384 -> 192 -> ...
        396 -> 198 -> 99

    so only 2 common levels are possible.
    """

    levels = 0

    h = int(height)
    w = int(width)

    while (
        h % 2 == 0
        and w % 2 == 0
    ):

        h //= 2
        w //= 2

        levels += 1

    return levels


def wavelab_d4_forward(
    x: torch.Tensor,
    levels: int,
):
    """
    Rectangular extension of Wavelab
    FWT2_PO using the original:

        Daubechies-4-tap QMF
        periodic convolution
        orthogonal analysis filters
        Wavelab coefficient packing

    Original FWT2_PO assumes square dyadic
    arrays. For fastMRI we retain the same
    filtering convention but permit rectangular
    arrays as long as both dimensions are
    divisible by 2 at every requested level.

    Parameters
    ----------
    x:
        [H,W], real or complex.

    levels:
        Number of dyadic decomposition stages.

    Returns
    -------
    coeffs:
        Same shape as x.

    meta:
        Transform metadata.
    """

    if x.ndim != 2:
        raise ValueError(
            f"Expected [H,W], "
            f"got {x.shape}"
        )

    height, width = (
        x.shape
    )

    max_levels = (
        max_common_levels(
            height,
            width,
        )
    )

    if levels < 1:
        raise ValueError(
            "levels must be >= 1"
        )

    if levels > max_levels:
        raise ValueError(
            f"Requested {levels} levels, "
            f"but shape {x.shape} supports "
            f"at most {max_levels}."
        )

    coeffs = x.clone()

    f = _filter_tensor(
        x
    )

    current_h = height
    current_w = width

    for _ in range(
        levels
    ):

        region = coeffs[
            :current_h,
            :current_w,
        ].clone()

        #
        # Original FWT2_PO:
        # first transform rows.
        #
        row_low = (
            _down_dyad_lo(
                region,
                f,
                dim=-1,
            )
        )

        row_high = (
            _down_dyad_hi(
                region,
                f,
                dim=-1,
            )
        )

        row_packed = torch.cat(
            [
                row_low,
                row_high,
            ],
            dim=-1,
        )

        #
        # Then transform columns.
        #
        col_low = (
            _down_dyad_lo(
                row_packed,
                f,
                dim=-2,
            )
        )

        col_high = (
            _down_dyad_hi(
                row_packed,
                f,
                dim=-2,
            )
        )

        packed = torch.cat(
            [
                col_low,
                col_high,
            ],
            dim=-2,
        )

        coeffs[
            :current_h,
            :current_w,
        ] = packed

        current_h //= 2
        current_w //= 2

    meta = {
        "shape": (
            height,
            width,
        ),
        "levels":
            levels,
        "filter":
            "Wavelab Daubechies-4-tap",
        "boundary":
            "periodic",
    }

    return (
        coeffs,
        meta,
    )


def wavelab_d4_inverse(
    coeffs: torch.Tensor,
    meta,
):
    """
    Inverse / adjoint of
    wavelab_d4_forward().
    """

    if coeffs.ndim != 2:
        raise ValueError(
            f"Expected [H,W], "
            f"got {coeffs.shape}"
        )

    height, width = (
        meta["shape"]
    )

    levels = int(
        meta["levels"]
    )

    if tuple(
        coeffs.shape
    ) != (
        height,
        width,
    ):
        raise ValueError(
            f"Coefficient shape "
            f"{coeffs.shape} "
            f"does not match "
            f"{meta['shape']}."
        )

    x = coeffs.clone()

    f = _filter_tensor(
        coeffs
    )

    #
    # Deepest packed block reconstructed first.
    #
    current_h = (
        height
        // (
            2
            ** (
                levels - 1
            )
        )
    )

    current_w = (
        width
        // (
            2
            ** (
                levels - 1
            )
        )
    )

    for _ in range(
        levels
    ):

        region = x[
            :current_h,
            :current_w,
        ].clone()

        half_h = (
            current_h // 2
        )

        half_w = (
            current_w // 2
        )

        #
        # Original IWT2_PO:
        # reconstruct columns first.
        #
        col_low = region[
            :half_h,
            :
        ]

        col_high = region[
            half_h:current_h,
            :
        ]

        columns_reconstructed = (
            _up_dyad_lo(
                col_low,
                f,
                dim=-2,
            )
            +
            _up_dyad_hi(
                col_high,
                f,
                dim=-2,
            )
        )

        #
        # Then reconstruct rows.
        #
        row_low = (
            columns_reconstructed[
                :,
                :half_w,
            ]
        )

        row_high = (
            columns_reconstructed[
                :,
                half_w:current_w,
            ]
        )

        reconstructed = (
            _up_dyad_lo(
                row_low,
                f,
                dim=-1,
            )
            +
            _up_dyad_hi(
                row_high,
                f,
                dim=-1,
            )
        )

        x[
            :current_h,
            :current_w,
        ] = reconstructed

        current_h *= 2
        current_w *= 2

    return x
