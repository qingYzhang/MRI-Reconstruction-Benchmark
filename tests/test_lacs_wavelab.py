import sys
from pathlib import Path

import torch


CODE_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(CODE_ROOT)
)


from methods.lacs.wavelab import (
    D4_QMF,
    max_common_levels,
    wavelab_d4_forward,
    wavelab_d4_inverse,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def complex_inner(
    x,
    y,
):
    return torch.sum(
        torch.conj(x)
        * y
    )


def run_test(
    shape,
    levels,
):

    torch.manual_seed(0)

    x = (
        torch.randn(
            shape,
            device=DEVICE,
        )
        +
        1j
        * torch.randn(
            shape,
            device=DEVICE,
        )
    ).to(
        torch.complex64
    )

    coeffs, meta = (
        wavelab_d4_forward(
            x,
            levels=levels,
        )
    )

    reconstructed = (
        wavelab_d4_inverse(
            coeffs,
            meta,
        )
    )

    inverse_error = (
        torch.linalg.norm(
            reconstructed
            - x
        )
        /
        torch.linalg.norm(
            x
        )
    )

    input_energy = torch.sum(
        torch.abs(x) ** 2
    )

    coefficient_energy = (
        torch.sum(
            torch.abs(
                coeffs
            ) ** 2
        )
    )

    energy_error = (
        torch.abs(
            input_energy
            - coefficient_energy
        )
        /
        input_energy
    )

    #
    # Adjoint test.
    #
    y = (
        torch.randn(
            coeffs.shape,
            device=DEVICE,
        )
        +
        1j
        * torch.randn(
            coeffs.shape,
            device=DEVICE,
        )
    ).to(
        torch.complex64
    )

    psi_x, meta2 = (
        wavelab_d4_forward(
            x,
            levels=levels,
        )
    )

    psi_h_y = (
        wavelab_d4_inverse(
            y,
            meta2,
        )
    )

    lhs = complex_inner(
        psi_x,
        y,
    )

    rhs = complex_inner(
        x,
        psi_h_y,
    )

    adjoint_error = (
        torch.abs(
            lhs - rhs
        )
        /
        (
            torch.abs(lhs)
            + torch.abs(rhs)
            + 1e-12
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"shape={shape}, "
        f"levels={levels}"
    )

    print(
        "=" * 80
    )

    print(
        "max supported levels:",
        max_common_levels(
            *shape
        )
    )

    print(
        "inverse relative error:",
        inverse_error.item()
    )

    print(
        "energy relative error:",
        energy_error.item()
    )

    print(
        "adjoint relative error:",
        adjoint_error.item()
    )

    print(
        "finite coefficients:",
        torch.isfinite(
            coeffs
        ).all().item()
    )


def main():

    print("=" * 80)
    print("WAVELAB D4 PORT TEST")
    print("=" * 80)

    print(
        "device:",
        DEVICE
    )

    print(
        "D4 QMF:",
        D4_QMF
    )

    #
    # Original-style dyadic test.
    #
    # Wavelet default wavScale=1 on
    # 256x256 means:
    #
    # J=8
    # jscal=7,...,1
    #
    # => 7 decomposition stages.
    #
    run_test(
        shape=(
            256,
            256,
        ),
        levels=7,
    )

    #
    # Our actual fastMRI native
    # encoded image domain.
    #
    run_test(
        shape=(
            768,
            396,
        ),
        levels=2,
    )


if __name__ == "__main__":
    main()
