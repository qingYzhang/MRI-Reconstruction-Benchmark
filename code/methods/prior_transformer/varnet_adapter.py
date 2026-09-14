import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Import the ORIGINAL Shamaei/DIRECT EndToEndVarNet implementation.
#
# We intentionally do not rewrite the VarNet architecture here.
# ---------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

SHAMAEI_ROOT = (
    PROJECT_ROOT
    / "external"
    / "shamaei_original"
)

if not SHAMAEI_ROOT.exists():
    raise RuntimeError(
        f"Shamaei source repo not found: "
        f"{SHAMAEI_ROOT}"
    )

if str(SHAMAEI_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SHAMAEI_ROOT),
    )


from src.utils.direct.nn.varnet.varnet import (
    EndToEndVarNet,
)


def _to_complex(
    x,
):
    """
    DIRECT format:
        [..., 2]

    -> torch complex tensor.
    """

    if x.shape[-1] != 2:
        raise ValueError(
            "Expected real/imag-last tensor "
            f"with final dimension 2, "
            f"got {tuple(x.shape)}"
        )

    return torch.view_as_complex(
        x.contiguous()
    )


def _to_ri(
    x,
):
    """
    torch complex -> DIRECT real/imag-last.
    """

    if not torch.is_complex(x):
        raise ValueError(
            "Expected complex tensor."
        )

    return torch.view_as_real(
        x
    )


def fastmri_fft2(
    x,
    dim=(2, 3),
):
    """
    Centered orthonormal FFT in DIRECT
    real/imag-last representation.

    This replaces the original Calgary-Campinas
    non-centered/non-normalized Fourier convention
    while leaving the VarNet architecture unchanged.
    """

    z = _to_complex(
        x
    )

    z = torch.fft.ifftshift(
        z,
        dim=dim,
    )

    z = torch.fft.fft2(
        z,
        dim=dim,
        norm="ortho",
    )

    z = torch.fft.fftshift(
        z,
        dim=dim,
    )

    return _to_ri(
        z
    )


def fastmri_ifft2(
    x,
    dim=(2, 3),
):
    """
    Centered orthonormal inverse FFT in DIRECT
    real/imag-last representation.
    """

    z = _to_complex(
        x
    )

    z = torch.fft.ifftshift(
        z,
        dim=dim,
    )

    z = torch.fft.ifft2(
        z,
        dim=dim,
        norm="ortho",
    )

    z = torch.fft.fftshift(
        z,
        dim=dim,
    )

    return _to_ri(
        z
    )


def complex_to_direct(
    x,
):
    """
    Native torch complex:
        [B, C, H, W]

    -> DIRECT:
        [B, C, H, W, 2]
    """

    if not torch.is_complex(
        x
    ):
        raise ValueError(
            "Expected native complex tensor."
        )

    return torch.view_as_real(
        x
    )


def direct_to_complex(
    x,
):
    """
    DIRECT:
        [B,C,H,W,2]

    -> native complex.
    """

    return _to_complex(
        x
    )


class ShamaeiE2EVarNetFastMRI(
    nn.Module
):
    """
    Source-faithful Shamaei E2E-VarNet core
    adapted to fastMRI Fourier convention.

    Source architecture:
        cascades            = 12
        regularizer filters = 18
        regularizer pools   = 4
        complex channels    = 2

    Sensitivity maps are supplied externally.

    For our fastMRI reproduction we use the
    ESPIRiT maps already validated in the common
    reconstruction pipeline instead of reproducing
    the Calgary-Campinas-specific sensitivity
    preprocessing engine.
    """

    def __init__(
        self,
        num_layers=12,
        regularizer_num_filters=18,
        regularizer_num_pull_layers=4,
        regularizer_dropout=0.0,
    ):
        super().__init__()

        self.num_layers = int(
            num_layers
        )

        self.model = EndToEndVarNet(
            forward_operator=
                fastmri_fft2,

            backward_operator=
                fastmri_ifft2,

            num_layers=
                num_layers,

            regularizer_num_filters=
                regularizer_num_filters,

            regularizer_num_pull_layers=
                regularizer_num_pull_layers,

            regularizer_dropout=
                regularizer_dropout,

            in_channels=2,
        )

    def forward(
        self,
        masked_kspace,
        mask,
        sensitivity_maps,
    ):
        """
        Parameters
        ----------
        masked_kspace:
            native complex
            [B,C,H,W]

        mask:
            bool or numeric mask, broadcastable
            from [B,1,H,W] or [B,1,H,W,1]

        sensitivity_maps:
            native complex
            [B,C,H,W]

        Returns
        -------
        output_kspace:
            native complex [B,C,H,W]
        """

        if not torch.is_complex(
            masked_kspace
        ):
            raise ValueError(
                "masked_kspace must be "
                "torch complex."
            )

        if not torch.is_complex(
            sensitivity_maps
        ):
            raise ValueError(
                "sensitivity_maps must be "
                "torch complex."
            )

        kspace_ri = (
            complex_to_direct(
                masked_kspace
            )
        )

        sens_ri = (
            complex_to_direct(
                sensitivity_maps
            )
        )

        if mask.ndim == 4:

            mask = mask.unsqueeze(
                -1
            )

        if mask.ndim != 5:
            raise ValueError(
                "mask must have shape "
                "[B,1,H,W] or "
                "[B,1,H,W,1], "
                f"got {tuple(mask.shape)}"
            )

        output_ri = self.model(
            masked_kspace=
                kspace_ri,

            sampling_mask=
                mask,

            sensitivity_map=
                sens_ri,
        )

        return direct_to_complex(
            output_ri
        )

    def reconstruct_rss(
        self,
        masked_kspace,
        mask,
        sensitivity_maps,
    ):
        """
        Return:
            image magnitude RSS [B,H,W]
            predicted k-space   [B,C,H,W]
        """

        output_kspace = self(
            masked_kspace=
                masked_kspace,

            mask=
                mask,

            sensitivity_maps=
                sensitivity_maps,
        )

        coil_images_ri = (
            fastmri_ifft2(
                complex_to_direct(
                    output_kspace
                )
            )
        )

        coil_images = (
            direct_to_complex(
                coil_images_ri
            )
        )

        rss = torch.sqrt(
            torch.sum(
                torch.abs(
                    coil_images
                ) ** 2,
                dim=1,
            )
        )

        return (
            rss,
            output_kspace,
        )
