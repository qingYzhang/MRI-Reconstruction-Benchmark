import numpy as np
import torch

import sigpy as sp
from sigpy.mri.app import EspiritCalib


def extract_acs(
    kspace: torch.Tensor,
    num_low_frequencies: int,
):
    """
    Extract central ACS region in phase-encoding direction.

    Parameters
    ----------
    kspace:
        complex tensor [C, H, W]

    num_low_frequencies:
        number of fully sampled central PE lines

    Returns
    -------
    acs:
        complex tensor [C, H, W]
        zero outside ACS region
    """

    if kspace.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {kspace.shape}"
        )

    _, _, width = kspace.shape

    nlow = int(num_low_frequencies)

    if nlow <= 0 or nlow > width:
        raise ValueError(
            f"Invalid ACS width: {nlow}"
        )

    left = (width - nlow + 1) // 2
    right = left + nlow

    acs = torch.zeros_like(
        kspace
    )

    acs[..., left:right] = (
        kspace[..., left:right]
    )

    return acs


def estimate_sens_espirit(
    kspace: torch.Tensor,
    num_low_frequencies: int,
    kernel_width: int = 6,
    crop: float = 0.95,
    thresh: float = 0.02,
    eps: float = 1e-8,
):
    """
    ESPIRiT coil sensitivity estimation.

    Only the fully sampled ACS region is used.

    Parameters
    ----------
    kspace:
        complex torch tensor [C,H,W]

    num_low_frequencies:
        number of central ACS lines

    Returns
    -------
    sens_maps:
        complex torch tensor [C,H,W]
        on the same device as input
    """

    if not torch.is_complex(kspace):
        raise TypeError(
            "kspace must be complex"
        )

    original_device = (
        kspace.device
    )

    #
    # Explicitly zero all non-ACS measurements.
    #
    acs = extract_acs(
        kspace,
        num_low_frequencies,
    )

    acs_np = (
        acs
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    #
    # SigPy performs ESPIRiT calibration
    # from central calibration k-space.
    #
    maps_np = EspiritCalib(
        acs_np,
        calib_width=int(
            num_low_frequencies
        ),
        kernel_width=kernel_width,
        crop=crop,
        thresh=thresh,
        device=sp.Device(-1),
        show_pbar=False,
    ).run()

    sens_maps = torch.from_numpy(
        np.asarray(
            maps_np,
            dtype=np.complex64,
        )
    ).to(
        original_device
    )

    #
    # Ensure S^H S ~= 1.
    #
    power = torch.sqrt(
        torch.sum(
            torch.abs(sens_maps) ** 2,
            dim=0,
            keepdim=True,
        )
    )

    sens_maps = (
        sens_maps
        / torch.clamp(
            power,
            min=eps,
        )
    )

    return sens_maps
