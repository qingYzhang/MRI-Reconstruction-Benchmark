import sys
from pathlib import Path

import torch

CODE_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(CODE_ROOT)
)


from data.fastmri_brain import (
    FastMRIBrainDataset,
    apply_cartesian_mask,
)

from operators.sense import (
    ifft2c,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)


ROOT = Path(
    "extracted/multicoil_val"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def sensitivity_model_error(
    full_kspace,
    sens_maps,
):
    coil_images = ifft2c(
        full_kspace
    )

    x_fit = torch.sum(
        torch.conj(sens_maps)
        * coil_images,
        dim=0,
    )

    coil_fit = (
        sens_maps
        * x_fit.unsqueeze(0)
    )

    error = (
        torch.linalg.norm(
            coil_fit - coil_images
        )
        /
        torch.linalg.norm(
            coil_images
        )
    )

    return error


dataset = FastMRIBrainDataset(
    root=ROOT,
    acquisition="AXT2",
    cache_dir="code/.cache",
)


sample = dataset[0]

full_kspace = (
    sample["kspace"]
    .to(DEVICE)
)


print("=" * 80)
print("ESPIRiT TEST")
print("=" * 80)

print(
    "file:",
    sample["fname"]
)

print(
    "kspace:",
    full_kspace.shape
)


for acceleration in [
    4,
    8,
]:

    print(
        "\n" + "=" * 80
    )

    print(
        f"R = {acceleration}"
    )

    print(
        "=" * 80
    )

    _, _, num_low = (
        apply_cartesian_mask(
            sample,
            acceleration,
        )
    )

    print(
        "ACS lines:",
        num_low
    )

    sens_maps = (
        estimate_sens_espirit(
            full_kspace,
            num_low_frequencies=num_low,
        )
    )

    print(
        "sens maps:",
        sens_maps.shape,
        sens_maps.dtype,
    )

    power = torch.sum(
        torch.abs(sens_maps) ** 2,
        dim=0,
    )

    print(
        "sensitivity power mean:",
        power.mean().item()
    )

    print(
        "sensitivity power std:",
        power.std().item()
    )

    model_error = (
        sensitivity_model_error(
            full_kspace,
            sens_maps,
        )
    )

    print(
        "coil model relative error:",
        model_error.item()
    )
