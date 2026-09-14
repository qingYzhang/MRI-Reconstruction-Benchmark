import sys
from pathlib import Path
from fastmri.data.transforms import center_crop

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
    fft2c,
    ifft2c,
    prepare_mask,
    estimate_sens_acs,
    sense_forward,
    sense_adjoint,
    complex_inner,
)


ROOT = Path(
    "extracted/multicoil_val"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
print("\n" + "=" * 80)
print("FULLY SAMPLED RSS CONSISTENCY")
print("=" * 80)

coil_images = ifft2c(
    full_kspace
)

full_rss = torch.sqrt(
    torch.sum(
        torch.abs(coil_images) ** 2,
        dim=0,
    )
)

full_rss = center_crop(
    full_rss,
    sample["target"].shape[-2:],
)

target = sample["target"].to(DEVICE)

nmse_full = (
    torch.sum(
        (full_rss - target) ** 2
    )
    /
    torch.sum(
        target ** 2
    )
)

max_error = torch.max(
    torch.abs(full_rss - target)
)

print(
    "full RSS NMSE vs target:",
    nmse_full.item()
)

print(
    "max abs error:",
    max_error.item()
)

print("=" * 80)
print("FFT INVERSE TEST")
print("=" * 80)

test = torch.randn(
    384,
    384,
    device=DEVICE,
    dtype=torch.complex64,
)

recovered = ifft2c(
    fft2c(test)
)

fft_error = (
    torch.linalg.norm(
        recovered - test
    )
    / torch.linalg.norm(test)
)

print(
    "relative FFT inverse error:",
    fft_error.item(),
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

for acceleration in [4, 8]:

    print(
        "\n" + "=" * 80
    )
    print(
        f"R = {acceleration}"
    )
    print(
        "=" * 80
    )

    masked_kspace, raw_mask, num_low = (
        apply_cartesian_mask(
            sample,
            acceleration,
        )
    )

    masked_kspace = (
        masked_kspace.to(DEVICE)
    )

    mask = prepare_mask(
        raw_mask,
        width=full_kspace.shape[-1],
        device=DEVICE,
    )

    sens_maps = estimate_sens_acs(
        full_kspace,
        num_low,
    )

    print(
        "full kspace:",
        full_kspace.shape
    )

    print(
        "mask:",
        mask.shape
    )

    print(
        "sens maps:",
        sens_maps.shape
    )

    print(
        "num low:",
        num_low
    )

    sens_power = torch.sum(
        torch.abs(sens_maps) ** 2,
        dim=0,
    )

    print(
        "sensitivity power:",
        float(sens_power.mean()),
        "+/-",
        float(sens_power.std()),
    )

    #
    # Adjoint test:
    #
    # <Ax,y> = <x,A^Hy>
    #
    H = full_kspace.shape[-2]
    W = full_kspace.shape[-1]
    C = full_kspace.shape[0]

    x = torch.randn(
        H,
        W,
        dtype=torch.complex64,
        device=DEVICE,
    )

    y = torch.randn(
        C,
        H,
        W,
        dtype=torch.complex64,
        device=DEVICE,
    )

    Ax = sense_forward(
        x,
        sens_maps,
        mask,
    )

    AHy = sense_adjoint(
        y,
        sens_maps,
        mask,
    )

    lhs = torch.sum(
        torch.conj(Ax) * y
    )

    rhs = torch.sum(
        torch.conj(x) * AHy
    )

    adjoint_error = (
        torch.abs(lhs - rhs)
        /
        (
            torch.abs(lhs)
            + torch.abs(rhs)
            + 1e-12
        )
    )

    print("lhs:", lhs.item())
    print("rhs:", rhs.item())

    print(
        "complex adjoint relative error:",
        adjoint_error.item()
    )
    model_error = sensitivity_model_error(
        full_kspace,
        sens_maps,
    )

    print(
        "coil model relative error:",
        model_error.item()
    )
