import sys
from pathlib import Path

import torch


CODE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(CODE_ROOT)
)


from data.fastmri_brain import (
    FastMRIBrainDataset,
    zero_filled_rss,
)

from data.fastmri_256 import (
    FASTMRI_256_SIZE,
    build_fastmri_256_problem,
    build_fastmri_256_sample,
    validate_fastmri_256_sample,
)

from operators.sense import (
    prepare_mask,
    cg_sense,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def nmse(
    target,
    prediction,
):
    return (
        torch.sum(
            (
                target
                - prediction
            ) ** 2
        )
        /
        torch.sum(
            target ** 2
        )
    ).item()


dataset = FastMRIBrainDataset(
    root=
        "extracted/multicoil_train",

    acquisition=
        "AXT2",

    cache_dir=
        "code/.cache",
)

sample = dataset[532]

sample_256 = (
    build_fastmri_256_sample(
        sample
    )
)

validation = (
    validate_fastmri_256_sample(
        original_sample=
            sample,

        sample_256=
            sample_256,
    )
)

print("=" * 80)
print("FASTMRI 256 ADAPTER TEST")
print("=" * 80)

for key, value in (
    validation.items()
):
    print(
        f"{key}:",
        value,
    )

assert (
    validation[
        "adapted_kspace_shape"
    ][-2:]
    ==
    (
        FASTMRI_256_SIZE,
        FASTMRI_256_SIZE,
    )
)

assert (
    validation[
        "fft_roundtrip_rel_error"
    ]
    < 1e-5
)

assert (
    validation[
        "rss_target_rel_error"
    ]
    < 1e-5
)


for acceleration in [
    4,
    8,
]:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"R={acceleration}"
    )

    print(
        "=" * 80
    )

    (
        sample_256,
        masked_kspace,
        raw_mask,
        num_low,
    ) = build_fastmri_256_problem(
        sample,
        acceleration,
    )

    masked_kspace = (
        masked_kspace.to(
            DEVICE
        )
    )

    target = (
        sample_256[
            "target"
        ]
        .to(
            DEVICE
        )
    )

    mask = prepare_mask(
        raw_mask,
        width=
            FASTMRI_256_SIZE,
        device=
            DEVICE,
    )

    sens_maps = (
        estimate_sens_espirit(
            masked_kspace,
            num_low_frequencies=
                num_low,
            crop=
                0.0,
        )
    )

    zf = (
        zero_filled_rss(
            masked_kspace,
            (
                FASTMRI_256_SIZE,
                FASTMRI_256_SIZE,
            ),
        )
        .to(
            DEVICE
        )
    )

    cg = cg_sense(
        measured_kspace=
            masked_kspace,

        sens_maps=
            sens_maps,

        mask=
            mask,

        num_iters=
            50,

        lambda_reg=
            0.01,
    )

    cg = torch.abs(
        cg
    )

    print(
        "shape:",
        tuple(
            masked_kspace.shape
        ),
    )

    print(
        "ACS:",
        num_low,
    )

    print(
        "sampling:",
        mask.float()
        .mean()
        .item(),
    )

    print(
        "ZF NMSE:",
        nmse(
            target,
            zf,
        ),
    )

    print(
        "CG NMSE:",
        nmse(
            target,
            cg,
        ),
    )

    assert torch.isfinite(
        sens_maps
    ).all()

    assert torch.isfinite(
        cg
    ).all()


print(
    "\nFASTMRI 256 ADAPTER TEST PASSED"
)
