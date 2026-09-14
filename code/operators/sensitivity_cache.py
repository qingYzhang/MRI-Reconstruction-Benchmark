import hashlib
import os
from pathlib import Path

import torch

from operators.sensitivity import (
    estimate_sens_espirit,
)


CACHE_VERSION = 1


def tensor_sha256(x):
    x = (
        x.detach()
        .cpu()
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )

    return hashlib.sha256(
        x.tobytes()
    ).hexdigest()


def get_sens_cache_path(
    cache_root,
    dataset_tag,
    acquisition,
    acceleration,
    fname,
    slice_num,
):
    return (
        Path(cache_root)
        / dataset_tag
        / acquisition
        / f"R{int(acceleration)}"
        / Path(fname).stem
        / f"slice_{int(slice_num):04d}.pt"
    )


def build_metadata(
    dataset_tag,
    acquisition,
    acceleration,
    fname,
    slice_num,
    masked_kspace,
    mask,
    num_low_frequencies,
    crop,
):
    return {
        "cache_version":
            CACHE_VERSION,

        "dataset_tag":
            str(dataset_tag),

        "acquisition":
            str(acquisition),

        "acceleration":
            int(acceleration),

        "fname":
            str(fname),

        "slice_num":
            int(slice_num),

        "kspace_shape":
            list(
                masked_kspace.shape
            ),

        "num_low_frequencies":
            int(num_low_frequencies),

        "crop":
            float(crop),

        "mask_sha256":
            tensor_sha256(mask),
    }


def validate_metadata(
    cached,
    expected,
):
    for key, value in expected.items():

        if cached.get(key) != value:

            raise RuntimeError(
                "ESPIRiT cache metadata mismatch:\n"
                f"  key:      {key}\n"
                f"  cached:   {cached.get(key)}\n"
                f"  expected: {value}\n"
                "Delete the stale cache entry or "
                "rerun with recompute=True."
            )


def load_torch(path):

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def get_sensitivity_maps_cached(
    *,
    cache_root,
    dataset_tag,
    acquisition,
    acceleration,
    fname,
    slice_num,
    masked_kspace,
    mask,
    num_low_frequencies,
    crop,
    device,
    recompute=False,
):
    """
    Load ESPIRiT maps from cache when available.

    On cache miss:
      estimate from the SAME masked k-space,
      save CPU complex64 maps atomically,
      and return them on `device`.

    Returns
    -------
    sens_maps
    cache_hit
    cache_path
    """

    cache_path = get_sens_cache_path(
        cache_root=
            cache_root,

        dataset_tag=
            dataset_tag,

        acquisition=
            acquisition,

        acceleration=
            acceleration,

        fname=
            fname,

        slice_num=
            slice_num,
    )

    metadata = build_metadata(
        dataset_tag=
            dataset_tag,

        acquisition=
            acquisition,

        acceleration=
            acceleration,

        fname=
            fname,

        slice_num=
            slice_num,

        masked_kspace=
            masked_kspace,

        mask=
            mask,

        num_low_frequencies=
            num_low_frequencies,

        crop=
            crop,
    )

    if (
        cache_path.exists()
        and not recompute
    ):

        payload = load_torch(
            cache_path
        )

        validate_metadata(
            cached=
                payload["metadata"],

            expected=
                metadata,
        )

        sens_maps = (
            payload["sens_maps"]
            .to(
                device=device,
                dtype=torch.complex64,
            )
        )

        if not torch.isfinite(
            sens_maps
        ).all():

            raise RuntimeError(
                "Cached ESPIRiT maps "
                f"are non-finite: {cache_path}"
            )

        return (
            sens_maps,
            True,
            cache_path,
        )

    sens_maps = (
        estimate_sens_espirit(
            masked_kspace,
            num_low_frequencies=
                num_low_frequencies,
            crop=
                crop,
        )
    )

    if not torch.isfinite(
        sens_maps
    ).all():

        raise RuntimeError(
            "New ESPIRiT maps are non-finite."
        )

    payload = {
        "metadata":
            metadata,

        "sens_maps":
            (
                sens_maps
                .detach()
                .cpu()
                .to(torch.complex64)
            ),
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = cache_path.with_name(
        cache_path.name
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        payload,
        tmp_path,
    )

    os.replace(
        tmp_path,
        cache_path,
    )

    return (
        sens_maps,
        False,
        cache_path,
    )
