import numpy as np

from fastmri.evaluate import (
    nmse,
    psnr,
)

from skimage.metrics import (
    structural_similarity,
)


def compute_ssim(
    target,
    prediction,
    max_value,
):
    """
    Compute SSIM slice-by-slice, then average over slices.

    Parameters
    ----------
    target:
        [S, H, W]

    prediction:
        [S, H, W]

    max_value:
        volume-level data range from fastMRI attrs["max"]
    """

    if target.ndim != 3:
        raise ValueError(
            f"Expected target [S,H,W], got {target.shape}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected prediction [S,H,W], got {prediction.shape}"
        )

    values = []

    for slice_idx in range(target.shape[0]):

        value = structural_similarity(
            target[slice_idx],
            prediction[slice_idx],
            data_range=max_value,
        )

        values.append(value)

    return float(
        np.mean(values)
    )


def compute_volume_metrics(
    target,
    prediction,
    max_value=None,
):
    """
    Compute reconstruction metrics for one complete volume.

    Parameters
    ----------
    target:
        numpy array [slices, H, W]

    prediction:
        numpy array [slices, H, W]

    max_value:
        fastMRI volume max value from HDF5 attrs["max"].

    Returns
    -------
    dict
        nmse, psnr, ssim
    """

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if target.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"target={target.shape}, "
            f"prediction={prediction.shape}"
        )

    if target.ndim != 3:
        raise ValueError(
            f"Expected volume [S,H,W], "
            f"got {target.shape}"
        )

    if max_value is None or max_value <= 0:
        max_value = float(
            target.max()
        )

    return {
        "nmse": float(
            nmse(
                target,
                prediction,
            )
        ),

        "psnr": float(
            psnr(
                target,
                prediction,
                maxval=max_value,
            )
        ),

        "ssim": compute_ssim(
            target,
            prediction,
            max_value=max_value,
        ),
    }