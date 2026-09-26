#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CODE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROJECT_ROOT = (
    CODE_ROOT.parent
)

sys.path.insert(
    0,
    str(CODE_ROOT),
)

SHAMAEI_ROOT = (
    PROJECT_ROOT
    / "external"
    / "shamaei_original"
)

sys.path.insert(
    0,
    str(SHAMAEI_ROOT),
)


from data.fastmri_brain import (
    FastMRIBrainDataset,
    zero_filled_rss,
)

from data.fastmri_256 import (
    build_fastmri_256_problem,
)

from evaluation.metrics import (
    compute_volume_metrics,
)

from operators.sense import (
    prepare_mask,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)

from methods.prior_transformer.varnet_adapter import (
    ShamaeiE2EVarNetFastMRI,
)


CALIBRATIONS = [
    "complex_alpha",
    "kspace_norm",
    "zf_l2",
    "zf_p995",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help=(
            "Use label=path, e.g. "
            "best=outputs/.../best.pt "
            "last=outputs/.../last.pt"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=(
            "outputs/"
            "shamaei_fastmri256_training_manifest.json"
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=(
            "extracted/"
            "multicoil_val"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help=(
            "Uniformly sample this many entries "
            "across the full validation list. "
            "Prefer this over --limit_samples "
            "for representative diagnostics."
        ),
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
    )

    return parser.parse_args()


def parse_checkpoints(items):

    result = {}

    for item in items:

        if "=" not in item:
            raise ValueError(
                "--checkpoints entries must "
                "use label=path"
            )

        label, path = (
            item.split(
                "=",
                1,
            )
        )

        result[
            label
        ] = path

    return result


def load_model(
    checkpoint_path,
    acceleration,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = (
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        len(state)
        and all(
            key.startswith(
                "module."
            )
            for key
            in state
        )
    ):

        state = {
            key[
                len("module.") :
            ]:
                value
            for key, value
            in state.items()
        }

    model = (
        ShamaeiE2EVarNetFastMRI(
            num_layers=12,
            regularizer_num_filters=18,
            regularizer_num_pull_layers=4,
            regularizer_dropout=0.0,
        )
        .to(device)
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    count = sum(
        p.numel()
        for p in model.parameters()
    )

    assert (
        count
        == 29452068
    ), count

    ckpt_acceleration = (
        checkpoint
        .get(
            "adaptation",
            {},
        )
        .get(
            "acceleration",
            None,
        )
    )

    if (
        ckpt_acceleration
        is not None
        and int(
            ckpt_acceleration
        )
        != acceleration
    ):
        raise RuntimeError(
            f"{checkpoint_path}: "
            f"checkpoint R{ckpt_acceleration}, "
            f"requested R{acceleration}"
        )

    return model


def raw_fname(
    raw,
):

    if hasattr(
        raw,
        "fname",
    ):
        return Path(
            raw.fname
        ).name

    if isinstance(
        raw,
        tuple,
    ):
        return Path(
            raw[0]
        ).name

    raise RuntimeError(
        "Cannot extract fname "
        "from raw sample."
    )


def raw_slice(
    raw,
):

    for attr in [
        "slice_ind",
        "slice_num",
        "slice_index",
    ]:

        if hasattr(
            raw,
            attr,
        ):
            return int(
                getattr(
                    raw,
                    attr,
                )
            )

    if (
        isinstance(
            raw,
            tuple,
        )
        and len(raw) > 1
    ):
        return int(
            raw[1]
        )

    raise RuntimeError(
        "Cannot extract slice index "
        "from raw sample."
    )


def build_dataset_lookup(
    dataset,
):

    if hasattr(
        dataset,
        "raw_samples",
    ):
        raw_samples = (
            dataset.raw_samples
        )

    elif hasattr(
        dataset,
        "examples",
    ):
        raw_samples = (
            dataset.examples
        )

    else:
        raise RuntimeError(
            "Dataset has no raw_samples/examples."
        )

    lookup = {}

    for idx, raw in enumerate(
        raw_samples
    ):

        key = (
            raw_fname(
                raw
            ),
            raw_slice(
                raw
            ),
        )

        lookup[
            key
        ] = idx

    return lookup


def _looks_like_sample_entry(
    value,
):
    if not isinstance(
        value,
        dict,
    ):
        return False

    fname_keys = {
        "fname",
        "filename",
        "file",
    }

    slice_keys = {
        "slice_num",
        "slice_idx",
        "slice_index",
        "slice",
    }

    return (
        any(
            key in value
            for key in fname_keys
        )
        and
        any(
            key in value
            for key in slice_keys
        )
    )


def _collect_sample_entries(
    value,
):
    """
    Recursively find manifest entries that contain
    both a filename field and a slice-index field.
    """

    result = []

    if _looks_like_sample_entry(
        value
    ):
        result.append(
            value
        )
        return result

    if isinstance(
        value,
        list,
    ):
        for item in value:
            result.extend(
                _collect_sample_entries(
                    item
                )
            )

    elif isinstance(
        value,
        dict,
    ):
        for item in value.values():
            result.extend(
                _collect_sample_entries(
                    item
                )
            )

    return result


def get_manifest_entries(
    path,
):

    with open(
        path
    ) as f:
        manifest = json.load(
            f
        )

    if (
        "validation"
        not in manifest
    ):
        raise RuntimeError(
            "Manifest has no "
            "'validation' section."
        )

    validation = (
        manifest[
            "validation"
        ]
    )

    print(
        "validation section type:",
        type(
            validation
        ).__name__,
    )

    if isinstance(
        validation,
        dict,
    ):
        print(
            "validation section keys:",
            list(
                validation.keys()
            ),
        )

    entries = (
        _collect_sample_entries(
            validation
        )
    )

    if not entries:
        raise RuntimeError(
            "Could not find any validation "
            "sample entries containing both "
            "filename and slice-index fields."
        )

    #
    # Deduplicate defensively in case manifest metadata
    # references the same sample in more than one place.
    #
    unique = {}

    for entry in entries:

        fname = (
            entry_fname(
                entry
            )
        )

        slice_num = (
            entry_slice(
                entry
            )
        )

        key = (
            fname,
            slice_num,
        )

        unique[
            key
        ] = entry

    entries = list(
        unique.values()
    )

    entries.sort(
        key=lambda entry: (
            entry_fname(
                entry
            ),
            entry_slice(
                entry
            ),
        )
    )

    print(
        "parsed validation samples:",
        len(
            entries
        ),
    )

    print(
        "first parsed sample:",
        entry_fname(
            entries[0]
        ),
        entry_slice(
            entries[0]
        ),
    )

    return entries


def entry_fname(
    entry,
):

    for key in [
        "fname",
        "filename",
        "file",
    ]:

        if key in entry:
            return Path(
                entry[
                    key
                ]
            ).name

    raise RuntimeError(
        "Cannot find fname in manifest "
        f"entry keys: {list(entry.keys())}"
    )


def entry_slice(
    entry,
):

    for key in [
        "slice_num",
        "slice_idx",
        "slice_index",
        "slice",
    ]:

        if key in entry:
            return int(
                entry[
                    key
                ]
            )

    raise RuntimeError(
        "Cannot find slice number in "
        "manifest entry keys: "
        f"{list(entry.keys())}"
    )


def make_model_mask(
    common_mask,
    height,
    width,
):

    mask_2d = (
        common_mask
        .squeeze()
        .bool()
    )

    if (
        mask_2d.ndim
        == 1
    ):

        if (
            mask_2d.numel()
            != width
        ):
            raise RuntimeError(
                "Unexpected 1-D mask "
                f"{tuple(mask_2d.shape)}"
            )

        mask_2d = (
            mask_2d[
                None,
                :
            ]
            .expand(
                height,
                width,
            )
        )

    if (
        tuple(
            mask_2d.shape
        )
        != (
            height,
            width,
        )
    ):
        raise RuntimeError(
            "Unexpected 2-D mask "
            f"{tuple(mask_2d.shape)}"
        )

    return (
        mask_2d,
        mask_2d[
            None,
            None,
            ...
        ],
    )


@torch.no_grad()
def reconstruct_all_calibrations(
    model,
    masked_kspace,
    common_mask,
    sens_maps,
):

    height = int(
        masked_kspace.shape[-2]
    )

    width = int(
        masked_kspace.shape[-1]
    )

    zf = (
        zero_filled_rss(
            masked_kspace,
            (
                height,
                width,
            ),
        )
        .to(
            masked_kspace.device
        )
    )

    measurement_scale = (
        torch.quantile(
            zf,
            0.999,
        )
    )

    if (
        measurement_scale.item()
        <= 0
    ):
        raise RuntimeError(
            "Invalid measurement scale."
        )

    measured_scaled = (
        masked_kspace
        / measurement_scale
    )

    (
        mask_2d,
        model_mask,
    ) = make_model_mask(
        common_mask,
        height,
        width,
    )

    (
        rss_scaled,
        output_kspace,
    ) = (
        model.reconstruct_rss(
            measured_scaled[
                None,
                ...
            ],
            model_mask,
            sens_maps[
                None,
                ...
            ],
        )
    )

    rss_scaled = (
        rss_scaled[
            0
        ]
    )

    pred_kspace = (
        output_kspace[
            0
        ]
    )

    raw_image = (
        rss_scaled
        * measurement_scale
    )

    acquired = (
        mask_2d[
            None,
            ...
        ]
        .expand_as(
            pred_kspace
        )
    )

    kp = (
        pred_kspace[
            acquired
        ]
    )

    yy = (
        measured_scaled[
            acquired
        ]
    )

    kp_power = (
        torch.sum(
            torch.abs(
                kp
            ) ** 2
        )
    )

    y_power = (
        torch.sum(
            torch.abs(
                yy
            ) ** 2
        )
    )

    cross = (
        torch.sum(
            torch.conj(
                kp
            )
            * yy
        )
    )

    alpha_complex = (
        cross
        / kp_power
    )

    scale_complex = (
        torch.abs(
            alpha_complex
        )
    )

    scale_kspace_norm = (
        torch.sqrt(
            y_power
            / kp_power
        )
    )

    scale_zf_l2 = (
        torch.linalg.vector_norm(
            zf
        )
        / torch.linalg.vector_norm(
            raw_image
        )
    )

    scale_zf_p995 = (
        torch.quantile(
            zf,
            0.995,
        )
        / torch.quantile(
            raw_image,
            0.995,
        )
    )

    scales = {
        "complex_alpha":
            scale_complex,

        "kspace_norm":
            scale_kspace_norm,

        "zf_l2":
            scale_zf_l2,

        "zf_p995":
            scale_zf_p995,
    }

    predictions = {
        name:
            (
                raw_image
                * scale
            )
        for name, scale
        in scales.items()
    }

    coherence = (
        torch.abs(
            cross
        )
        / torch.sqrt(
            kp_power
            * y_power
        )
    )

    diagnostics = {
        "measurement_scale":
            float(
                measurement_scale
                .item()
            ),

        "complex_coherence":
            float(
                coherence.item()
            ),

        **{
            f"scale_{name}":
                float(
                    value.item()
                )
            for name, value
            in scales.items()
        },
    }

    return (
        predictions,
        diagnostics,
    )


def metric_one_slice(
    target,
    prediction,
):

    target_np = (
        target
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    pred_np = (
        prediction
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    values = (
        compute_volume_metrics(
            target=
                target_np[
                    None,
                    ...
                ],

            prediction=
                pred_np[
                    None,
                    ...
                ],

            max_value=
                float(
                    target_np.max()
                ),
        )
    )

    norm_ratio = (
        np.linalg.norm(
            pred_np.ravel()
        )
        / max(
            np.linalg.norm(
                target_np.ravel()
            ),
            1e-12,
        )
    )

    return {
        "nmse":
            float(
                values[
                    "nmse"
                ]
            ),

        "psnr":
            float(
                values[
                    "psnr"
                ]
            ),

        "ssim":
            float(
                values[
                    "ssim"
                ]
            ),

        "norm_ratio":
            float(
                norm_ratio
            ),
    }


def write_csv(
    path,
    rows,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=
                    list(
                        rows[
                            0
                        ].keys()
                    ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def print_summary(
    rows,
):

    groups = defaultdict(
        list
    )

    for row in rows:

        groups[
            (
                row[
                    "checkpoint"
                ],
                row[
                    "calibration"
                ],
            )
        ].append(
            row
        )

    print()
    print("=" * 120)
    print("VALIDATION SUMMARY")
    print("=" * 120)

    print(
        f"{'checkpoint':12s}"
        f"{'calibration':18s}"
        f"{'N':>7s}"
        f"{'NMSE':>14s}"
        f"{'PSNR':>14s}"
        f"{'SSIM':>14s}"
        f"{'norm ratio':>16s}"
    )

    print("-" * 120)

    for (
        checkpoint,
        calibration,
    ), group in sorted(
        groups.items()
    ):

        n = len(
            group
        )

        nmse = np.asarray(
            [
                r["nmse"]
                for r in group
            ],
            dtype=np.float64,
        )

        psnr = np.asarray(
            [
                r["psnr"]
                for r in group
            ],
            dtype=np.float64,
        )

        ssim = np.asarray(
            [
                r["ssim"]
                for r in group
            ],
            dtype=np.float64,
        )

        ratio = np.asarray(
            [
                r["norm_ratio"]
                for r in group
            ],
            dtype=np.float64,
        )

        print(
            f"{checkpoint:12s}"
            f"{calibration:18s}"
            f"{n:7d}"
            f"{nmse.mean():14.6f}"
            f"{psnr.mean():14.3f}"
            f"{ssim.mean():14.4f}"
            f"{ratio.mean():16.4f}"
        )


def main():

    args = parse_args()

    if not (
        0
        <= args.shard_id
        < args.num_shards
    ):
        raise ValueError(
            "Invalid shard_id."
        )

    device = torch.device(
        args.device
    )

    checkpoints = (
        parse_checkpoints(
            args.checkpoints
        )
    )

    print("=" * 120)
    print("SHAMAEI VALIDATION CALIBRATION SWEEP")
    print("=" * 120)

    print(
        "acceleration:",
        args.acceleration,
    )

    print(
        "checkpoints:",
        checkpoints,
    )

    print(
        "device:",
        device,
    )

    print(
        "shard:",
        args.shard_id,
        "/",
        args.num_shards,
    )

    dataset = (
        FastMRIBrainDataset(
            root=
                args.data_root,

            acquisition=
                "AXT2",

            cache_dir=
                "code/.cache",
        )
    )

    lookup = (
        build_dataset_lookup(
            dataset
        )
    )

    entries = (
        get_manifest_entries(
            args.manifest
        )
    )

    entries = [
        entry
        for i, entry
        in enumerate(
            entries
        )
        if (
            i
            % args.num_shards
            == args.shard_id
        )
    ]

    if (
        args.sample_count
        is not None
    ):

        if (
            args.sample_count
            > len(entries)
        ):
            raise ValueError(
                "--sample_count exceeds "
                "available validation samples."
            )

        positions = np.linspace(
            0,
            len(entries) - 1,
            num=args.sample_count,
            dtype=int,
        )

        entries = [
            entries[
                int(position)
            ]
            for position
            in positions
        ]

    elif (
        args.limit_samples
        is not None
    ):

        entries = (
            entries[
                :
                args.limit_samples
            ]
        )

    print(
        "validation samples:",
        len(entries),
    )

    models = {
        label:
            load_model(
                checkpoint_path=
                    path,

                acceleration=
                    args.acceleration,

                device=
                    device,
            )
        for label, path
        in checkpoints.items()
    }

    rows = []

    for entry in tqdm(
        entries,
        desc=(
            f"Shamaei R"
            f"{args.acceleration}"
        ),
    ):

        fname = (
            entry_fname(
                entry
            )
        )

        slice_num = (
            entry_slice(
                entry
            )
        )

        key = (
            fname,
            slice_num,
        )

        if key not in lookup:

            raise RuntimeError(
                "Manifest entry not "
                "found in dataset: "
                f"{key}"
            )

        sample = (
            dataset[
                lookup[
                    key
                ]
            ]
        )

        (
            sample_256,
            masked_kspace,
            raw_mask,
            num_low,
        ) = (
            build_fastmri_256_problem(
                sample,
                args.acceleration,
            )
        )

        masked_kspace = (
            masked_kspace
            .to(
                device,
                dtype=
                    torch.complex64,
            )
        )

        target = (
            sample_256[
                "target"
            ]
            .to(
                device,
                dtype=
                    torch.float32,
            )
        )

        common_mask = (
            prepare_mask(
                raw_mask,
                width=256,
                device=device,
            )
        )

        sens_maps = (
            estimate_sens_espirit(
                masked_kspace,
                num_low_frequencies=
                    num_low,
                crop=0.0,
            )
            .to(
                device,
                dtype=
                    torch.complex64,
            )
        )

        for (
            checkpoint_label,
            model,
        ) in models.items():

            (
                predictions,
                diagnostics,
            ) = (
                reconstruct_all_calibrations(
                    model=
                        model,

                    masked_kspace=
                        masked_kspace,

                    common_mask=
                        common_mask,

                    sens_maps=
                        sens_maps,
                )
            )

            for calibration in (
                CALIBRATIONS
            ):

                metrics = (
                    metric_one_slice(
                        target=
                            target,

                        prediction=
                            predictions[
                                calibration
                            ],
                    )
                )

                rows.append(
                    {
                        "acceleration":
                            args.acceleration,

                        "checkpoint":
                            checkpoint_label,

                        "calibration":
                            calibration,

                        "fname":
                            fname,

                        "slice_num":
                            slice_num,

                        "nmse":
                            metrics[
                                "nmse"
                            ],

                        "psnr":
                            metrics[
                                "psnr"
                            ],

                        "ssim":
                            metrics[
                                "ssim"
                            ],

                        "norm_ratio":
                            metrics[
                                "norm_ratio"
                            ],

                        "measurement_scale":
                            diagnostics[
                                "measurement_scale"
                            ],

                        "complex_coherence":
                            diagnostics[
                                "complex_coherence"
                            ],

                        "calibration_scale":
                            diagnostics[
                                f"scale_"
                                f"{calibration}"
                            ],
                    }
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(
        args.output,
        rows,
    )

    print_summary(
        rows
    )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
