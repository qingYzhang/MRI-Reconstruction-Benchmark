import json
import os
import random
import shutil
import sys
from pathlib import Path

import h5py


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CODE_ROOT = (
    PROJECT_ROOT
    / "code"
)

sys.path.insert(
    0,
    str(CODE_ROOT),
)


from data.fastmri_brain import (
    FastMRIBrainDataset,
)


GRID_SIZE = 256

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_training_manifest.json"
)

BACKUP_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_training_manifest_pre_grid_fix.json"
)

REPLACEMENT_SEED_OFFSET = 2000


def get_raw_samples(dataset):

    if hasattr(
        dataset,
        "raw_samples"
    ):
        return dataset.raw_samples

    if hasattr(
        dataset,
        "examples"
    ):
        return dataset.examples

    raise RuntimeError(
        "Dataset has neither raw_samples nor examples."
    )


def raw_path(raw_sample):

    if hasattr(
        raw_sample,
        "fname"
    ):
        return Path(
            raw_sample.fname
        )

    if isinstance(
        raw_sample,
        tuple
    ):
        return Path(
            raw_sample[0]
        )

    raise RuntimeError(
        f"Cannot parse raw sample: {raw_sample}"
    )


def raw_slice_num(raw_sample):

    if hasattr(
        raw_sample,
        "slice_ind"
    ):
        return int(
            raw_sample.slice_ind
        )

    if isinstance(
        raw_sample,
        tuple
    ):
        return int(
            raw_sample[1]
        )

    raise RuntimeError(
        f"Cannot parse slice number: {raw_sample}"
    )


class ShapeCache:

    def __init__(self):
        self.cache = {}

    def get(self, path):

        path = Path(path)

        if path not in self.cache:

            with h5py.File(
                path,
                "r"
            ) as f:

                shape = tuple(
                    int(x)
                    for x in
                    f["kspace"].shape
                )

            if len(shape) != 4:
                raise RuntimeError(
                    f"Unexpected kspace shape "
                    f"for {path}: {shape}"
                )

            self.cache[
                path
            ] = (
                shape[-2],
                shape[-1],
            )

        return self.cache[
            path
        ]


def compatible(
    h,
    w,
):

    return (
        h >= GRID_SIZE
        and w >= GRID_SIZE
    )


def check_entry(
    dataset,
    entry,
    shape_cache,
):

    raw_samples = (
        get_raw_samples(
            dataset
        )
    )

    idx = int(
        entry[
            "dataset_index"
        ]
    )

    raw = raw_samples[
        idx
    ]

    path = raw_path(
        raw
    )

    fname = path.name

    slice_num = (
        raw_slice_num(
            raw
        )
    )

    if (
        fname
        != entry[
            "fname"
        ]
    ):
        raise RuntimeError(
            "Manifest filename mismatch: "
            f"{fname} != "
            f"{entry['fname']}"
        )

    if (
        slice_num
        != int(
            entry[
                "slice_num"
            ]
        )
    ):
        raise RuntimeError(
            "Manifest slice mismatch."
        )

    h, w = shape_cache.get(
        path
    )

    return (
        compatible(
            h,
            w
        ),
        h,
        w,
    )


def candidate_from_raw(
    idx,
    raw,
):

    return {
        "dataset_index":
            int(idx),

        "fname":
            raw_path(
                raw
            ).name,

        "slice_num":
            raw_slice_num(
                raw
            ),
    }


def main():

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "MANIFEST GRID-COMPATIBILITY REPAIR"
    )

    print(
        "=" * 90
    )

    with MANIFEST_PATH.open() as f:

        manifest = json.load(
            f
        )

    #
    # Preserve exact original manifest.
    #
    if not BACKUP_PATH.exists():

        shutil.copy2(
            MANIFEST_PATH,
            BACKUP_PATH,
        )

        print(
            "backup created:",
            BACKUP_PATH,
        )

    else:

        print(
            "backup already exists:",
            BACKUP_PATH,
        )

    train_dataset = (
        FastMRIBrainDataset(
            root=(
                "extracted/"
                "multicoil_train"
            ),
            acquisition="AXT2",
            cache_dir="code/.cache",
        )
    )

    val_dataset = (
        FastMRIBrainDataset(
            root=(
                "extracted/"
                "multicoil_val"
            ),
            acquisition="AXT2",
            cache_dir="code/.cache",
        )
    )

    shape_cache = ShapeCache()

    #
    # ----------------------------------------------------------
    # Audit TRAIN.
    #
    # This should be zero because all 6708
    # train cache entries already completed.
    # ----------------------------------------------------------
    #

    invalid_train = []

    for position, entry in enumerate(
        manifest[
            "train"
        ][
            "samples"
        ]
    ):

        ok, h, w = check_entry(
            train_dataset,
            entry,
            shape_cache,
        )

        if not ok:

            invalid_train.append(
                (
                    position,
                    entry,
                    h,
                    w,
                )
            )

    print()
    print(
        "invalid train samples:",
        len(
            invalid_train
        ),
    )

    if invalid_train:

        for (
            position,
            entry,
            h,
            w,
        ) in invalid_train[:20]:

            print(
                "  ",
                position,
                entry[
                    "fname"
                ],
                "slice=",
                entry[
                    "slice_num"
                ],
                "shape=",
                (
                    h,
                    w,
                ),
            )

        raise RuntimeError(
            "Training manifest contains "
            "incompatible samples. "
            "Do not modify it automatically."
        )

    #
    # ----------------------------------------------------------
    # Find invalid VALIDATION entries.
    # ----------------------------------------------------------
    #

    val_entries = (
        manifest[
            "validation"
        ][
            "samples"
        ]
    )

    invalid_val = []

    for position, entry in enumerate(
        val_entries
    ):

        ok, h, w = check_entry(
            val_dataset,
            entry,
            shape_cache,
        )

        if not ok:

            invalid_val.append(
                (
                    position,
                    entry,
                    h,
                    w,
                )
            )

    print(
        "invalid validation samples:",
        len(
            invalid_val
        ),
    )

    for (
        position,
        entry,
        h,
        w,
    ) in invalid_val:

        print(
            f"  position={position:4d} "
            f"{entry['fname']} "
            f"slice={entry['slice_num']} "
            f"native=({h},{w})"
        )

    if not invalid_val:

        print()
        print(
            "No repair necessary."
        )
        return

    #
    # ----------------------------------------------------------
    # Build deterministic replacement pool.
    #
    # IMPORTANT:
    #
    # - only validation data
    # - native H,W >= 256
    # - exclude every sample already selected
    # - no benchmark/train leakage possible because this is
    #   the separate official fastMRI validation root
    # ----------------------------------------------------------
    #

    selected_keys = {
        (
            entry[
                "fname"
            ],
            int(
                entry[
                    "slice_num"
                ]
            ),
        )
        for entry
        in val_entries
    }

    raw_samples = (
        get_raw_samples(
            val_dataset
        )
    )

    replacement_pool = []

    for idx, raw in enumerate(
        raw_samples
    ):

        path = raw_path(
            raw
        )

        h, w = shape_cache.get(
            path
        )

        if not compatible(
            h,
            w,
        ):
            continue

        key = (
            path.name,
            raw_slice_num(
                raw
            ),
        )

        if key in selected_keys:
            continue

        replacement_pool.append(
            candidate_from_raw(
                idx,
                raw,
            )
        )

    print()
    print(
        "eligible unused "
        "validation pool:",
        len(
            replacement_pool
        ),
    )

    if (
        len(
            replacement_pool
        )
        <
        len(
            invalid_val
        )
    ):

        raise RuntimeError(
            "Not enough compatible "
            "replacement samples."
        )

    base_seed = int(
        manifest[
            "seed"
        ]
    )

    replacement_seed = (
        base_seed
        + REPLACEMENT_SEED_OFFSET
    )

    rng = random.Random(
        replacement_seed
    )

    replacements = rng.sample(
        replacement_pool,
        len(
            invalid_val
        ),
    )

    #
    # Replace IN PLACE.
    #
    # This preserves original list positions,
    # therefore preserves 8-way shard assignment.
    #
    repair_records = []

    for (
        invalid_item,
        replacement,
    ) in zip(
        invalid_val,
        replacements,
    ):

        (
            position,
            old_entry,
            old_h,
            old_w,
        ) = invalid_item

        new_raw = (
            raw_samples[
                replacement[
                    "dataset_index"
                ]
            ]
        )

        new_h, new_w = (
            shape_cache.get(
                raw_path(
                    new_raw
                )
            )
        )

        assert compatible(
            new_h,
            new_w,
        )

        val_entries[
            position
        ] = replacement

        repair_records.append(
            {
                "position":
                    position,

                "old": {
                    **old_entry,
                    "native_height":
                        old_h,
                    "native_width":
                        old_w,
                },

                "new": {
                    **replacement,
                    "native_height":
                        new_h,
                    "native_width":
                        new_w,
                },
            }
        )

    #
    # ----------------------------------------------------------
    # Final integrity checks.
    # ----------------------------------------------------------
    #

    keys = [
        (
            entry[
                "fname"
            ],
            int(
                entry[
                    "slice_num"
                ]
            ),
        )
        for entry
        in val_entries
    ]

    if (
        len(
            keys
        )
        != len(
            set(
                keys
            )
        )
    ):

        raise RuntimeError(
            "Duplicate validation "
            "samples after repair."
        )

    remaining_invalid = []

    for position, entry in enumerate(
        val_entries
    ):

        ok, h, w = check_entry(
            val_dataset,
            entry,
            shape_cache,
        )

        if not ok:

            remaining_invalid.append(
                (
                    position,
                    entry,
                    h,
                    w,
                )
            )

    if remaining_invalid:

        raise RuntimeError(
            "Manifest still contains "
            "incompatible validation "
            "samples after repair."
        )

    benchmark_fnames = set(
        manifest[
            "benchmark"
        ][
            "fnames"
        ]
    )

    val_fnames = {
        entry[
            "fname"
        ]
        for entry
        in val_entries
    }

    if (
        val_fnames
        & benchmark_fnames
    ):

        raise RuntimeError(
            "Validation/benchmark leakage."
        )

    manifest[
        "validation"
    ][
        "num_volumes_represented"
    ] = len(
        val_fnames
    )

    manifest[
        "fastmri_adaptation"
    ][
        "native_grid_requirement"
    ] = (
        "native k-space spatial "
        "dimensions H>=256 and W>=256; "
        "then image-domain center crop "
        "to 256x256"
    )

    manifest[
        "grid_compatibility_repair"
    ] = {
        "reason":
            (
                "fastMRI-256 adaptation "
                "uses image-domain center crop; "
                "native dimensions smaller "
                "than 256 are excluded"
            ),

        "replacement_seed":
            replacement_seed,

        "num_replaced_validation_slices":
            len(
                repair_records
            ),

        "positions_preserved":
            True,

        "records":
            repair_records,
    }

    #
    # Atomic manifest write.
    #
    tmp = (
        MANIFEST_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    with tmp.open(
        "w"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    os.replace(
        tmp,
        MANIFEST_PATH,
    )

    print()
    print(
        "replaced validation slices:",
        len(
            repair_records
        ),
    )

    print(
        "validation volumes represented:",
        len(
            val_fnames
        ),
    )

    print(
        "replacement seed:",
        replacement_seed,
    )

    print(
        "manifest saved:",
        MANIFEST_PATH,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "MANIFEST REPAIR PASSED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
