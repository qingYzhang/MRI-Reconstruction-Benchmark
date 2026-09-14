import json
import random
import sys
from collections import defaultdict
from pathlib import Path


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


SEED = 20260904
BENCHMARK_VOLUMES = 30

TRAIN_ROOT = (
    "extracted/"
    "multicoil_train"
)

VAL_ROOT = (
    "extracted/"
    "multicoil_val"
)

ACQUISITION = "AXT2"

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_split.json"
)


def get_fname(raw_sample):

    if hasattr(
        raw_sample,
        "fname",
    ):
        return Path(
            raw_sample.fname
        ).name

    if isinstance(
        raw_sample,
        tuple,
    ):
        return Path(
            raw_sample[0]
        ).name

    raise RuntimeError(
        f"Cannot parse raw sample: "
        f"{raw_sample}"
    )


def build_volume_map(
    dataset,
):

    volume_map = (
        defaultdict(list)
    )

    raw_samples = (
        dataset.raw_samples
        if hasattr(
            dataset,
            "raw_samples",
        )
        else dataset.examples
    )

    for idx, raw_sample in enumerate(
        raw_samples
    ):

        fname = get_fname(
            raw_sample
        )

        volume_map[
            fname
        ].append(
            idx
        )

    return dict(
        volume_map
    )


def count_slices(
    volume_map,
    fnames,
):

    return sum(
        len(
            volume_map[
                fname
            ]
        )
        for fname in fnames
    )


def main():

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "DATA SPLIT GATE"
    )

    print(
        "=" * 90
    )

    train_dataset = (
        FastMRIBrainDataset(
            root=
                TRAIN_ROOT,

            acquisition=
                ACQUISITION,

            cache_dir=
                "code/.cache",
        )
    )

    val_dataset = (
        FastMRIBrainDataset(
            root=
                VAL_ROOT,

            acquisition=
                ACQUISITION,

            cache_dir=
                "code/.cache",
        )
    )

    train_map = (
        build_volume_map(
            train_dataset
        )
    )

    val_map = (
        build_volume_map(
            val_dataset
        )
    )

    all_train_fnames = sorted(
        train_map.keys()
    )

    all_val_fnames = sorted(
        val_map.keys()
    )

    #
    # EXACT SAME selection used by
    # our frozen tuning30 benchmark.
    #
    rng = random.Random(
        SEED
    )

    benchmark_fnames = (
        rng.sample(
            all_train_fnames,
            min(
                BENCHMARK_VOLUMES,
                len(
                    all_train_fnames
                ),
            ),
        )
    )

    benchmark_fnames = sorted(
        benchmark_fnames
    )

    benchmark_set = set(
        benchmark_fnames
    )

    train_fnames = [
        fname
        for fname
        in all_train_fnames
        if fname
        not in benchmark_set
    ]

    val_fnames = list(
        all_val_fnames
    )

    #
    # Hard leakage checks.
    #
    assert (
        set(
            train_fnames
        )
        .isdisjoint(
            benchmark_set
        )
    )

    assert (
        set(
            val_fnames
        )
        .isdisjoint(
            benchmark_set
        )
    )

    assert (
        set(
            train_fnames
        )
        .isdisjoint(
            set(
                val_fnames
            )
        )
    )

    assert (
        len(
            benchmark_fnames
        )
        == 30
    )

    train_slices = (
        count_slices(
            train_map,
            train_fnames,
        )
    )

    benchmark_slices = (
        count_slices(
            train_map,
            benchmark_fnames,
        )
    )

    val_slices = (
        count_slices(
            val_map,
            val_fnames,
        )
    )

    result = {
        "seed":
            SEED,

        "acquisition":
            ACQUISITION,

        "train_root":
            TRAIN_ROOT,

        "val_root":
            VAL_ROOT,

        "train": {
            "num_volumes":
                len(
                    train_fnames
                ),

            "num_slices":
                train_slices,

            "fnames":
                train_fnames,
        },

        "validation": {
            "num_volumes":
                len(
                    val_fnames
                ),

            "num_slices":
                val_slices,

            "fnames":
                val_fnames,
        },

        "benchmark": {
            "name":
                "tuning30",

            "num_volumes":
                len(
                    benchmark_fnames
                ),

            "num_slices":
                benchmark_slices,

            "fnames":
                benchmark_fnames,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Train volumes:",
        len(
            train_fnames
        ),
    )

    print(
        "Train slices:",
        train_slices,
    )

    print(
        "Validation volumes:",
        len(
            val_fnames
        ),
    )

    print(
        "Validation slices:",
        val_slices,
    )

    print(
        "Benchmark volumes:",
        len(
            benchmark_fnames
        ),
    )

    print(
        "Benchmark slices:",
        benchmark_slices,
    )

    print()
    print(
        "Benchmark cohort:"
    )

    for fname in (
        benchmark_fnames
    ):
        print(
            " ",
            fname,
        )

    print()
    print(
        "train ∩ benchmark:",
        len(
            set(
                train_fnames
            )
            &
            benchmark_set
        ),
    )

    print(
        "val ∩ benchmark:",
        len(
            set(
                val_fnames
            )
            &
            benchmark_set
        ),
    )

    print(
        "train ∩ val:",
        len(
            set(
                train_fnames
            )
            &
            set(
                val_fnames
            )
        ),
    )

    print()
    print(
        "saved:",
        OUTPUT,
    )

    print(
        "\n"
        "SHAMAEI DATA SPLIT "
        "GATE PASSED"
    )


if __name__ == "__main__":
    main()
