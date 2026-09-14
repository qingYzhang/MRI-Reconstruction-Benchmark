import json
import random
import sys
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


SEED = 12345

SOURCE_TRAIN_SLICES = 6708
SOURCE_VAL_SLICES = 2808

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_split.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "shamaei_fastmri256_training_manifest.json"
)


def get_raw_samples(dataset):

    if hasattr(dataset, "raw_samples"):
        return dataset.raw_samples

    if hasattr(dataset, "examples"):
        return dataset.examples

    raise RuntimeError(
        "Dataset has neither raw_samples nor examples."
    )


def raw_fname(raw_sample):

    if hasattr(raw_sample, "fname"):
        return Path(
            raw_sample.fname
        ).name

    return Path(
        raw_sample[0]
    ).name


def build_candidates(
    dataset,
    allowed_fnames,
):

    allowed_fnames = set(
        allowed_fnames
    )

    candidates = []

    raw_samples = get_raw_samples(
        dataset
    )

    for idx, raw_sample in enumerate(
        raw_samples
    ):

        fname = raw_fname(
            raw_sample
        )

        if fname not in allowed_fnames:
            continue

        sample = dataset[idx]

        candidates.append(
            {
                "dataset_index":
                    idx,

                "fname":
                    sample["fname"],

                "slice_num":
                    int(
                        sample["slice_num"]
                    ),
            }
        )

    return candidates


def choose(
    candidates,
    count,
    seed,
):

    if len(candidates) < count:
        raise RuntimeError(
            f"Need {count} samples, "
            f"only have {len(candidates)}."
        )

    rng = random.Random(
        seed
    )

    chosen = rng.sample(
        candidates,
        count,
    )

    return sorted(
        chosen,
        key=lambda x: (
            x["fname"],
            x["slice_num"],
        ),
    )


def main():

    print(
        "=" * 90
    )

    print(
        "SHAMAEI FASTMRI-256 "
        "FORMAL TRAINING MANIFEST"
    )

    print(
        "=" * 90
    )

    with SPLIT_PATH.open() as f:

        split = json.load(
            f
        )

    train_fnames = set(
        split["train"]["fnames"]
    )

    val_fnames = set(
        split["validation"]["fnames"]
    )

    benchmark_fnames = set(
        split["benchmark"]["fnames"]
    )

    assert train_fnames.isdisjoint(
        benchmark_fnames
    )

    assert val_fnames.isdisjoint(
        benchmark_fnames
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

    train_candidates = (
        build_candidates(
            train_dataset,
            train_fnames,
        )
    )

    val_candidates = (
        build_candidates(
            val_dataset,
            val_fnames,
        )
    )

    print(
        "available train slices:",
        len(train_candidates),
    )

    print(
        "available val slices:",
        len(val_candidates),
    )

    train_samples = choose(
        train_candidates,
        SOURCE_TRAIN_SLICES,
        SEED,
    )

    val_samples = choose(
        val_candidates,
        SOURCE_VAL_SLICES,
        SEED + 1,
    )

    train_selected_fnames = {
        x["fname"]
        for x in train_samples
    }

    val_selected_fnames = {
        x["fname"]
        for x in val_samples
    }

    assert train_selected_fnames.isdisjoint(
        benchmark_fnames
    )

    assert val_selected_fnames.isdisjoint(
        benchmark_fnames
    )

    assert train_selected_fnames.isdisjoint(
        val_selected_fnames
    )

    result = {
        "seed":
            SEED,

        "selection":
            "slice_count_matched_to_shamaei_paper",

        "paper_reference_budget": {
            "train_slices":
                SOURCE_TRAIN_SLICES,

            "validation_slices":
                SOURCE_VAL_SLICES,

            "max_epochs":
                200,

            "optimizer":
                "Adam",

            "learning_rate":
                1e-3,

            "effective_batch_size":
                64,

            "loss":
                "SSIMLoss",

            "early_stopping_patience":
                10,
        },

        "fastmri_adaptation": {
            "grid":
                "256x256",

            "acquisition":
                "AXT2",

            "benchmark_excluded":
                True,

            "measurement_scale":
                "ZF_RSS_q0.999",

            "output_scale_calibration":
                "acquired_kspace_complex_least_squares",

            "sensitivity":
                "ESPIRiT_crop0",
        },

        "train": {
            "num_slices":
                len(train_samples),

            "num_volumes_represented":
                len(
                    train_selected_fnames
                ),

            "samples":
                train_samples,
        },

        "validation": {
            "num_slices":
                len(val_samples),

            "num_volumes_represented":
                len(
                    val_selected_fnames
                ),

            "samples":
                val_samples,
        },

        "benchmark": {
            "num_volumes":
                len(
                    benchmark_fnames
                ),

            "fnames":
                sorted(
                    benchmark_fnames
                ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "selected train slices:",
        len(train_samples),
    )

    print(
        "train volumes represented:",
        len(
            train_selected_fnames
        ),
    )

    print(
        "selected val slices:",
        len(val_samples),
    )

    print(
        "val volumes represented:",
        len(
            val_selected_fnames
        ),
    )

    print()
    print(
        "train ∩ benchmark:",
        len(
            train_selected_fnames
            & benchmark_fnames
        ),
    )

    print(
        "val ∩ benchmark:",
        len(
            val_selected_fnames
            & benchmark_fnames
        ),
    )

    print(
        "train ∩ val:",
        len(
            train_selected_fnames
            & val_selected_fnames
        ),
    )

    print()
    print(
        "saved:",
        OUTPUT_PATH,
    )

    print()
    print(
        "SHAMAEI FORMAL "
        "TRAINING MANIFEST PASSED"
    )


if __name__ == "__main__":
    main()
