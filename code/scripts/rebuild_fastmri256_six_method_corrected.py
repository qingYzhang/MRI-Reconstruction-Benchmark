#!/usr/bin/env python3

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("outputs")

BASE_FILES = {
    4: ROOT / "fastmri256_tuning30_r4.csv",
    8: ROOT / "fastmri256_tuning30_r8.csv",
}

SHAMAEI_FILES = {
    4: ROOT / "shamaei_fastmri256_zfl2_tuning30_r4.csv",
    8: ROOT / "shamaei_fastmri256_zfl2_tuning30_r8.csv",
}

DETAIL_OUT = (
    ROOT
    / "fastmri256_tuning30_six_method_corrected.csv"
)

SUMMARY_OUT = (
    ROOT
    / "fastmri256_tuning30_six_method_summary.csv"
)

OLD_SUMMARY_BACKUP = (
    ROOT
    / "fastmri256_tuning30_six_method_summary_pre_zfl2.csv"
)

METHOD_ORDER = [
    "ZF",
    "CG",
    "LACS",
    "NeRP",
    "CAPS",
    "Shamaei E2E-VarNet",
]


def read_csv(path):
    with path.open() as f:
        return list(
            csv.DictReader(f)
        )


def canonical_method(name):

    s = (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if (
        s == "zf"
        or "zero filled" in s
        or "zero-filled" in s
    ):
        return "ZF"

    if (
        s == "cg"
        or "cg sense" in s
        or "cgsense" in s
    ):
        return "CG"

    if "lacs" in s:
        return "LACS"

    if "nerp" in s:
        return "NeRP"

    if "caps" in s:
        return "CAPS"

    raise RuntimeError(
        f"Unknown base method: {name}"
    )


def metric_value(row, key):
    return float(
        row[key]
    )


all_rows = []

for acceleration in [4, 8]:

    base_rows = read_csv(
        BASE_FILES[
            acceleration
        ]
    )

    shamaei_rows = read_csv(
        SHAMAEI_FILES[
            acceleration
        ]
    )

    base_fnames = {
        row["fname"]
        for row in base_rows
    }

    shamaei_fnames = {
        row["fname"]
        for row in shamaei_rows
    }

    print()
    print("=" * 90)
    print(
        f"R={acceleration}"
    )
    print("=" * 90)

    print(
        "base volumes:",
        len(base_fnames),
    )

    print(
        "Shamaei volumes:",
        len(shamaei_fnames),
    )

    if len(base_fnames) != 30:
        raise RuntimeError(
            f"R{acceleration}: "
            f"expected 30 base volumes, "
            f"got {len(base_fnames)}"
        )

    if len(shamaei_fnames) != 30:
        raise RuntimeError(
            f"R{acceleration}: "
            f"expected 30 Shamaei volumes, "
            f"got {len(shamaei_fnames)}"
        )

    if (
        base_fnames
        != shamaei_fnames
    ):
        raise RuntimeError(
            f"R{acceleration}: "
            "base/Shamaei cohorts differ."
        )

    seen = set()

    for row in base_rows:

        method = canonical_method(
            row["method"]
        )

        key = (
            row["fname"],
            method,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate base row: {key}"
            )

        seen.add(key)

        all_rows.append(
            {
                "acceleration":
                    acceleration,

                "fname":
                    row["fname"],

                "method":
                    method,

                "nmse":
                    metric_value(
                        row,
                        "nmse",
                    ),

                "psnr":
                    metric_value(
                        row,
                        "psnr",
                    ),

                "ssim":
                    metric_value(
                        row,
                        "ssim",
                    ),

                "source":
                    str(
                        BASE_FILES[
                            acceleration
                        ]
                    ),
            }
        )

    for row in shamaei_rows:

        all_rows.append(
            {
                "acceleration":
                    acceleration,

                "fname":
                    row["fname"],

                "method":
                    "Shamaei E2E-VarNet",

                "nmse":
                    metric_value(
                        row,
                        "nmse",
                    ),

                "psnr":
                    metric_value(
                        row,
                        "psnr",
                    ),

                "ssim":
                    metric_value(
                        row,
                        "ssim",
                    ),

                "source":
                    str(
                        SHAMAEI_FILES[
                            acceleration
                        ]
                    ),
            }
        )


#
# Validate 30 volumes x 6 methods x 2 accelerations.
#
expected_rows = (
    30
    * 6
    * 2
)

if (
    len(all_rows)
    != expected_rows
):
    raise RuntimeError(
        f"Expected {expected_rows} "
        f"combined rows, got "
        f"{len(all_rows)}"
    )


#
# Write corrected per-volume combined table.
#
with DETAIL_OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "acceleration",
            "fname",
            "method",
            "nmse",
            "psnr",
            "ssim",
            "source",
        ],
    )

    writer.writeheader()
    writer.writerows(
        all_rows
    )


#
# Preserve old summary before replacing it.
#
if (
    SUMMARY_OUT.exists()
    and not OLD_SUMMARY_BACKUP.exists()
):
    shutil.copy2(
        SUMMARY_OUT,
        OLD_SUMMARY_BACKUP,
    )


grouped = defaultdict(
    list
)

for row in all_rows:

    grouped[
        (
            row[
                "acceleration"
            ],
            row[
                "method"
            ],
        )
    ].append(
        row
    )


summary_rows = []

for acceleration in [4, 8]:

    print()
    print("=" * 90)
    print(
        f"FINAL CORRECTED "
        f"FASTMRI-256 R={acceleration}"
    )
    print("=" * 90)

    print(
        f"{'Method':24s}"
        f"{'NMSE':>22s}"
        f"{'PSNR':>22s}"
        f"{'SSIM':>22s}"
    )

    print("-" * 90)

    for method in METHOD_ORDER:

        rows = grouped[
            (
                acceleration,
                method,
            )
        ]

        if len(rows) != 30:
            raise RuntimeError(
                f"R{acceleration} "
                f"{method}: expected "
                f"30 rows, got {len(rows)}"
            )

        result = {
            "acceleration":
                acceleration,

            "method":
                method,

            "n_volumes":
                len(rows),
        }

        formatted = {}

        for metric in [
            "nmse",
            "psnr",
            "ssim",
        ]:

            x = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row
                    in rows
                ],
                dtype=np.float64,
            )

            result[
                f"{metric}_mean"
            ] = float(
                x.mean()
            )

            result[
                f"{metric}_std"
            ] = float(
                x.std(
                    ddof=0
                )
            )

            formatted[
                metric
            ] = (
                f"{x.mean():.6f} "
                f"+/- "
                f"{x.std(ddof=0):.6f}"
            )

        summary_rows.append(
            result
        )

        print(
            f"{method:24s}"
            f"{formatted['nmse']:>22s}"
            f"{formatted['psnr']:>22s}"
            f"{formatted['ssim']:>22s}"
        )


with SUMMARY_OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "acceleration",
            "method",
            "n_volumes",
            "nmse_mean",
            "nmse_std",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
        ],
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


print()
print(
    "Saved detail:",
    DETAIL_OUT,
)

print(
    "Saved summary:",
    SUMMARY_OUT,
)

print(
    "Old summary backup:",
    OLD_SUMMARY_BACKUP,
)
