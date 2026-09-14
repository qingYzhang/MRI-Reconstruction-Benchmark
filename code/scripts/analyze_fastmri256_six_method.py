import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_FILES = {
    4: Path(
        "outputs/"
        "fastmri256_tuning30_r4.csv"
    ),
    8: Path(
        "outputs/"
        "fastmri256_tuning30_r8.csv"
    ),
}

SHAMAEI_FILES = {
    4: Path(
        "outputs/"
        "shamaei_fastmri256_tuning30_r4.csv"
    ),
    8: Path(
        "outputs/"
        "shamaei_fastmri256_tuning30_r8.csv"
    ),
}

METHOD_ORDER = [
    "zf",
    "cg",
    "lacs",
    "nerp",
    "caps",
    "shamaei_e2e_varnet",
]

DISPLAY_NAMES = {
    "zf":
        "ZF",

    "cg":
        "CG",

    "lacs":
        "LACS",

    "nerp":
        "NeRP",

    "caps":
        "CAPS",

    "shamaei_e2e_varnet":
        "Shamaei E2E-VarNet",
}

METRICS = [
    "nmse",
    "psnr",
    "ssim",
]

#
# Keep the same convention as the
# existing fastMRI-256 tuning30 analysis:
# population standard deviation.
#
STD_DDOF = 0


def read_csv(path):

    with path.open() as f:

        return list(
            csv.DictReader(f)
        )


def audit_and_merge(
    acceleration,
):

    base = read_csv(
        BASE_FILES[
            acceleration
        ]
    )

    shamaei = read_csv(
        SHAMAEI_FILES[
            acceleration
        ]
    )

    if len(base) != 150:

        raise RuntimeError(
            f"R{acceleration}: "
            f"expected 150 base rows, "
            f"got {len(base)}"
        )

    if len(shamaei) != 30:

        raise RuntimeError(
            f"R{acceleration}: "
            f"expected 30 Shamaei rows, "
            f"got {len(shamaei)}"
        )

    by_method = defaultdict(
        list
    )

    for row in (
        base
        + shamaei
    ):

        method = row[
            "method"
        ]

        by_method[
            method
        ].append(
            row
        )

    missing_methods = [
        m
        for m in METHOD_ORDER
        if m not in by_method
    ]

    if missing_methods:

        raise RuntimeError(
            f"R{acceleration}: "
            f"missing methods "
            f"{missing_methods}"
        )

    reference_fnames = None

    for method in METHOD_ORDER:

        rows = (
            by_method[
                method
            ]
        )

        if len(rows) != 30:

            raise RuntimeError(
                f"R{acceleration} "
                f"{method}: "
                f"expected 30 rows, "
                f"got {len(rows)}"
            )

        fnames = [
            row[
                "fname"
            ]
            for row in rows
        ]

        if len(
            set(
                fnames
            )
        ) != 30:

            raise RuntimeError(
                f"R{acceleration} "
                f"{method}: "
                "duplicate volumes"
            )

        fname_set = set(
            fnames
        )

        if (
            reference_fnames
            is None
        ):

            reference_fnames = (
                fname_set
            )

        elif (
            fname_set
            != reference_fnames
        ):

            raise RuntimeError(
                f"R{acceleration} "
                f"{method}: "
                "cohort mismatch"
            )

    print(
        f"R{acceleration} AUDIT: "
        "6 methods × 30 volumes, "
        "exact paired cohort"
    )

    return {
        method: {
            row[
                "fname"
            ]:
                row
            for row in
                by_method[
                    method
                ]
        }
        for method in
            METHOD_ORDER
    }


def metric_array(
    data,
    method,
    metric,
    fnames,
):

    return np.asarray(
        [
            float(
                data[
                    method
                ][
                    fname
                ][
                    metric
                ]
            )
            for fname
            in fnames
        ],
        dtype=np.float64,
    )


def summarize(
    acceleration,
    data,
):

    fnames = sorted(
        data[
            METHOD_ORDER[
                0
            ]
        ].keys()
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"FASTMRI-256 ADAPTATION "
        f"TUNING30 — R{acceleration}"
    )

    print(
        f"SD convention: "
        f"ddof={STD_DDOF}"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Method':<24}"
        f"{'NMSE':>22}"
        f"{'PSNR':>22}"
        f"{'SSIM':>22}"
    )

    print(
        "-" * 90
    )

    summary_rows = []

    for method in METHOD_ORDER:

        values = {}

        for metric in METRICS:

            arr = metric_array(
                data,
                method,
                metric,
                fnames,
            )

            values[
                metric
            ] = {
                "mean":
                    float(
                        arr.mean()
                    ),

                "std":
                    float(
                        arr.std(
                            ddof=
                                STD_DDOF
                        )
                    ),
            }

        print(
            f"{DISPLAY_NAMES[method]:<24}"
            f"{values['nmse']['mean']:>10.6f}"
            f" ± "
            f"{values['nmse']['std']:<9.6f}"
            f"{values['psnr']['mean']:>8.3f}"
            f" ± "
            f"{values['psnr']['std']:<9.3f}"
            f"{values['ssim']['mean']:>8.4f}"
            f" ± "
            f"{values['ssim']['std']:<9.4f}"
        )

        summary_rows.append(
            {
                "acceleration":
                    acceleration,

                "method":
                    method,

                "display_name":
                    DISPLAY_NAMES[
                        method
                    ],

                "num_volumes":
                    len(
                        fnames
                    ),

                "nmse_mean":
                    values[
                        "nmse"
                    ][
                        "mean"
                    ],

                "nmse_std":
                    values[
                        "nmse"
                    ][
                        "std"
                    ],

                "psnr_mean":
                    values[
                        "psnr"
                    ][
                        "mean"
                    ],

                "psnr_std":
                    values[
                        "psnr"
                    ][
                        "std"
                    ],

                "ssim_mean":
                    values[
                        "ssim"
                    ][
                        "mean"
                    ],

                "ssim_std":
                    values[
                        "ssim"
                    ][
                        "std"
                    ],

                "std_ddof":
                    STD_DDOF,
            }
        )

    return (
        fnames,
        summary_rows,
    )


def paired_shamaei_comparison(
    acceleration,
    data,
    fnames,
):

    print()
    print(
        "=" * 100
    )

    print(
        f"PAIRED SHAMAEI COMPARISON "
        f"— R{acceleration}"
    )

    print(
        "=" * 100
    )

    shamaei = {
        metric:
            metric_array(
                data,
                "shamaei_e2e_varnet",
                metric,
                fnames,
            )
        for metric
        in METRICS
    }

    paired_rows = []

    for baseline in [
        "zf",
        "cg",
        "lacs",
        "nerp",
        "caps",
    ]:

        base = {
            metric:
                metric_array(
                    data,
                    baseline,
                    metric,
                    fnames,
                )
            for metric
            in METRICS
        }

        #
        # Positive delta means Shamaei
        # has a larger numerical value.
        #
        delta_nmse = (
            shamaei[
                "nmse"
            ]
            - base[
                "nmse"
            ]
        )

        delta_psnr = (
            shamaei[
                "psnr"
            ]
            - base[
                "psnr"
            ]
        )

        delta_ssim = (
            shamaei[
                "ssim"
            ]
            - base[
                "ssim"
            ]
        )

        #
        # Win convention:
        # NMSE lower is better.
        # PSNR/SSIM higher is better.
        #
        nmse_wins = int(
            np.sum(
                shamaei[
                    "nmse"
                ]
                <
                base[
                    "nmse"
                ]
            )
        )

        psnr_wins = int(
            np.sum(
                shamaei[
                    "psnr"
                ]
                >
                base[
                    "psnr"
                ]
            )
        )

        ssim_wins = int(
            np.sum(
                shamaei[
                    "ssim"
                ]
                >
                base[
                    "ssim"
                ]
            )
        )

        print()
        print(
            "Shamaei vs",
            DISPLAY_NAMES[
                baseline
            ],
        )

        print(
            "  ΔNMSE "
            f"{delta_nmse.mean():+.6f} "
            f"| wins "
            f"{nmse_wins}/30"
        )

        print(
            "  ΔPSNR "
            f"{delta_psnr.mean():+.3f} dB "
            f"| wins "
            f"{psnr_wins}/30"
        )

        print(
            "  ΔSSIM "
            f"{delta_ssim.mean():+.4f} "
            f"| wins "
            f"{ssim_wins}/30"
        )

        paired_rows.append(
            {
                "acceleration":
                    acceleration,

                "comparison":
                    (
                        "shamaei_e2e_varnet"
                        f"_vs_{baseline}"
                    ),

                "baseline":
                    baseline,

                "delta_nmse_mean":
                    float(
                        delta_nmse.mean()
                    ),

                "delta_psnr_mean":
                    float(
                        delta_psnr.mean()
                    ),

                "delta_ssim_mean":
                    float(
                        delta_ssim.mean()
                    ),

                "nmse_wins":
                    nmse_wins,

                "psnr_wins":
                    psnr_wins,

                "ssim_wins":
                    ssim_wins,

                "num_volumes":
                    len(
                        fnames
                    ),
            }
        )

    return paired_rows


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

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    all_summary = []
    all_paired = []

    for acceleration in [
        4,
        8,
    ]:

        data = audit_and_merge(
            acceleration
        )

        (
            fnames,
            summary_rows,
        ) = summarize(
            acceleration,
            data,
        )

        paired_rows = (
            paired_shamaei_comparison(
                acceleration,
                data,
                fnames,
            )
        )

        all_summary.extend(
            summary_rows
        )

        all_paired.extend(
            paired_rows
        )

    write_csv(
        "outputs/"
        "fastmri256_tuning30_"
        "six_method_summary.csv",
        all_summary,
    )

    write_csv(
        "outputs/"
        "fastmri256_tuning30_"
        "shamaei_paired.csv",
        all_paired,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SIX-METHOD ANALYSIS PASSED"
    )

    print(
        "=" * 100
    )

    print(
        "saved: "
        "outputs/"
        "fastmri256_tuning30_"
        "six_method_summary.csv"
    )

    print(
        "saved: "
        "outputs/"
        "fastmri256_tuning30_"
        "shamaei_paired.csv"
    )


if __name__ == "__main__":
    main()
