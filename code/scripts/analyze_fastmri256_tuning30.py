import csv
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path("outputs")

METHODS = [
    "zf",
    "cg",
    "lacs",
    "nerp",
    "caps",
]

METRICS = [
    "nmse",
    "psnr",
    "ssim",
]


def load(acceleration):
    path = (
        OUTPUT_DIR
        / f"fastmri256_tuning30_r{acceleration}.csv"
    )

    with path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 150

    data = {}

    for row in rows:
        fname = row["fname"]
        method = row["method"]

        data.setdefault(
            fname,
            {},
        )

        assert method not in data[fname]

        data[fname][method] = {
            metric: float(row[metric])
            for metric in METRICS
        }

    assert len(data) == 30

    for fname in data:
        assert set(data[fname]) == set(METHODS)

    return data


def summarize(acceleration, data):
    print()
    print("=" * 90)
    print(f"FASTMRI-256 TUNING30 R={acceleration}")
    print("=" * 90)

    print(
        f"{'METHOD':<10}"
        f"{'NMSE':>22}"
        f"{'PSNR':>22}"
        f"{'SSIM':>22}"
    )

    print("-" * 76)

    summary_rows = []

    for method in METHODS:
        vals = {
            metric: np.asarray(
                [
                    data[fname][method][metric]
                    for fname in sorted(data)
                ],
                dtype=np.float64,
            )
            for metric in METRICS
        }

        means = {
            metric: vals[metric].mean()
            for metric in METRICS
        }

        stds = {
            metric: vals[metric].std()
            for metric in METRICS
        }

        print(
            f"{method.upper():<10}"
            f"{means['nmse']:>10.6f} ± {stds['nmse']:<9.6f}"
            f"{means['psnr']:>10.3f} ± {stds['psnr']:<9.3f}"
            f"{means['ssim']:>10.4f} ± {stds['ssim']:<9.4f}"
        )

        summary_rows.append(
            {
                "acceleration": acceleration,
                "method": method,
                "nmse_mean": means["nmse"],
                "nmse_std": stds["nmse"],
                "psnr_mean": means["psnr"],
                "psnr_std": stds["psnr"],
                "ssim_mean": means["ssim"],
                "ssim_std": stds["ssim"],
            }
        )

    return summary_rows


def paired_caps(acceleration, data):
    print()
    print("=" * 90)
    print(f"CAPS PAIRED COMPARISONS — R={acceleration}")
    print("=" * 90)

    paired_rows = []

    fnames = sorted(data)

    for baseline in [
        "zf",
        "cg",
        "lacs",
        "nerp",
    ]:

        print()
        print(
            f"CAPS vs {baseline.upper()}"
        )

        caps_nmse = np.asarray(
            [
                data[f]["caps"]["nmse"]
                for f in fnames
            ]
        )

        base_nmse = np.asarray(
            [
                data[f][baseline]["nmse"]
                for f in fnames
            ]
        )

        caps_psnr = np.asarray(
            [
                data[f]["caps"]["psnr"]
                for f in fnames
            ]
        )

        base_psnr = np.asarray(
            [
                data[f][baseline]["psnr"]
                for f in fnames
            ]
        )

        caps_ssim = np.asarray(
            [
                data[f]["caps"]["ssim"]
                for f in fnames
            ]
        )

        base_ssim = np.asarray(
            [
                data[f][baseline]["ssim"]
                for f in fnames
            ]
        )

        d_nmse = (
            caps_nmse
            - base_nmse
        )

        d_psnr = (
            caps_psnr
            - base_psnr
        )

        d_ssim = (
            caps_ssim
            - base_ssim
        )

        nmse_wins = int(
            np.sum(
                caps_nmse
                < base_nmse
            )
        )

        psnr_wins = int(
            np.sum(
                caps_psnr
                > base_psnr
            )
        )

        ssim_wins = int(
            np.sum(
                caps_ssim
                > base_ssim
            )
        )

        relative_nmse_change = (
            (
                caps_nmse.mean()
                - base_nmse.mean()
            )
            / base_nmse.mean()
            * 100.0
        )

        print(
            "ΔNMSE: "
            f"{d_nmse.mean():+.6f} "
            f"± {d_nmse.std():.6f} "
            f"| CAPS better {nmse_wins}/30"
        )

        print(
            "ΔPSNR: "
            f"{d_psnr.mean():+.3f} "
            f"± {d_psnr.std():.3f} "
            f"| CAPS better {psnr_wins}/30"
        )

        print(
            "ΔSSIM: "
            f"{d_ssim.mean():+.4f} "
            f"± {d_ssim.std():.4f} "
            f"| CAPS better {ssim_wins}/30"
        )

        print(
            "Mean NMSE change: "
            f"{relative_nmse_change:+.2f}%"
        )

        paired_rows.append(
            {
                "acceleration": acceleration,
                "comparison":
                    f"caps_vs_{baseline}",

                "delta_nmse_mean":
                    d_nmse.mean(),

                "delta_nmse_std":
                    d_nmse.std(),

                "nmse_wins":
                    nmse_wins,

                "delta_psnr_mean":
                    d_psnr.mean(),

                "delta_psnr_std":
                    d_psnr.std(),

                "psnr_wins":
                    psnr_wins,

                "delta_ssim_mean":
                    d_ssim.mean(),

                "delta_ssim_std":
                    d_ssim.std(),

                "ssim_wins":
                    ssim_wins,

                "relative_nmse_change_percent":
                    relative_nmse_change,
            }
        )

    return paired_rows


def write_csv(path, rows):
    assert rows

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    r4 = load(4)
    r8 = load(8)

    #
    # R4/R8 must use the exact same
    # 30-volume tuning cohort.
    #
    assert set(r4) == set(r8)

    print(
        "R4/R8 cohort match:",
        len(r4),
        "volumes"
    )

    summary_rows = []
    paired_rows = []

    for acceleration, data in [
        (4, r4),
        (8, r8),
    ]:
        summary_rows.extend(
            summarize(
                acceleration,
                data,
            )
        )

        paired_rows.extend(
            paired_caps(
                acceleration,
                data,
            )
        )

    summary_path = (
        OUTPUT_DIR
        / "fastmri256_tuning30_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "fastmri256_tuning30_caps_paired.csv"
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    write_csv(
        paired_path,
        paired_rows,
    )

    print()
    print("=" * 90)
    print("ANALYSIS PASSED")
    print("=" * 90)

    print(
        "summary:",
        summary_path,
    )

    print(
        "paired:",
        paired_path,
    )


if __name__ == "__main__":
    main()
