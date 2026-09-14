import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fastmri.data.transforms import center_crop


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_ROOT))


from data.fastmri_brain import (
    FastMRIBrainDataset,
    apply_cartesian_mask,
)

from operators.sense import (
    prepare_mask,
    cg_sense,
)

from operators.sensitivity import (
    estimate_sens_espirit,
)

from evaluation.metrics import (
    compute_volume_metrics,
)


LAMBDAS = [
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="extracted/multicoil_train",
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="AXT2",
    )

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 8],
        required=True,
    )

    parser.add_argument(
        "--num_volumes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    parser.add_argument(
        "--cg_iters",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--espirit_crop",
        type=float,
        default=0.0,
        help="ESPIRiT eigenvalue crop threshold",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    return parser.parse_args()


def get_raw_samples(dataset):

    if hasattr(dataset, "raw_samples"):
        return dataset.raw_samples

    if hasattr(dataset, "examples"):
        return dataset.examples

    raise RuntimeError(
        "Cannot find raw_samples/examples "
        "inside SliceDataset."
    )


def get_fname(raw_sample):

    if hasattr(raw_sample, "fname"):
        return Path(raw_sample.fname).name

    if isinstance(raw_sample, tuple):
        return Path(raw_sample[0]).name

    raise RuntimeError(
        f"Cannot parse raw sample: {raw_sample}"
    )


def main():

    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("CG-SENSE LAMBDA TUNING")
    print("=" * 80)

    print("device:", device)
    print("acceleration:", args.acceleration)
    print("num volumes:", args.num_volumes)
    print("seed:", args.seed)
    print("CG iterations:", args.cg_iters)
    print("CG iterations:", args.cg_iters)
    print("ESPIRiT crop:", args.espirit_crop)
    print("lambdas:", LAMBDAS)

    dataset = FastMRIBrainDataset(
        root=args.data_root,
        acquisition=args.acquisition,
        cache_dir="code/.cache",
    )

    raw_samples = get_raw_samples(
        dataset
    )

    volume_indices = defaultdict(list)

    for idx, raw_sample in enumerate(
        raw_samples
    ):
        fname = get_fname(
            raw_sample
        )

        volume_indices[fname].append(
            idx
        )

    all_fnames = sorted(
        volume_indices.keys()
    )

    print(
        "available training volumes:",
        len(all_fnames)
    )

    rng = random.Random(
        args.seed
    )

    selected_fnames = rng.sample(
        all_fnames,
        min(
            args.num_volumes,
            len(all_fnames),
        ),
    )

    selected_fnames = sorted(
        selected_fnames
    )

    print("\nSelected tuning volumes:")

    for fname in selected_fnames:
        print(" ", fname)

    #
    # results[lambda] =
    #     list of volume-level metric dicts
    #
    results = {
        lam: []
        for lam in LAMBDAS
    }

    detailed_rows = []

    for volume_number, fname in enumerate(
        tqdm(
            selected_fnames,
            desc=f"Tune CG R{args.acceleration}",
        ),
        start=1,
    ):

        indices = volume_indices[
            fname
        ]

        predictions = {
            lam: []
            for lam in LAMBDAS
        }

        targets = []

        max_value = None

        for idx in indices:

            sample = dataset[idx]

            full_kspace = (
                sample["kspace"]
                .to(device)
            )

            (
                masked_kspace,
                raw_mask,
                num_low,
            ) = apply_cartesian_mask(
                sample,
                args.acceleration,
            )

            masked_kspace = (
                masked_kspace
                .to(device)
            )

            mask = prepare_mask(
                raw_mask,
                width=full_kspace.shape[-1],
                device=device,
            )

            #
            # IMPORTANT:
            # ESPIRiT uses ONLY ACS internally.
            #
            sens_maps = (
                estimate_sens_espirit(
                    full_kspace,
                    num_low_frequencies=num_low,
                    crop=args.espirit_crop,
                )
            )

            target = (
                sample["target"]
                .cpu()
                .numpy()
            )

            targets.append(
                target
            )

            max_value = sample[
                "max_value"
            ]

            for lam in LAMBDAS:

                x = cg_sense(
                    measured_kspace=masked_kspace,
                    sens_maps=sens_maps,
                    mask=mask,
                    num_iters=args.cg_iters,
                    lambda_reg=lam,
                )

                reconstruction = (
                    torch.abs(x)
                )

                reconstruction = center_crop(
                    reconstruction,
                    sample["target"].shape[-2:],
                )

                predictions[lam].append(
                    reconstruction
                    .cpu()
                    .numpy()
                )

        target_volume = np.stack(
            targets,
            axis=0,
        )

        for lam in LAMBDAS:

            prediction_volume = np.stack(
                predictions[lam],
                axis=0,
            )

            metrics = (
                compute_volume_metrics(
                    target_volume,
                    prediction_volume,
                    max_value=max_value,
                )
            )

            results[lam].append(
                metrics
            )

            detailed_rows.append(
                {
                    "fname": fname,
                    "acceleration":
                        args.acceleration,
                    "espirit_crop":
                        args.espirit_crop,
                    "lambda": lam,
                    "nmse":
                        metrics["nmse"],
                    "psnr":
                        metrics["psnr"],
                    "ssim":
                        metrics["ssim"],
                }
            )

        print(
            f"\n[{volume_number}] "
            f"{fname}"
        )

        for lam in LAMBDAS:

            metric = results[lam][-1]

            print(
                f"  λ={lam:<8g} "
                f"NMSE={metric['nmse']:.6f} "
                f"PSNR={metric['psnr']:.3f} "
                f"SSIM={metric['ssim']:.4f}"
            )

    #
    # Save detailed results
    #
    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fname",
                "acceleration",
                "espirit_crop",
                "lambda",
                "nmse",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(
            detailed_rows
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = []

    for lam in LAMBDAS:

        nmse_values = np.asarray(
            [
                x["nmse"]
                for x in results[lam]
            ]
        )

        psnr_values = np.asarray(
            [
                x["psnr"]
                for x in results[lam]
            ]
        )

        ssim_values = np.asarray(
            [
                x["ssim"]
                for x in results[lam]
            ]
        )

        row = {
            "lambda": lam,
            "nmse_mean":
                nmse_values.mean(),
            "nmse_std":
                nmse_values.std(),
            "psnr_mean":
                psnr_values.mean(),
            "psnr_std":
                psnr_values.std(),
            "ssim_mean":
                ssim_values.mean(),
            "ssim_std":
                ssim_values.std(),
        }

        summary.append(
            row
        )

        print(
            f"λ={lam:<8g} | "
            f"NMSE "
            f"{row['nmse_mean']:.6f} "
            f"+/- "
            f"{row['nmse_std']:.6f} | "
            f"PSNR "
            f"{row['psnr_mean']:.3f} "
            f"+/- "
            f"{row['psnr_std']:.3f} | "
            f"SSIM "
            f"{row['ssim_mean']:.4f} "
            f"+/- "
            f"{row['ssim_std']:.4f}"
        )

    #
    # Select lambda ONLY by mean NMSE.
    #
    best = min(
        summary,
        key=lambda x:
            x["nmse_mean"],
    )

    print("\n" + "=" * 80)
    print("SELECTED LAMBDA")
    print("=" * 80)

    print(
        "acceleration:",
        args.acceleration
    )

    print(
        "best lambda:",
        best["lambda"]
    )

    print(
        "mean NMSE:",
        best["nmse_mean"]
    )

    print(
        "\nSaved detailed results:",
        output
    )


if __name__ == "__main__":
    main()
