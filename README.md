# MRI Reconstruction Benchmark

A unified benchmark for MRI reconstruction methods on fastMRI brain multicoil data, with additional validation of longitudinal/prior-based reconstruction methods.

The repository contains implementations and evaluation pipelines for:
* Zero-filled reconstruction (ZF)
* CG-SENSE
* LACS-MRI
* NeRP
* CAPS / LAPS
* Shamaei et al. E2E-VarNet adaptation

The main benchmark uses a frozen 30-volume AXT2 cohort at acceleration factors $R=4$ and $R=8$.

---

## Repository Structure

```text
code/
  data/                 fastMRI loading and fastMRI-256 adaptation
  operators/            SENSE, ESPIRiT, sensitivity caching
  methods/
    lacs/
    nerp/
    prior_transformer/  Shamaei E2E-VarNet fastMRI adapter
  scripts/              preparation, training, evaluation, analysis
configs/                 benchmark configurations
slurm/                   cluster execution scripts
tests/                   implementation and scientific validation tests
reproducibility/
  cohorts/               frozen benchmark/training manifests
  results/               final benchmark CSV results
third_party/             pinned external repositories and compatibility patch
docs/                    benchmark protocol
```

---

## Environment Setup

Create and activate the Conda environment recorded in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate mri-reconstruction-benchmark
```

---

## Third-Party Repositories

Set up the pinned external repositories:

```bash
./third_party/setup_third_party.sh
```

Pinned commits used by the benchmark:
* **LAPS:** `ca1b5cc8d0d24b164a848c6fbd06b3fc5ec7d99b`
* **Shamaei et al.:** `8ef48bc8a9956744c28bce068fb23aff0c268f2a`

A compatibility patch for the vendored diffusers version used by LAPS is located at:
`third_party/patches/laps_diffusers_compat.patch`

> **Note:** The NeRP implementation in `code/methods/nerp/` is adapted from the official NeRP repository and benchmarked in the no-prior setting.

---

## Data

The benchmark uses fastMRI brain multicoil data with AXT2 acquisition. Expected dataset structure in the cluster setup:

```text
extracted/
  multicoil_train/
  multicoil_val/
```

*Raw MRI data and model checkpoints are intentionally excluded from Git.*

---

## Benchmark Settings

Two benchmark spaces are evaluated:

### 1. Native fastMRI Benchmark
Native fastMRI geometry is used for:
* ZF
* CG-SENSE
* LACS-MRI
* NeRP

### 2. fastMRI-256 Adaptation Benchmark
CAPS/LAPS requires $256 \times 256$ images. A self-consistent fastMRI-256 benchmark is constructed as follows:
1. Fully sampled multicoil k-space is transformed to coil-image space.
2. Complex coil images are center-cropped to $256 \times 256$.
3. Cropped coil images are transformed back to k-space.

All six evaluated methods operate on this same adapted measurement problem:
* ZF
* CG
* LACS
* NeRP
* CAPS
* Shamaei E2E-VarNet *(reported as `Shamaei E2E-VarNet (non-enhanced, fastMRI-256 adaptation)`)*

> **Note:** Because fastMRI lacks longitudinal prior examinations, the Shamaei baseline evaluates its single-visit E2E-VarNet reconstruction branch rather than the complete longitudinal method.

---

## Frozen fastMRI-256 Results

Results are reported as $\text{mean} \pm \text{population standard deviation}$ across 30 volumes. Exact machine-readable results are available under `reproducibility/results/`.

| Method | R=4 NMSE ↓ | R=4 PSNR ↑ | R=4 SSIM ↑ | R=8 NMSE ↓ | R=8 PSNR ↑ | R=8 SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ZF** | 0.0504 ± 0.0109 | 26.312 ± 1.366 | 0.7487 ± 0.0271 | 0.1093 ± 0.0196 | 22.932 ± 1.278 | 0.6178 ± 0.0312 |
| **CG** | 0.0365 ± 0.0110 | 27.802 ± 1.559 | 0.7516 ± 0.0463 | 0.0882 ± 0.0196 | 23.901 ± 1.447 | 0.6460 ± 0.0378 |
| **LACS** | 0.0320 ± 0.0095 | 28.377 ± 1.545 | 0.7417 ± 0.0424 | 0.0847 ± 0.0191 | 24.078 ± 1.484 | 0.6106 ± 0.0375 |
| **NeRP** | 0.0387 ± 0.0116 | 27.553 ± 1.298 | 0.7019 ± 0.0390 | 0.1106 ± 0.0290 | 22.960 ± 1.689 | 0.5543 ± 0.0427 |
| **CAPS** | 0.0251 ± 0.0198 | 30.036 ± 2.464 | 0.8163 ± 0.0437 | 0.0893 ± 0.0367 | 24.080 ± 2.183 | 0.6925 ± 0.0505 |
| **Shamaei E2E-VarNet** | 1.1473 ± 1.4608 | 14.982 ± 4.375 | 0.4806 ± 0.1971 | 5.8117 ± 10.4473 | 11.673 ± 7.563 | 0.4462 ± 0.2008 |

---

## Running Evaluations

Run commands from the repository root directory. Cluster execution scripts are provided under `slurm/`.

* **Native benchmark:**
  ```bash
  export PYTHONPATH=code
  python code/scripts/eval_fastmri_prior_free.py --help
  ```

* **fastMRI-256 / CAPS benchmark:**
  ```bash
  export PYTHONPATH="code:$PWD/external/laps_original/src"
  python code/scripts/eval_fastmri256_prior_free.py --help
  ```

* **Shamaei evaluation:**
  ```bash
  export PYTHONPATH=code
  python code/scripts/eval_shamaei_fastmri256_tuning30.py --help
  ```

---

## Reproducibility

The repository includes frozen Shamaei training/benchmark manifests and final result tables. **Do not regenerate benchmark cohorts** when reproducing reported results. Refer to `docs/benchmark_protocol.md` for detailed methodological guidelines.