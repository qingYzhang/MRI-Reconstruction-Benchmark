# Benchmark Protocol

## Dataset

The experiments use the fastMRI brain multicoil dataset and restrict the primary experiments to the AXT2 acquisition.

The frozen tuning benchmark contains 30 training volumes selected with seed `20260904`. Every method within a benchmark uses the same volumes and the same deterministic undersampling realization.

Metrics are computed per volume and summarized across the 30 volumes.

---

## Acceleration

Experiments are performed at:
* $R = 4$
* $R = 8$

The deterministic Cartesian sampling policy is shared across compared methods.

---

## Reconstruction Operators

SENSE-based methods use ESPIRiT sensitivity maps estimated from the acquired k-space.

The common CG-SENSE baseline uses:
* 50 CG iterations
* $\lambda = 0.01$

---

## Native fastMRI Benchmark

The native benchmark preserves the original fastMRI spatial geometry.

**Methods:**
* Zero-filled reconstruction (ZF)
* CG-SENSE
* LACS-MRI without prior
* NeRP without prior

> **Note:** This benchmark must not be mixed numerically with the $256 \times 256$ adaptation benchmark.

---

## fastMRI-256 Adaptation

The official LAPS/CAPS reconstruction pipeline requires $256 \times 256$ spatial dimensions.

To obtain a self-consistent $256 \times 256$ multicoil problem:
1. Start from fully sampled native fastMRI multicoil k-space.
2. Apply centered orthonormal inverse FFT to obtain complex coil images.
3. Center-crop each complex coil image to $256 \times 256$.
4. Apply centered orthonormal FFT to obtain the adapted fully sampled $256 \times 256$ k-space.
5. Apply the deterministic $R=4$ or $R=8$ Cartesian mask.

No zero-padding is used for native acquisitions smaller than the required spatial size.

**Methods compared on this adapted problem:**
* ZF
* CG-SENSE
* LACS-MRI
* NeRP
* CAPS
* Shamaei E2E-VarNet adaptation

---

## LACS-MRI

The final implementation uses the source-faithful fixed-mask pathway, including the Wavelab-compatible periodized orthogonal wavelet and official-style FISTA reconstruction.

The fastMRI experiments use the no-prior condition.

---

## NeRP

The no-prior fastMRI configuration uses:
* 8-layer MLP
* Hidden width: 512
* Embedding size: 256
* Gaussian encoding standard deviation: 3
* Optimizer: Adam
* Learning rate: 1e-5
* 1000 reconstruction iterations

The reported results correspond to the frozen 1000-step configuration.

---

## CAPS / LAPS

CAPS is the prior-free LAPS-family comparator.

The official LAPS reconstruction logic and $256 \times 256$ image requirement are retained. The LAPS source repository is pinned to the commit recorded under `third_party/`.

**Configuration:**
* `start_with_prior = False`
* `start_with_cg = True`
* `prior_start_timestep = 200`

For $R=4$, the final output data consistency uses the official low-acceleration settings. For $R=8$, the corresponding official high-acceleration settings are used.

The final reconstruction is restored to raw adapted fastMRI intensity units using the measurement-derived normalization scale. No target-derived scaling is used.

---

## Shamaei E2E-VarNet Adaptation

The Shamaei comparison evaluates:
`Shamaei E2E-VarNet (non-enhanced, fastMRI-256 adaptation)`

The complete method in the original work includes longitudinal registration and transformer-based enhancement. fastMRI does not contain repeated longitudinal examinations, so those components are not evaluated here.

**Reconstruction Core Architecture (EndToEndVarNet):**
* 12 cascades
* 18 regularizer filters
* 4 pooling levels
* Dropout: 0

The fastMRI adaptation uses external ESPIRiT sensitivity maps shared with the benchmark rather than the original learned sensitivity-map network.

**Training Settings:**
* Optimizer: Adam
* Learning rate: 1e-3
* Loss: Source-style independently max-normalized SSIM loss
* Maximum 200 epochs
* Effective batch size: 64
* Early stopping based on calibrated raw-image SSIM on the validation set

Because source-style max-normalized SSIM training weakly constrains global complex amplitude, validation and inference use an acquired-k-space-only least-squares complex calibration:

$$\alpha = \frac{\langle k_{\text{pred}}, y \rangle}{\langle k_{\text{pred}}, k_{\text{pred}} \rangle}$$

where the inner products are evaluated only on acquired k-space entries.

The final magnitude image is scaled using $|\alpha|$ and the measurement-derived normalization factor. Ground-truth images are never used to estimate the calibration.

---

## Metrics

The reported metrics are:
* NMSE
* PSNR
* SSIM

For the fastMRI-256 benchmark, PSNR uses the maximum value of the adapted target volume.

Summary values are reported as:

$$\text{mean} \pm \text{population standard deviation}$$

with:
* $\text{ddof} = 0$
* $n = 30\text{ volumes}$

---

## Frozen Results

Machine-readable final results are stored under:
`reproducibility/results/`

The authoritative six-method summary is:
`fastmri256_tuning30_six_method_summary.csv`

The exact Shamaei benchmark/training cohort definitions are stored under:
`reproducibility/cohorts/`
