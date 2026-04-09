# ICIM 3C — 3-Compartment IVIM PINN for 7T Human Brain DWI

This pipeline estimates three-compartment IVIM parameters from 7T brain diffusion data using a Physics-Informed Neural Network. Both IR and non-IR acquisition sequences are supported, with the appropriate signal model applied for each. The approach follows [Voorter et al., MRM 2023](https://doi.org/10.1002/mrm.29754). If you use this, please cite that paper.

---

## What it estimates

Six parameters per voxel, always in this order:

| Parameter | What it is |
|---|---|
| `Dpar` | Parenchymal diffusivity (mm²/s) |
| `Fint` | Interstitial fluid fraction |
| `Dint` | Interstitial diffusivity (mm²/s) |
| `Fmv` | Microvascular fraction |
| `Dmv` | Microvascular pseudo-diffusivity (mm²/s) |
| `S0` | Baseline signal scale |

The 4D output NIfTI stacks them in this exact order, and there's a `_ORDER.txt` file in every output folder so you never have to guess.

---

## Setup

You'll need Python ≥ 3.9 and PyTorch. A GPU is strongly recommended for training — the default ensemble of 20 networks is slow on CPU.

```bash
# 1. Clone this repo
git clone https://github.com/<your-org>/icim-3c-7t.git
cd icim-3c-7t

# 2. Install IVIMNET (required dependency)
git clone https://github.com/oliverchampion/IVIMNET.git
pip install -e IVIMNET/

# 3. Set up a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 4. Install the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# CPU-only alternative:
# pip install torch torchvision

pip install nibabel numpy scipy tqdm joblib matplotlib
```

---

## How it works — two steps

```
Step 0: Preprocessing
Step 1: train_pinn_7t.py    train a model for your specific protocol (once)
Step 2: PINN_7T_style.py    apply that model to your subjects
```

You train once per protocol (IR vs non-IR, and your TE/TR/TI), then reuse the model for all subjects scanned with that protocol.
---

## Step 0 — Preprocessing

Before running the pipeline, your DWI data should be preprocessed. The following steps are recommended:

1. **Denoising** — e.g. MP-PCA using `dwidenoise` (MRtrix3)
2. **Gibbs ringing correction** — e.g. `mrdegibbs` (MRtrix3)
3. **Motion and eddy current correction** — e.g. `eddy` (FSL)
4. **Brain masking** — e.g. `bet` (FSL) to generate the mask required by Step 2

The pipeline expects:
- A **4D DWI NIfTI** with all volumes in acquisition order
- A **3D brain mask** matching the DWI dimensions
- A **`.bval` file** with one b-value per volume, in acquisition order

---
---

## Step 1 — Training

```bash
python train_pinn_7t.py \
  --out    ./models/IR_TE58_TR18000_TI2300 \
  --ir     true \
  --te     58 \
  --tr     18000 \
  --ti     2300 \
  --maxit  500 \
  --lr     3e-5 \
  --ensemble 20
```

Non-IR is the same, just flip the flag and set TI to 0:

```bash
python train_pinn_7t.py \
  --out ./models/nonIR_TE58_TR8000 \
  --ir false --te 58 --tr 8000 --ti 0
```

### All flags

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--out` | ✓ | — | Where to save the model and plots |
| `--ir` | ✓ | — | `true` or `false` |
| `--te` | ✓ | — | Echo time in ms |
| `--tr` | ✓ | — | Repetition time in ms |
| `--ti` | ✓ | — | Inversion time in ms — use `0` for non-IR |
| `--maxit` | | `500` | Training iterations |
| `--lr` | | `3e-5` | Learning rate |
| `--ensemble` | | `20` | Number of networks to average.|
| `--repeats` | | `1` | Independent repeat runs |
| `--jobs` | | `1` | Parallel workers |
| `--bvalues` | | from `hyperparams.py` | Comma-separated b-values matching your `.bval` file exactly |

**If your b-value scheme differs from the default, pass all b-values in acquisition order — one per volume, including all directions. For example, if you acquired 3 directions per shell, each b-value appears 3 times. 

```bash
python train_pinn_7t.py \
  --out ./models/custom \
  --ir true --te 58 --tr 18000 --ti 2300 \
  --bvalues "0,10,10,10,20,20,20,40,40,40,60,60,60,90,90,90,120,120,120,200,200,200,300,300,300,400,400,400,500,500,500,600,600,600,700,700,700,800,800,800,1000,1000,1000,1200,1200,1200"
```

### What gets saved

```
models/IR_TE58_TR18000_TI2300/
├── PINN_7T_trained_IR.pt       ← the model weights you'll use in Step 2
├── results_PINN.npy            ← simulation accuracy
├── stability.npy               ← ensemble stability
├── train_metadata.json         ← everything needed to reproduce this run
└── plots/                      ← accuracy and dependency plots
```

---

## Step 2 — Inference on real data

```bash
python PINN_7T_style.py \
  --input  /path/to/sub-01_dwi.nii.gz \    # your 4D DWI
  --mask   /path/to/sub-01_mask.nii.gz \   # your brain mask
  --bvals  /path/to/sub-01_dwi.bval \      # your b-values file
  --model  /path/to/PINN_7T_trained_IR.pt \ # trained model from Step 1
  --out    /path/to/results/sub-01 \        # where outputs go
  --ir     true \
  --te     58 \
  --tr     18000 \
  --ti     2300
```

> **Important:** the `--ir`, `--te`, `--tr`, `--ti` flags here must match what you used during training. Mismatched timing = wrong signal model.

### All flags

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--input` | ✓ | — | 4D DWI NIfTI |
| `--mask` | ✓ | — | 3D brain mask |
| `--bvals` | ✓ | — | Plain-text b-values, one per volume |
| `--model` | ✓ | — | The `.pt` from Step 1 |
| `--out` | ✓ | — | Output directory |
| `--ir` | ✓ | — | Must match training |
| `--te / --tr / --ti` | ✓ | — | Must match training |
| `--run-nnls` | | `true` | Also run NNLS as a reference — set to `false` to skip |

### What gets saved

```
results/sub-01/
├── PINN_Dpar.nii.gz
├── PINN_Fint.nii.gz
├── PINN_Dint.nii.gz
├── PINN_Fmv.nii.gz
├── PINN_Dmv.nii.gz
├── PINN_S0.nii.gz
├── PINN_params_4d.nii.gz       ← all 6 params stacked
├── PINN_ORDER.txt              ← confirms the volume order
├── NNLS_*.nii.gz               ← same structure (if --run-nnls true)
└── run_metadata.json
```

---

## Relaxation times

The defaults in `hyperparams.py` are set for 7T:

| Tissue | T1 (ms) | T2 (ms) |
|---|---|---|
| Blood | 2600 | 23 |
| Parenchyma | 1200 | 46 |
| ISF | 4300 | 100 |

If your site has different reference values, edit the `rel_times` class in `hyperparams.py`.

---

## GPU / device selection

By default the pipeline uses whatever CUDA device is available. To run on a specific GPU or force CPU:

```bash
HP_DEVICE=cuda:1 python train_pinn_7t.py ...   # second GPU
HP_DEVICE=cpu    python train_pinn_7t.py ...   # CPU only
```

---

## Common issues

**`ModuleNotFoundError: No module named 'IVIMNET'`**
Run `pip install -e IVIMNET/` from the repo root, or add the IVIMNET directory to your `PYTHONPATH`.

**CUDA out of memory during training**
Lower `HP_SIMS` (e.g. `HP_SIMS=5000000`) or reduce `--ensemble`.

**Mask/data shape mismatch**
The mask must be a 3D file matching the first three dimensions of your 4D DWI. Check with `fslinfo`.

**b-values count doesn't match volumes**
Your `.bval` file needs exactly one value per volume, in acquisition order. Double-check with `fslinfo your_dwi.nii.gz`.

---
