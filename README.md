# A Study of Designing Heat Exchanger
This repository supports the workflow:

1) Generate design samples (LHS)  
2) Convert design samples to porous parameters (Darcy–Forchheimer: 1/K, C2, etc.)  
3) Train GP surrogate models (Q'' and ΔP) and optionally run GA optimization / prediction
4) Validate the Value with CFD Data

## I. Step-by-Step

### 0) Setup (recommended)

From the repository root (`HeatExchanger/`):

```bash
# Go to repo root
cd /path/to/HeatExchanger

# (Recommended) create & activate a venv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -U pip
pip install numpy pandas scipy scikit-learn pygad
```


### 0.1 (Optional) Conda environment

```bash
cd /path/to/HeatExchanger

conda create -n wind-hx python=3.10 -y
conda activate wind-hx

# install dependencies
pip install -U pip
pip install -r requirements.txt
```

> Pick **one** environment manager: venv or conda (you don't need both).


---

### 1) Create / Prepare the Sampling Dataset

You need a dataset with (at minimum) these design variables:

- `S1_mm`
- `fin_height_fh_mm`
- `fin_spacing_fs_mm`

There are two options.

#### 1.1 Option A — Download a prepared dataset

Download `porousdata.xlsx` (or an equivalent prepared dataset) and place it into:

```bash
cd /path/to/HeatExchanger
mkdir -p data
# put your downloaded file into HeatExchanger/data/
ls -lh data
```

Expected examples:
- `data/porousdata.xlsx`
- `data/total_2D_Data.xlsx`

#### 1.2 Option B — Generate the dataset locally (Sampling)

Run the sampling script (LHS sampling + porous parameter calculation):

- Script: `HeatExchanger/scripts/surrogate/sampling.py`
- Output directory: `HeatExchanger/data/` (recommended convention)

```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/sampling.py

# Check generated file(s)
ls -lh data | tail -n 20
```

Expected output example:
- `data/LHS_Porous_Params_Result.csv`

#### 1.3 (Optional) Select representative points (similarity / downsampling)

If you select representative points after sampling:

- Script: `HeatExchanger/scripts/finding_similar.py`
- Output directory: `HeatExchanger/data/` (recommended)

```bash
cd /path/to/HeatExchanger

python3 scripts/finding_similar.py

# Check generated file(s)
ls -lh data | tail -n 20
```

---

### 2) Convert Design CSV → Porous CSV (porous_converter.py)

If you already have a **design-only CSV** (S1/FH/FS) and want to convert it into a porous-parameter CSV:

- Script: `HeatExchanger/scripts/porous_calc/porous_converter.py`
- Input directory: `HeatExchanger/data/`
- Output directory: `HeatExchanger/data/`

```bash
cd /path/to/HeatExchanger

# Example:
#   input  : data/LHS_design_samples.csv
#   output : data/porous_from_design.csv
python3 scripts/porous_calc/porous_converter.py   --in data/LHS_design_samples.csv   --out data/porous_from_design.csv   --T 14.8   --v 2.019   --Dc 24.0   --delta_f 0.5   --pitch_ratio 1.0   --N 4   --v_min 0.5   --v_max 3.5   --n_points 50   --check_constraint

# Verify output
ls -lh data/porous_from_design.csv
```

**Column name requirements (input CSV)**  
The input CSV must contain these columns (case-insensitive matching is supported):

- S1: `S1_mm` / `S1` / `s1_mm` / `s1`
- fin height: `fin_height_fh_mm` / `fh_mm` / `fin_height` / `fh` / `hf_mm` / `hf`
- fin spacing: `fin_spacing_fs_mm` / `fs_mm` / `fin_spacing` / `Fs_mm` / `Fs` / `fs`

---

### 3) Train GP Surrogate Models (Q'' and ΔP)

This stage trains Gaussian Process models for:
- `Q''` (heat flux, W/m²)
- `ΔP` (pressure drop, Pa)

> Your surrogate training script(s) may differ by filename. Use the commands below as a template.

#### 3.1 Train surrogate

```bash
cd /path/to/HeatExchanger

# Example: run a surrogate training script
python3 scripts/surrogate/train_surrogate.py --data data/total_2D_Data.xlsx
```

If your script is a plain Python file without CLI args, just run it directly:
```bash
cd /path/to/HeatExchanger
python3 scripts/surrogate/train_surrogate.py
```

#### 3.2 Run GA optimization (optional)

```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/train_surrogate.py --mode ga --data data/total_2D_Data.xlsx
```

#### 3.3 Predict using a user-defined design (optional)

```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/train_surrogate.py --mode predict   --data data/total_2D_Data.xlsx   --s1 181.0394 --fh 28.9923 --fs 2.6038
```

---

### 4) Validate the value with CFD data

Compare:
- Surrogate-predicted `Q''` and `ΔP`
vs.
- CFD-evaluated `Q''` and `ΔP`

#### 4.1 Prepare CFD validation dataset

Put your CFD results into `HeatExchanger/data/` as a CSV or Excel file.

**Recommended columns**
- Inputs:
  - `S1_mm`, `fin_height_fh_mm`, `fin_spacing_fs_mm`
- CFD outputs (suggested names):
  - `Q_CFD` (or `Qpp_CFD`)
  - `dP_CFD` (or `DeltaP_CFD`)

```bash
cd /path/to/HeatExchanger
ls -lh data | grep -i cfd
```

#### 4.2 Run validation script

```bash
cd /path/to/HeatExchanger

python3 scripts/validation/validate_with_cfd.py   --train data/total_2D_Data.xlsx   --cfd   data/CFD_validation.csv   --out   data/validation_report.csv
```

Expected outputs:
- `data/validation_report.csv` (per-sample error summary)
- (Optional) parity plots / error histograms (if implemented)

#### 4.3 What to report

Common validation metrics:
- R² (CFD vs surrogate)
- MAE / RMSE for `Q''` and `ΔP`
- Parity plot (y = x) and error distribution
- Worst-case error samples (top-k)

---

## II. Dataset

### A) Prebuilt dataset (download)

Place into:
- `HeatExchanger/data/`

Examples:
- `data/porousdata.xlsx`
- `data/total_2D_Data.xlsx`

### B) Locally generated dataset

Generated by:
- `scripts/surrogate/sampling.py`

Output example:
- `data/LHS_Porous_Params_Result.csv`

---
