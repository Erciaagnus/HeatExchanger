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
- `data/porouos_from_design.csv`
- `data/porous_LHS_100_projected.csv`
- `data/constrained_LHS_100k.csv`

Notes
- if you downloaded already includes a representative subset(porous_from_design.csv) you can skip section 1.3
- if you only downloaded constrained_LHS_100k.csv, go to section 1.2, and execute mapping code

#### 1.2 Option B — Generate the dataset locally (Sampling)

##### a. Generate Points
Run the sampling script (LHS sampling + porous parameter calculation):

- Script: `HeatExchanger/scripts/sampling/generate_point.py`
- Output directory: `HeatExchanger/data/` (recommended convention)

```bash
cd /path/to/HeatExchanger

python3 scripts/sampling/generate_point.py

# Check generated file(s)
ls -lh data | tail -n 20
```

Expected output example:
- `data/constrained_LHS_100k.csv`

##### b. Mapping
- Script: `HeatExchanger/scripts/sampling/generate_point.py`
- Output directory: `HeatExchanger/data/` (recommended convention)

```bash
cd /path/to/HeatExchanger

python3 scripts/sampling/mapping_to_porous.py

# Check generated file(s)
ls -lh data | tail -n 20
```
Expected output example:
- `data/porous_from_design.csv`

#### 1.3 Select representative points (similarity / downsampling)

Select representative points using LHS sampling

- Script: `HeatExchanger/scripts/lhs_sampling.py`
- Output directory: `HeatExchanger/data/` (recommended)

```bash
cd /path/to/HeatExchanger

python3 scripts/sampling/lhs_sampling.py

# Check generated file(s)
ls -lh data | tail -n 20
```

Expected output example:
- `data/porous_LHS_100_porjected.csv`

---

### 2) Visualizing Sampling Results

If you want check the sampling results, visualize the results following this instructions

- Script: `HeatExchanger/scripts/visualization/plot_porous_sampling.py`
- Input directory: `HeatExchanger/data/`
- Output directory: `HeatExchanger/data/`

```bash
cd /path/to/HeatExchanger

# Example:
#   input  : data/LHS_design_samples.csv
#   output : data/porous_from_design.csv
python3 scripts/visualization/plot_porous_sampling.py

# Verify output
ls -lh figure/porous_plots/dist_3d_log.png
```
Expected output example:
- `figure/porous_plots/dist_3d_log.png`
- `figure/porous_plots/hist_inertial_log.png`
- `figure/porous_plots/hist_porosity.png`
- `figure/porous_plots/hist_viscous_log.png`

### 3) Train GP Surrogate Models (Q'' and ΔP) and Find Optimization Value

This stage trains Gaussian Process models for:
- `Q''` (heat flux, W/m²)
- `ΔP` (pressure drop, Pa)

> You should download the "data file"(total_2D_Data.xlsx) or get your own data using CFD

#### 3.1 Train surrogate and optimize

##### a. Training and Results
If you want to see the result directly using this code
```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/surrogate_model.py
```

##### b. Train Model
If you want store the trained model and use at others, using this code
```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/train_model.py
```
Expected output example:
- `model/gp_surrogate_bundle.joblib`

#### 3.2 Validate the surrogate Model

```bash
cd /path/to/HeatExchanger

python3 scripts/surrogate/validate_model.py
```

---

### 4) Validate the Results using 3D CFD Value

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

#### 4.2 Validate "Friction Factor" Correlation

```bash
cd /path/to/HeatExchanger

python3 scripts/verfication/validate_correlation.py
```

Expected outputs:
- `figure/correlation_bar.png` 


## II. Dataset

### A) Prebuilt dataset (download)

Place into:
- `HeatExchanger/data/`

| Dataset | Purpose | Target path (repo) | File name | Download link | Notes |
|---|---|---|---|---|---|
| constrained_LHS_100k.csv | Prebuilt discretized desing parameter space (1/K, C2, etc.) | `data/` | `constrained_LHS_100k.csv` | [Download](<PUT_LINK_HERE>) | Put the file exactly at `HeatExchanger/data/constrained_LHS_100k.csv` |
| porous_from_design | Prebuilt mapping data from discretized design parameter space into porous parameter space | `data/` | `porous_form_design.csv` | [Download](<PUT_LINK_HERE>) | Put the file exactly at `HeatExchanger/data/constrained_LHS_100k.csv` |
| porous_LHS_100_projected | LHS Sampling data | `data/` | `porous_LHS_100_projected.xlsx` | [Download](<PUT_LINK_HERE>) | Put the file exactly at `HeatExchanger/data/constrained_LHS_100k.csv` |
| total_2D_Data | CFD/2D training dataset for surrogate (Q'', ΔP) | `data/` | `total_2D_Data.xlsx` | [Download](<PUT_LINK_HERE>) | Used by GP surrogate training + GA optimization |
| correlation_validation | CFD/Correlation Pressure drop Results data for comparing | `data/` | `correlation.xlsx` | [Download](<PUT_LINK_HERE>) | Validate the correlation |

Examples:
- `data/constrained_LHS_100_projected.xlsx`
- `data/total_2D_Data.xlsx`

---
