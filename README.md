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

### 2.3 (Optional) Conda environment
```bash
conda create -n wind-hx python=3.10 -y
conda activate wind-hx
pip install -r requirements.txt

