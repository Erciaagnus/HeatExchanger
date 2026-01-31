#!/usr/bin/env python3
"""
Plot 3D scatter + per-variable histograms for porous-parameter sample CSV.

Configuration
-------------
Modify INPUT_CSV, OUTPUT_DIR, USE_LOG, HIST_BINS below as needed.

Usage
-----
python plot_porous_3d_and_hist.py
"""

import os
from typing import Dict
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== Configuration ==========
INPUT_CSV = "../../data/porous_LHS_100_projected.csv"
OUTPUT_DIR = "../../figure/porous_plots"
USE_LOG = True  # Set to False for linear scale
HIST_BINS = 15
# ===================================


CANON = ["Porosity", "Viscous_Resistance_1_m2", "Inertial_Resistance_1_m"]
ALIASES = {
    "Porosity": ["Porosity", "porosity", "epsilon", "eps"],
    "Viscous_Resistance_1_m2": ["Viscous_Resistance_1_m2", "inv_K_1_m2", "viscous_resistance", "viscous"],
    "Inertial_Resistance_1_m": ["Inertial_Resistance_1_m", "C2_1_m", "inertial_resistance", "inertial", "C2"],
}


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Find the correct column names from aliases."""
    cols = set(df.columns)
    mapping: Dict[str, str] = {}
    for canon, alias_list in ALIASES.items():
        found = None
        for a in alias_list:
            if a in cols:
                found = a
                break
        if found is None:
            raise ValueError(
                f"Missing required column for {canon}. Accepted: {alias_list}. Found: {list(df.columns)}"
            )
        mapping[canon] = found
    return mapping


def safe_log10(x: np.ndarray) -> np.ndarray:
    """Apply log10, handling non-positive values."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        min_pos = np.min(x[x > 0]) if np.any(x > 0) else 1.0
        x = np.where(x <= 0, min_pos * 1e-6, x)
    return np.log10(x)


def save_hist(data: np.ndarray, xlabel: str, out_path: str, bins: int) -> None:
    """Save a histogram plot."""
    fig = plt.figure(figsize=(7, 4))
    plt.hist(data, bins=bins, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"Histogram: {xlabel}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main plotting function."""
    # Build absolute paths based on script location
    input_path = os.path.join(SCRIPT_DIR, INPUT_CSV)
    output_dir = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    
    print(f"Reading: {input_path}")
    
    # Read CSV with encoding support
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='cp949')
        except Exception:
            df = pd.read_csv(input_path, encoding='latin1')
    
    print(f"Loaded {len(df)} rows")
    
    cols = resolve_columns(df)
    os.makedirs(output_dir, exist_ok=True)

    P = df[cols["Porosity"]].astype(float).to_numpy()
    V = df[cols["Viscous_Resistance_1_m2"]].astype(float).to_numpy()
    I = df[cols["Inertial_Resistance_1_m"]].astype(float).to_numpy()

    if not USE_LOG:
        Vp, Ip = V, I
        vlab = "Viscous Resistance (1/m²)"
        ilab = "Inertial Resistance (1/m)"
        suffix = "linear"
    else:
        Vp, Ip = safe_log10(V), safe_log10(I)
        vlab = "log10(Viscous Resistance 1/m²)"
        ilab = "log10(Inertial Resistance 1/m)"
        suffix = "log"

    # 3D scatter
    print("Creating 3D scatter plot...")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P, Vp, Ip, s=55, alpha=0.9, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("Porosity")
    ax.set_ylabel(vlab)
    ax.set_zlabel(ilab)
    ax.set_title(f"3D distribution of selected porous samples ({suffix})")
    fig.tight_layout()
    p3d = os.path.join(output_dir, f"dist_3d_{suffix}.png")
    fig.savefig(p3d, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Histograms
    print("Creating histograms...")
    save_hist(P, "Porosity", os.path.join(output_dir, "hist_porosity.png"), bins=HIST_BINS)
    save_hist(Vp, vlab, os.path.join(output_dir, f"hist_viscous_{suffix}.png"), bins=HIST_BINS)
    save_hist(Ip, ilab, os.path.join(output_dir, f"hist_inertial_{suffix}.png"), bins=HIST_BINS)

    print("\n=== Saved plots ===")
    print("Input:", input_path)
    print("Out dir:", output_dir)
    for fn in [f"dist_3d_{suffix}.png", "hist_porosity.png", f"hist_viscous_{suffix}.png", f"hist_inertial_{suffix}.png"]:
        print(" -", fn)


if __name__ == "__main__":
    main()
