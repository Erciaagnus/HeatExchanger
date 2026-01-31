#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def _find_col(df: pd.DataFrame, candidates):
    """Find the first matching column name among candidates (case-insensitive, ignores spaces)."""
    norm = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in norm:
            return norm[key]
    raise KeyError(f"Could not find any of columns: {candidates}. Available: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default="../../data/correlation.xlsx",
                        help="Path to correlation.xlsx")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Sheet name (optional). If omitted, first sheet is used.")
    parser.add_argument("--out", type=str, default="../../figure/correlation_bar.png",
                        help="Output image filename")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path.resolve()}")

    df = pd.read_excel(xlsx_path, sheet_name=args.sheet)

    # sheet_name=None 이면 dict로 들어오므로 첫 시트 선택
    if isinstance(df, dict):
        first_sheet_name = next(iter(df.keys()))
        df = df[first_sheet_name]
    # Robust column lookup (handles minor naming differences)
    col_sample = _find_col(df, ["Sample No", "SampleNo", "Sample"])
    col_cfd    = _find_col(df, ["CFD Value", "CFD", "CFDValue"])
    col_nir    = _find_col(df, ["Nir Correlation", "Nir", "NirCorrelation"])
    col_corr   = _find_col(df, ["Correlation Value", "Correlation", "CorrelationValue"])

    # Keep only needed columns and drop rows with missing sample numbers
    data = df[[col_sample, col_cfd, col_nir, col_corr]].copy()
    data = data.dropna(subset=[col_sample])

    # Convert to numeric (coerce errors to NaN then drop)
    for c in [col_sample, col_cfd, col_nir, col_corr]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[col_sample])

    # Sort by sample no just in case
    data = data.sort_values(col_sample)

    samples = data[col_sample].astype(int).to_numpy()
    cfd_vals = data[col_cfd].to_numpy()
    nir_vals = data[col_nir].to_numpy()
    corr_vals = data[col_corr].to_numpy()

    # Grouped bar positions
    x = np.arange(len(samples))
    width = 0.26
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"          # 수식 글꼴을 Times 계열과 어울리게
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(max(8, 0.6 * len(samples)), 4.5))
    plt.bar(x - width, cfd_vals, width=width, label="CFD Value")
    plt.bar(x,         nir_vals, width=width, label="Nir Correlation")
    plt.bar(x + width, corr_vals, width=width, label="Correlation Value")

    plt.xticks(x, samples)
    plt.xlabel("Sample No.", fontsize=15)
    plt.ylabel("Pressure drop (Pa)", fontsize=15)  # 단위가 다르면 여기만 바꾸면 됨
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
