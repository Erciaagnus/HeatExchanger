
"""
Generate constrained LHS samples for fin/tube design variables.

Constraints (mm):
- s := S1 = S2 in [45, 200]
- f_s (fin spacing) in [2, 8]
- f_h (fin height) satisfies:
    6 <= f_h <= 0.5*( s/sqrt(2) - t_d ) - 0.4
  with fixed tube diameter t_d = 24 mm

Important:
- Because f_h has a lower bound (6 mm) and an s-dependent upper bound,
  some s values near the original lower bound become infeasible.
- This script enforces feasibility by sampling s only from the feasible range:
    s >= s_min_feasible such that fh_max(s) >= fh_min.

Output:
- CSV with columns: S1_mm, S2_mm, fin_height_fh_mm, fin_spacing_fs_mm
- Optionally also writes fh_upper_mm for diagnostics.

Usage:
  python generate_constrained_lhs_100k.py --n 100000 --seed 2025 --out constrained_LHS_100k.csv
"""

import argparse
from math import sqrt
import numpy as np
import pandas as pd
from scipy.stats import qmc


def fh_max(s: np.ndarray, td: float, margin_outside: float) -> np.ndarray:
    """
    Upper bound:
        f_h <= 0.5*( s/sqrt(2) - td ) - margin_outside
    """
    return 0.5*(s/sqrt(2) - td) - margin_outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="Number of samples")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed for LHS")
    ap.add_argument("--out", type=str, default="../../data/constrained_LHS_100k.csv", help="Output CSV path")
    ap.add_argument("--include_diag", action="store_true", help="Include fh_upper_mm diagnostic column")
    args = ap.parse_args()

    # Parameters / constraints (mm)
    s_min, s_max = 45.0, 200.0
    fh_min = 6.0
    fs_min, fs_max = 2.0, 8.0
    td = 24.0
    margin_outside = 0.4

    # Effective feasible s lower bound so that fh_max(s) >= fh_min
    # fh_min <= 0.5*(s/sqrt2 - td) - margin_outside
    # => s/sqrt2 >= 2*(fh_min + margin_outside) + td
    s_min_feasible = max(s_min, (2*(fh_min + margin_outside) + td) * sqrt(2))

    if s_min_feasible >= s_max:
        raise ValueError(
            f"Infeasible configuration: s_min_feasible={s_min_feasible:.6f} >= s_max={s_max}. "
            "Relax constraints or increase s_max."
        )

    # LHS in unit cube (3 dims: s, f_h, f_s)
    sampler = qmc.LatinHypercube(d=3, seed=args.seed)
    U = sampler.random(n=args.n)

    u_s, u_fh, u_fs = U[:, 0], U[:, 1], U[:, 2]

    # Map to physical variables
    s = s_min_feasible + u_s * (s_max - s_min_feasible)
    fs = fs_min + u_fs * (fs_max - fs_min)

    fh_upper = fh_max(s, td=td, margin_outside=margin_outside)
    span = fh_upper - fh_min
    if np.any(span <= 0):
        # Shouldn't happen if s_min_feasible is computed correctly
        raise RuntimeError("Infeasible samples found; check feasible s_min computation.")

    fh = fh_min + u_fh * span

    df = pd.DataFrame({
        "S1_mm": s,
        "S2_mm": s,  # S1 = S2
        "fin_height_fh_mm": fh,
        "fin_spacing_fs_mm": fs,
    })

    if args.include_diag:
        df["fh_upper_mm"] = fh_upper

    # Sanity checks
    eps = 1e-10
    violations = {
        "S_range": int(np.sum((df["S1_mm"] < s_min - eps) | (df["S1_mm"] > s_max + eps))),
        "fs_range": int(np.sum((df["fin_spacing_fs_mm"] < fs_min - eps) | (df["fin_spacing_fs_mm"] > fs_max + eps))),
        "fh_lower": int(np.sum(df["fin_height_fh_mm"] < fh_min - eps)),
        "fh_upper": int(np.sum(df["fin_height_fh_mm"] > fh_upper + eps)),
        # Equivalent coupled form:
        # fh <= 0.5*(s/sqrt2 - td) - 0.4  <=>  2*fh + td + 0.8 <= s/sqrt2
        "coupled": int(np.sum(2*df["fin_height_fh_mm"] + td + 2*margin_outside > (df["S1_mm"]/sqrt(2)) + eps)),
    }

    print("=== Constrained LHS generation summary ===")
    print(f"n={args.n}, seed={args.seed}")
    print(f"s_min_feasible={s_min_feasible:.6f} mm (implied by fh_min and coupled constraint)")
    print(f"S1 range: [{df['S1_mm'].min():.6f}, {df['S1_mm'].max():.6f}]")
    print(f"fh range: [{df['fin_height_fh_mm'].min():.6f}, {df['fin_height_fh_mm'].max():.6f}]")
    print(f"fs range: [{df['fin_spacing_fs_mm'].min():.6f}, {df['fin_spacing_fs_mm'].max():.6f}]")
    print("violations:", violations)

    out_path = args.out
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
