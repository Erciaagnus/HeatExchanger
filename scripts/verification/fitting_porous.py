#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Default conditions (Haenam annual averages)
# =========================
DEFAULT_L_M = 0.625          # 2000 mm -> 2.0 m
DEFAULT_RHO = 1.226        # kg/m^3
DEFAULT_MU = 1.79e-5       # Pa·s

# CSV Column Data
DEFAULT_U_COL = "u [m/s]"
DEFAULT_DP_COL = "dp [Pa]"


def fit_ab(U: np.ndarray, dP: np.ndarray, force_nonnegative: bool = False):
    """
    Fit dP = a U + b U^2 using least squares (no intercept).
    Returns a, b, and diagnostics.
    """
    U = np.asarray(U, dtype=float).reshape(-1)
    dP = np.asarray(dP, dtype=float).reshape(-1)

    if U.size != dP.size:
        raise ValueError("U and dP must have same length.")
    if U.size < 3:
        raise ValueError("Need at least 3 points to fit reliably.")

    X = np.column_stack([U, U**2])  # [U, U^2]
    coef, residuals, rank, s = np.linalg.lstsq(X, dP, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    if force_nonnegative:
        a = max(a, 0.0)
        b = max(b, 0.0)

    dP_hat = X @ np.array([a, b])
    resid = dP - dP_hat

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((dP - np.mean(dP)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))

    return a, b, dP_hat, resid, r2, rmse


def main():
    p = argparse.ArgumentParser()

    p.add_argument("csv", type=str, help="CSV file with velocity and pressure drop columns.")

    # column names
    p.add_argument("--u_col", type=str, default=DEFAULT_U_COL,
                   help=f"Velocity column name (m/s). Default: '{DEFAULT_U_COL}'")
    p.add_argument("--dp_col", type=str, default=DEFAULT_DP_COL,
                   help=f"Pressure drop column name (Pa). Default: '{DEFAULT_DP_COL}'")

    # porous-zone thickness (flow direction)
    p.add_argument("--L", type=float, default=DEFAULT_L_M,
                   help=f"Porous thickness/length L in flow direction (m). Default: {DEFAULT_L_M}")

    # environmental conditions
    p.add_argument("--rho", type=float, default=DEFAULT_RHO,
                   help=f"Air density rho (kg/m^3). Default: {DEFAULT_RHO}")
    p.add_argument("--mu", type=float, default=DEFAULT_MU,
                   help=f"Dynamic viscosity mu (Pa·s). Default: {DEFAULT_MU}")

    p.add_argument("--force_nonnegative", action="store_true",
                   help="Clamp negative a,b to 0 (quick safeguard; not a strict constrained fit).")
    p.add_argument("--save_fig", action="store_true", help="Save plots as PNG.")
    p.add_argument("--out_prefix", type=str, default="porous_fit", help="Output prefix for figs/csv.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    if args.u_col not in df.columns or args.dp_col not in df.columns:
        raise KeyError(
            f"CSV must contain columns '{args.u_col}' and '{args.dp_col}'. "
            f"Found: {list(df.columns)}"
        )

    U = df[args.u_col].to_numpy(dtype=float)
    dP = df[args.dp_col].to_numpy(dtype=float)

    # Sort by U for cleaner plots
    order = np.argsort(U)
    U, dP = U[order], dP[order]

    a, b, dP_hat, resid, r2, rmse = fit_ab(U, dP, force_nonnegative=args.force_nonnegative)

    # Convert to porous parameters (Fluent/CFD Darcy–Forchheimer)
    # ΔP = (μ/K) L U + (1/2 ρ C2) L U^2
    if a <= 0:
        K = np.inf
        print("[WARN] Fitted a <= 0, K becomes inf/invalid. Check data/definition.")
    else:
        K = args.mu * args.L / a

    C2 = (2.0 * b) / (args.rho * args.L) if args.L > 0 else np.nan

    print("\n=== Fit results (ΔP = a U + b U^2) ===")
    print(f"a = {a:.6e}")
    print(f"b = {b:.6e}")
    print(f"R^2 = {r2:.6f}")
    print(f"RMSE = {rmse:.6e} Pa")

    print("\n=== Porous parameters (Fluent-style) ===")
    print(f"L   = {args.L:.6g} m")
    print(f"rho = {args.rho:.6g} kg/m^3")
    print(f"mu  = {args.mu:.6g} Pa·s")
    print(f"K   = {K:.6e} m^2")
    print(f"C2  = {C2:.6e} 1/m")

    # Save fitted table
    out = pd.DataFrame({
        "U_mps": U,
        "dP_Pa": dP,
        "dP_fit_Pa": dP_hat,
        "resid_Pa": resid
    })
    out_csv = f"{args.out_prefix}_fitted.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved fitted table: {out_csv}")

    # ---- Plots ----
    plt.figure()
    plt.plot(U, dP, marker="o", linestyle="none", label="data")
    plt.plot(U, dP_hat, linestyle="-", label="fit: aU + bU^2")
    plt.xlabel("U (m/s)")
    plt.ylabel("ΔP (Pa)")
    plt.title("Porous fit: ΔP vs U")
    plt.legend()
    plt.grid(True)
    if args.save_fig:
        plt.savefig(f"{args.out_prefix}_dP_vs_U.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(U, dP / args.L, marker="o", linestyle="none", label="data (ΔP/L)")
    plt.plot(U, dP_hat / args.L, linestyle="-", label="fit (ΔP/L)")
    plt.xlabel("U (m/s)")
    plt.ylabel("ΔP/L (Pa/m)")
    plt.title("Pressure gradient: ΔP/L vs U")
    plt.legend()
    plt.grid(True)
    if args.save_fig:
        plt.savefig(f"{args.out_prefix}_dP_over_L_vs_U.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(U, resid, marker="o", linestyle="-")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("U (m/s)")
    plt.ylabel("Residual (Pa)")
    plt.title("Residuals (data - fit)")
    plt.grid(True)
    if args.save_fig:
        plt.savefig(f"{args.out_prefix}_residuals.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
