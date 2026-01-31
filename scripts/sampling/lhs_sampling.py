import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial import cKDTree, distance
from scipy.optimize import linear_sum_assignment


CANON = ["Porosity", "Viscous_Resistance_1_m2", "Inertial_Resistance_1_m"]
ALIASES = {
    "Porosity": ["Porosity", "porosity", "epsilon", "eps"],
    "Viscous_Resistance_1_m2": ["Viscous_Resistance_1_m2", "inv_K_1_m2", "viscous_resistance", "viscous"],
    "Inertial_Resistance_1_m": ["Inertial_Resistance_1_m", "C2_1_m", "inertial_resistance", "inertial", "C2"],
}


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = set(df.columns)
    mapping: Dict[str, str] = {}
    for canon, alias_list in ALIASES.items():
        found = None
        for a in alias_list:
            if a in cols:
                found = a
                break
        if found is None:
            raise ValueError(f"Missing required column for {canon}. Accepted: {alias_list}. Found: {list(df.columns)}")
        mapping[canon] = found
    return mapping


def safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        min_pos = np.min(x[x > 0]) if np.any(x > 0) else 1.0
        x = np.where(x <= 0, min_pos * 1e-6, x)
    return np.log10(x)


def minmax_scale(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x), lo, hi
    return (x - lo) / (hi - lo), lo, hi


def min_pairwise_dist(U: np.ndarray) -> float:
    D = distance.pdist(U, metric="euclidean")
    return float(D.min()) if D.size else 0.0


def score_selection(Xsel: np.ndarray, metric: str) -> float:
    """
    Higher is better.
    - maximin: maximize minimum pairwise distance in scaled space
    - meanNN : maximize mean nearest-neighbor distance
    """
    if Xsel.shape[0] < 2:
        return -np.inf
    if metric == "maximin":
        return min_pairwise_dist(Xsel)
    elif metric == "meanNN":
        D = distance.squareform(distance.pdist(Xsel, metric="euclidean"))
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        return float(np.mean(nn))
    else:
        raise ValueError("Unknown metric: " + metric)


def project_targets_to_rows(Xdata: np.ndarray, T: np.ndarray, k: int) -> np.ndarray:
    """
    Xdata: (Ndata,3) scaled coordinates of dataset rows
    T:     (n,3) scaled LHS targets
    k:     nearest neighbors per target to build candidate pool

    Returns indices into Xdata of chosen unique rows (length n).
    """
    tree = cKDTree(Xdata)
    n_rows = Xdata.shape[0]
    n = T.shape[0]
    k = int(min(max(10, k), n_rows))

    # candidate pool = union of kNN per target
    cand = []
    for t in T:
        _, idx = tree.query(t, k=k)
        idx = np.atleast_1d(idx).astype(int)
        cand.append(idx)
    pool = np.unique(np.concatenate(cand))
    M = len(pool)

    # cost matrix: n x M
    Xpool = Xdata[pool]
    cost = np.empty((n, M), dtype=float)
    for i in range(n):
        dif = Xpool - T[i]
        cost[i, :] = np.sqrt(np.sum(dif * dif, axis=1))

    _, col_ind = linear_sum_assignment(cost)
    chosen = pool[col_ind]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="../../data/porous_from_design.csv", help="Input CSV (full feasible dataset)")
    ap.add_argument("--out", dest="out_csv", default="../../data/porous_LHS_100_projected.csv", help="Output CSV (selected)")
    ap.add_argument("--report", dest="report_txt", default="porous_LHS_100_projected_report.txt", help="Text report")
    ap.add_argument("--n", type=int, default=100, help="Number of samples to select")
    ap.add_argument("--seed_base", type=int, default=2025, help="Base seed for multi-start")
    ap.add_argument("--trials", type=int, default=200, help="Number of LHS target trials")
    ap.add_argument("--k", type=int, default=250, help="kNN per target for candidate pool")
    ap.add_argument("--metric", type=str, default="maximin", choices=["maximin", "meanNN"], help="Selection quality metric")
    ap.add_argument("--no_log", action="store_true", help="Use linear resistance axes (not recommended)")
    ap.add_argument("--save_scaled_cols", action="store_true", help="Save scaled coordinates used for sampling")
    args = ap.parse_args()

    # Get script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_csv_path = os.path.join(script_dir, args.in_csv) if not os.path.isabs(args.in_csv) else args.in_csv
    out_csv_path = os.path.join(script_dir, args.out_csv) if not os.path.isabs(args.out_csv) else args.out_csv
    report_txt_path = os.path.join(script_dir, args.report_txt) if not os.path.isabs(args.report_txt) else args.report_txt

    # Read CSV with error handling and encoding support
    try:
        df = pd.read_csv(in_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(in_csv_path, encoding='cp949')
        except:
            df = pd.read_csv(in_csv_path, encoding='latin1')
    
    cols = resolve_columns(df)

    # Clean and keep original row index for traceability
    # Save original index before resetting
    df_with_orig_idx = df.dropna(subset=[cols[c] for c in CANON]).copy()
    df_with_orig_idx['original_row_index'] = df_with_orig_idx.index
    df_clean = df_with_orig_idx.reset_index(drop=True)

    n_rows = len(df_clean)
    n = int(min(args.n, n_rows))

    # Original-space min/max
    stats_orig = {}
    for canon in CANON:
        s = df_clean[cols[canon]].astype(float)
        stats_orig[canon] = {"min": float(s.min()), "max": float(s.max())}

    # Build sampling/matching space
    P = df_clean[cols["Porosity"]].astype(float).to_numpy()
    V = df_clean[cols["Viscous_Resistance_1_m2"]].astype(float).to_numpy()
    I = df_clean[cols["Inertial_Resistance_1_m"]].astype(float).to_numpy()

    if args.no_log:
        Vt, It = V, I
        v_label, i_label = "Viscous", "Inertial"
    else:
        Vt, It = safe_log10(V), safe_log10(I)
        v_label, i_label = "log10(Viscous)", "log10(Inertial)"

    Pn, Plo, Phi = minmax_scale(P)
    Vn, Vlo, Vhi = minmax_scale(Vt)
    In, Ilo, Ihi = minmax_scale(It)
    Xdata = np.column_stack([Pn, Vn, In])

    best = None
    best_score = -np.inf
    best_seed = None

    # Multi-start: choose LHS targets, project, score in scaled space
    for t in range(int(args.trials)):
        seed = int(args.seed_base + t)
        sampler = qmc.LatinHypercube(d=3, seed=seed)
        T = sampler.random(n=n)  # targets in [0,1]^3

        chosen = project_targets_to_rows(Xdata, T, k=int(args.k))
        Xsel = Xdata[chosen]
        sc = score_selection(Xsel, metric=args.metric)

        if sc > best_score:
            best_score = sc
            best = chosen
            best_seed = seed

    # Build selected dataframe
    sel = df_clean.iloc[best].copy()
    sel.insert(0, "Sample_No", np.arange(1, len(sel) + 1))

    if args.save_scaled_cols:
        sel["scaled_Porosity"] = Pn[best]
        sel[f"scaled_{v_label}"] = Vn[best]
        sel[f"scaled_{i_label}"] = In[best]

    # Save CSV with UTF-8 BOM for better compatibility
    sel.to_csv(out_csv_path, index=False, encoding='utf-8-sig')

    # Report
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Min/Max (original columns, full data) ===\n")
        for canon in CANON:
            f.write(f"{canon} ({cols[canon]}): min={stats_orig[canon]['min']}, max={stats_orig[canon]['max']}\n")
        f.write("\n=== Sampling space ===\n")
        f.write(f"Axis-1: Porosity (linear), min={Plo}, max={Phi}\n")
        f.write(f"Axis-2: {v_label}, min={Vlo}, max={Vhi}\n")
        f.write(f"Axis-3: {i_label}, min={Ilo}, max={Ihi}\n")
        f.write("\n=== Selection ===\n")
        f.write(f"rows_used={n_rows}\n")
        f.write(f"n_selected={len(sel)}\n")
        f.write(f"best_seed={best_seed}\n")
        f.write(f"trials={args.trials}\n")
        f.write(f"kNN_per_target={args.k}\n")
        f.write(f"metric={args.metric}\n")
        f.write(f"best_score={best_score}\n")

    print("=== Done ===")
    print("Selected CSV:", os.path.abspath(out_csv_path))
    print("Report:", os.path.abspath(report_txt_path))
    print("Min/Max (full data, original):")
    for canon in CANON:
        print(f"  {canon} ({cols[canon]}): min={stats_orig[canon]['min']}, max={stats_orig[canon]['max']}")
    print("Best seed:", best_seed, "| metric:", args.metric, "| score:", best_score)


if __name__ == "__main__":
    main()
