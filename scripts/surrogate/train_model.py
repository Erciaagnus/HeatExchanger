#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate (GP) training + GA optimization for fin/tube heat exchanger design.

- Trains two GP surrogates for:
    Q'' (heat flux) and ΔP (pressure drop)
- Uses GA (pygad) to maximize either:
    mean ratio: mu_Q / mu_dP
  or conservative ratio:
    (mu_Q - k*sigma_Q) / (mu_dP + k*sigma_dP)

Model persistence:
- Saves/loads a single bundle .joblib file to:
    ../../model/gp_surrogate_bundle.joblib
  (relative to THIS script file location)

Usage examples:

1) Train and save model only:
   python surrogate_ga_optimize.py --mode train --data total_2D_Data.xlsx

2) Run GA using saved model (no retraining):
   python surrogate_ga_optimize.py --mode ga

3) Train + GA in one shot (and save model):
   python surrogate_ga_optimize.py --mode train_ga --data total_2D_Data.xlsx

4) Predict at a custom design using saved model:
   python surrogate_ga_optimize.py --mode predict --s1 181.0 --fh 29.0 --fs 2.6

Notes:
- Input features are (S1, fin height, fin spacing).
- Targets are inferred from column names via keyword matching.
- GP uses StandardScaler for X and y. Viscous/thermal transforms are not applied here (fits your original code).
"""

import argparse
import os
import warnings
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pygad
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# =============================================================================
# Kernel selection (aligned with your design_version_5 style)
# =============================================================================
def find_best_kernel(X: np.ndarray, y: np.ndarray, random_state: int = 30):
    """
    Find the best kernel using 5-fold CV R^2, then fit on all data.
    """
    noise_bounds = (1e-10, 1e1)
    kernels = [
        C(1.0) * Matern(length_scale=[1.0] * 3, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RBF(length_scale=[1.0] * 3) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RationalQuadratic(alpha=0.1) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * (DotProduct() + Matern(nu=2.5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
    ]
    names = ["Matern(2.5)", "RBF", "RationalQuad", "Composite"]

    best_score = -np.inf
    best_model = None
    best_name = ""

    for k, name in zip(kernels, names):
        gp = GaussianProcessRegressor(
            kernel=k,
            n_restarts_optimizer=10,
            normalize_y=False,
            random_state=random_state,
        )
        scores = cross_val_score(gp, X, y, cv=5, scoring="r2")
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_model = gp
            best_name = name

    assert best_model is not None
    best_model.fit(X, y)
    try:
        print(f"      [Kernel fitted] {best_model.kernel_}")
    except Exception:
        pass

    return best_model, best_name, best_score


# =============================================================================
# Column detection helpers
# =============================================================================
def _find_column(df: pd.DataFrame, keywords) -> Optional[str]:
    for kw in keywords:
        matches = [c for c in df.columns if kw in c.lower()]
        if matches:
            return matches[0]
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns a mapping for:
      s1, h, s, q, dp
    """
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {
        "s1": _find_column(df, ["s1_mm", "s1"]),
        "h":  _find_column(df, ["fin_height", "fh_mm", "height"]),
        "s":  _find_column(df, ["fin_spacing", "fs_mm", "spacing"]),
        "q":  _find_column(df, ["q''", "qpp", "heat flux", "flux"]),
        "dp": _find_column(df, ["delta p", "delta_p", "deltap", "dp"]),
    }

    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return col_map


def read_table_auto(file_path: str) -> pd.DataFrame:
    """
    Read .xlsx/.xls with auto header row detection; else read .csv.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    if file_path.lower().endswith((".xlsx", ".xls")):
        try:
            df_raw = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
        except Exception:
            df_raw = pd.read_excel(file_path, header=None)

        header_row = 0
        for i in range(min(30, len(df_raw))):
            row_str = " ".join(
                [str(val).lower() for val in df_raw.iloc[i].values if pd.notna(val)]
            )
            if ("s1" in row_str and ("height" in row_str or "spacing" in row_str)) or ("s1_mm" in row_str):
                header_row = i
                print(f"   -> Header row detected: row {header_row}")
                break

        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1", header=header_row)
        except Exception:
            df = pd.read_excel(file_path, header=header_row)

        return df

    # CSV
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding="cp949")
        except Exception:
            df = pd.read_csv(file_path, encoding="latin1")

    # If header seems off, try header=2 fallback like your original
    lower_cols = [str(c).lower() for c in df.columns]
    if "s1" not in lower_cols and "s1_mm" not in lower_cols:
        try:
            df2 = pd.read_csv(file_path, header=2)
            df = df2
        except Exception:
            pass

    return df


# =============================================================================
# Model bundle save/load
# =============================================================================
def default_model_path() -> str:
    """
    Save to ../../model/gp_surrogate_bundle.joblib relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "model"))
    return os.path.join(model_dir, "gp_surrogate_bundle.joblib")


def save_surrogate_bundle(models, scaler_X, scalers_y, out_path: str, meta: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bundle = {
        "models": models,
        "scaler_X": scaler_X,
        "scalers_y": scalers_y,
        "meta": meta,
    }
    joblib.dump(bundle, out_path)
    print(f"   -> [Saved] surrogate bundle: {os.path.abspath(out_path)}")


def load_surrogate_bundle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    print(f"   -> [Loaded] surrogate bundle: {os.path.abspath(path)}")
    return bundle["models"], bundle["scaler_X"], bundle["scalers_y"], bundle.get("meta", {})


# =============================================================================
# Train surrogate models
# =============================================================================
def train_surrogates(
    data_path: str,
    random_state: int = 30,
):
    print("1. [System] 데이터 로드 및 대리 모델 학습 중...")

    # Resolve path (if relative, interpret relative to script location)
    if not os.path.isabs(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    # extension fallback
    if not os.path.exists(data_path):
        if os.path.exists(data_path + ".xlsx"):
            data_path += ".xlsx"
        elif os.path.exists(data_path + ".csv"):
            data_path += ".csv"
        else:
            raise FileNotFoundError(f"Cannot find data file: {data_path}")

    print(f"   -> 파일 로드: {data_path}")

    df = read_table_auto(data_path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"   -> 컬럼(앞 10개): {df.columns.tolist()[:10]}...")

    col_map = detect_columns(df)
    print(f"   -> 컬럼 매핑 완료: {col_map}")

    # Clean rows with missing targets
    df = df.dropna(subset=[col_map["dp"], col_map["q"]]).copy()

    # X and scaling
    X = df[[col_map["s1"], col_map["h"], col_map["s"]]].values.astype(float)
    scaler_X = StandardScaler().fit(X)
    X_sc = scaler_X.transform(X)

    models = {}
    scalers_y = {}

    target_labels = {"q": "Q'' [W/m^2]", "dp": "Delta P [Pa]"}
    cv_scores = {}

    for target in ["q", "dp"]:
        print(f"   -> [{target_labels[target]}] 모델 최적화 중...")
        y = df[col_map[target]].values.astype(float)

        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        y_sc = scaler_y.transform(y.reshape(-1, 1)).ravel()

        gp, k_name, k_cv = find_best_kernel(X_sc, y_sc, random_state=random_state)
        print(f"      Best Kernel: {k_name} | CV R2(mean): {k_cv:.4f}")

        models[target] = gp
        scalers_y[target] = scaler_y
        cv_scores[target] = {"kernel": k_name, "cv_r2_mean": float(k_cv)}

    print("   -> 대리 모델 학습 완료.")
    return models, scaler_X, scalers_y, col_map, os.path.abspath(data_path), cv_scores


# =============================================================================
# Prediction utilities
# =============================================================================
def _inverse_mean_std(scaler_y: StandardScaler, mu_sc: float, std_sc: float):
    """
    Convert mean/std from standardized y-space back to raw y-space.
    If y_sc = (y - mean)/scale, then:
      mu_raw = mu_sc*scale + mean
      std_raw = std_sc*scale
    """
    scale = float(scaler_y.scale_[0])
    mean = float(scaler_y.mean_[0])
    mu_raw = float(mu_sc) * scale + mean
    std_raw = float(std_sc) * scale
    return mu_raw, std_raw


def predict_q_dp(models, scaler_X, scalers_y, s1: float, fh: float, fs: float):
    x = np.array([[s1, fh, fs]], dtype=float)
    x_sc = scaler_X.transform(x)

    q_mu_sc, q_std_sc = models["q"].predict(x_sc, return_std=True)
    dp_mu_sc, dp_std_sc = models["dp"].predict(x_sc, return_std=True)

    q_mu, q_std = _inverse_mean_std(scalers_y["q"], float(q_mu_sc[0]), float(q_std_sc[0]))
    dp_mu, dp_std = _inverse_mean_std(scalers_y["dp"], float(dp_mu_sc[0]), float(dp_std_sc[0]))

    return q_mu, q_std, dp_mu, dp_std


# =============================================================================
# GA fitness factory
# =============================================================================
def make_fitness_func(
    models,
    scaler_X,
    scalers_y,
    use_conservative: bool,
    k_sigma: float,
    dp_floor: float,
):
    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness to maximize.

        Constraints:
          1) 45 <= s1 <= 200
          2) 6 <= fh <= 0.5*((s1/sqrt(2))-24)-0.4
          3) 2 <= fs <= 8
        """
        s1, fh, fs = solution

        # Constraint 2 (geometry)
        fh_limit = 0.5 * ((s1 / np.sqrt(2)) - 24.0) - 0.4
        if fh > (fh_limit + 1e-3):
            return -1000.0

        # Constraint 1 & 3 and fh lower bound
        if (s1 < 45) or (s1 > 200) or (fs < 2) or (fs > 8) or (fh < 6):
            return -1000.0

        # Predict
        try:
            q_mu, q_std, dp_mu, dp_std = predict_q_dp(models, scaler_X, scalers_y, s1, fh, fs)

            if use_conservative:
                q_eff = q_mu - k_sigma * q_std
                dp_eff = dp_mu + k_sigma * dp_std

                if (dp_eff <= dp_floor) or (not np.isfinite(dp_eff)) or (not np.isfinite(q_eff)):
                    return -500.0
                if q_eff <= 0:
                    return -500.0

                return float(q_eff / dp_eff)

            # mean only
            if (dp_mu <= dp_floor) or (not np.isfinite(dp_mu)) or (not np.isfinite(q_mu)):
                return -500.0
            if q_mu <= 0:
                return -500.0

            return float(q_mu / dp_mu)

        except Exception:
            return -1000.0

    return fitness_func


# =============================================================================
# GA runner
# =============================================================================
def run_ga(
    models,
    scaler_X,
    scalers_y,
    use_conservative: bool = True,
    k_sigma: float = 1.0,
    dp_floor: float = 1e-3,
    generations: int = 100,
    pop_size: int = 50,
    parents_mating: int = 10,
    random_seed: int = 30,
):
    print("\n2. [System] GA(유전 알고리즘) 최적화 시작...")
    print(f"   - Fitness: {'Conservative (mu±k·sigma)' if use_conservative else 'Mean (mu only)'}")
    if use_conservative:
        print(f"   - k (sigma multiplier): {k_sigma}")
    print(f"   - 세대 수(Generations): {generations}")
    print(f"   - 인구 수(Population): {pop_size}")
    print("   - 탐색 전략: Q''/dP 최대화 (제약조건 준수)")

    fitness_func = make_fitness_func(
        models=models,
        scaler_X=scaler_X,
        scalers_y=scalers_y,
        use_conservative=use_conservative,
        k_sigma=k_sigma,
        dp_floor=dp_floor,
    )

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=3,
        gene_space=[
            {"low": 45, "high": 200},  # S1
            {"low": 6, "high": 40},    # FH (upper bound enforced by constraint)
            {"low": 2, "high": 8},     # FS
        ],
        parent_selection_type="rank",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        random_seed=random_seed,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    opt_s1, opt_fh, opt_fs = [float(x) for x in solution]
    fh_limit = 0.5 * ((opt_s1 / np.sqrt(2)) - 24.0) - 0.4

    # Report mean ± std at optimum
    q_mu, q_std, dp_mu, dp_std = predict_q_dp(models, scaler_X, scalers_y, opt_s1, opt_fh, opt_fs)

    q_eff = q_mu - k_sigma * q_std
    dp_eff = dp_mu + k_sigma * dp_std

    mean_ratio = q_mu / max(dp_mu, dp_floor)
    cons_ratio = q_eff / max(dp_eff, dp_floor)

    print("\n" + "=" * 70)
    print("       🧬 GA(Genetic Algorithm) 최적 설계 결과 (Surrogate-based)")
    print("=" * 70)
    print("1. 최적 설계변수 (Optimal Design Variables):")
    print(f"   - S1 (mm)          : {opt_s1:.4f}")
    print(f"   - Fin Height (mm)  : {opt_fh:.4f}  (Limit: {fh_limit:.4f})")
    print(f"   - Fin Spacing (mm) : {opt_fs:.4f}")

    print("\n2. Surrogate 예측 (Mean ± Std):")
    print(f"   - Q''  : {q_mu:.2f} ± {q_std:.2f}  (W/m^2)")
    print(f"   - ΔP   : {dp_mu:.6f} ± {dp_std:.6f}  (Pa)")

    print("\n3. 목적함수 값:")
    print(f"   - Mean ratio         (mu_Q / mu_ΔP)                 : {mean_ratio:.6f}")
    print(f"   - Conservative ratio ((mu_Q-kσ_Q)/(mu_ΔP+kσ_ΔP))    : {cons_ratio:.6f}")
    print(f"   - GA가 최대화한 fitness 값                          : {float(solution_fitness):.6f}")

    print("\n4. 제약조건 검증:")
    is_valid_fh = opt_fh <= (fh_limit + 1e-3)
    is_valid_basic = (45 <= opt_s1 <= 200) and (2 <= opt_fs <= 8) and (opt_fh >= 6)
    print(f"   - Height 제약 만족?  {'[PASS]' if is_valid_fh else '[FAIL]'}")
    print(f"   - 기본 범위 만족?    {'[PASS]' if is_valid_basic else '[FAIL]'}")

    return {
        "S1_mm": opt_s1,
        "fin_height_fh_mm": opt_fh,
        "fin_spacing_fs_mm": opt_fs,
        "fh_limit_mm": fh_limit,
        "Q_mu": q_mu,
        "Q_std": q_std,
        "dP_mu": dp_mu,
        "dP_std": dp_std,
        "mean_ratio": mean_ratio,
        "conservative_ratio": cons_ratio,
        "fitness": float(solution_fitness),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train_ga", choices=["train", "ga", "train_ga", "predict"],
                    help="train: train+save | ga: load+GA | train_ga: train+save+GA | predict: load+predict")
    ap.add_argument("--data", type=str, default="total_2D_Data.xlsx",
                    help="Training data file (xlsx/csv). Used for train/train_ga.")
    ap.add_argument("--model_path", type=str, default=default_model_path(),
                    help="Path to .joblib model bundle (default: ../../model/gp_surrogate_bundle.joblib relative to script)")
    ap.add_argument("--random_state", type=int, default=30, help="Random seed for GP kernel selection / fitting")

    # Fitness settings
    ap.add_argument("--use_conservative", action="store_true", help="Use conservative fitness (default)")
    ap.add_argument("--mean_only", action="store_true", help="Use mean-only fitness (overrides --use_conservative)")
    ap.add_argument("--k_sigma", type=float, default=1.0, help="Sigma multiplier for conservative fitness")
    ap.add_argument("--dp_floor", type=float, default=1e-3, help="Floor for ΔP to avoid exploding ratios")

    # GA settings
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--parents", type=int, default=10)
    ap.add_argument("--ga_seed", type=int, default=30)

    # Predict settings
    ap.add_argument("--s1", type=float, default=None)
    ap.add_argument("--fh", type=float, default=None)
    ap.add_argument("--fs", type=float, default=None)

    args = ap.parse_args()

    # Determine fitness mode
    use_conservative = True
    if args.mean_only:
        use_conservative = False
    elif args.use_conservative:
        use_conservative = True

    # TRAIN
    if args.mode in ["train", "train_ga"]:
        models, scaler_X, scalers_y, col_map, abs_data_path, cv_scores = train_surrogates(
            args.data,
            random_state=args.random_state,
        )

        meta = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "source_data": abs_data_path,
            "columns": col_map,
            "cv_scores": cv_scores,
            "fitness_defaults": {
                "use_conservative": use_conservative,
                "k_sigma": float(args.k_sigma),
                "dp_floor": float(args.dp_floor),
            },
        }
        save_surrogate_bundle(models, scaler_X, scalers_y, args.model_path, meta)

        if args.mode == "train":
            return

    # LOAD (for ga / predict / train_ga second phase)
    if args.mode in ["ga", "predict", "train_ga"]:
        models, scaler_X, scalers_y, meta = load_surrogate_bundle(args.model_path)

    # GA
    if args.mode in ["ga", "train_ga"]:
        run_ga(
            models=models,
            scaler_X=scaler_X,
            scalers_y=scalers_y,
            use_conservative=use_conservative,
            k_sigma=float(args.k_sigma),
            dp_floor=float(args.dp_floor),
            generations=int(args.generations),
            pop_size=int(args.pop),
            parents_mating=int(args.parents),
            random_seed=int(args.ga_seed),
        )
        return

    # PREDICT
    if args.mode == "predict":
        if args.s1 is None or args.fh is None or args.fs is None:
            raise ValueError("For --mode predict, you must provide --s1 --fh --fs")

        s1, fh, fs = float(args.s1), float(args.fh), float(args.fs)

        # Check constraints (same as GA)
        fh_limit = 0.5 * ((s1 / np.sqrt(2)) - 24.0) - 0.4
        if not (45 <= s1 <= 200 and 2 <= fs <= 8 and fh >= 6 and fh <= fh_limit + 1e-3):
            print("[WARN] Given design violates constraints:")
            print(f"  S1 in [45,200]? {45 <= s1 <= 200}")
            print(f"  FS in [2,8]?    {2 <= fs <= 8}")
            print(f"  FH >= 6?        {fh >= 6}")
            print(f"  FH <= limit?    {fh <= fh_limit + 1e-3}  (limit={fh_limit:.4f})")

        q_mu, q_std, dp_mu, dp_std = predict_q_dp(models, scaler_X, scalers_y, s1, fh, fs)
        q_eff = q_mu - args.k_sigma * q_std
        dp_eff = dp_mu + args.k_sigma * dp_std

        mean_ratio = q_mu / max(dp_mu, args.dp_floor)
        cons_ratio = q_eff / max(dp_eff, args.dp_floor)

        print("\n" + "=" * 70)
        print("       🔎 Surrogate Prediction (Loaded Model)")
        print("=" * 70)
        print(f"Design: S1={s1:.4f} mm, FH={fh:.4f} mm (limit={fh_limit:.4f}), FS={fs:.4f} mm")
        print(f"Q'':  {q_mu:.2f} ± {q_std:.2f} (W/m^2)")
        print(f"ΔP :  {dp_mu:.6f} ± {dp_std:.6f} (Pa)")
        print(f"Mean ratio: {mean_ratio:.6f}")
        print(f"Conservative ratio (k={args.k_sigma}): {cons_ratio:.6f}")
        return


if __name__ == "__main__":
    main()
