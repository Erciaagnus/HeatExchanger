import pandas as pd
import numpy as np
import os
import warnings
import pygad  

# Surrogate modeling
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
# Kernel selection (aligned with design_version_5)
# =============================================================================
def find_best_kernel(X, y):
    """Find the best kernel using cross-validation (from design_version_5)"""
    from sklearn.model_selection import cross_val_score
    
    noise_bounds = (1e-10, 1e1)
    kernels = [
        C(1.0) * Matern(length_scale=[1.0]*3, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RBF(length_scale=[1.0]*3) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RationalQuadratic(alpha=0.1) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * (DotProduct() + Matern(nu=2.5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
    ]
    best_score = -np.inf
    best_model = None
    best_name = ""
    names = ["Matern(2.5)", "RBF", "RationalQuad", "Composite"]
    
    for k, name in zip(kernels, names):
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=False, random_state=30)
        scores = cross_val_score(gp, X, y, cv=5, scoring='r2')
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_model = gp
            best_name = name
            
    best_model.fit(X, y)
    try:
        print(f"      [Kernel fitted] {best_model.kernel_}")
    except Exception:
        pass
    return best_model, best_name

# =============================================================================
# If True, GA optimizes a *conservative* score using GP uncertainty:
#   maximize (mu_Q - k*std_Q) / (mu_dP + k*std_dP)
# If False, GA optimizes the mean ratio mu_Q / mu_dP
USE_CONSERVATIVE_FITNESS = True
K_SIGMA = 1.0  # 1.0 ~ 2.0 are common; larger => more conservative

DP_FLOOR = 1e-3  # avoid exploding ratio when dP is very small (Pa)

# =============================================================================
# 1. Train surrogate models (logic aligned with design_version_4 pipeline)
# =============================================================================
def load_and_train_model(file_path: str):
    print("1. [System] 데이터 로드 및 대리 모델 학습 중...")

    # Resolve path
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(file_path):
        if os.path.exists(file_path + ".xlsx"):
            file_path += ".xlsx"
        elif os.path.exists(file_path + ".csv"):
            file_path += ".csv"
        else:
            print(f"   [오류] 파일을 찾을 수 없습니다: {file_path}")
            return None, None, None

    print(f"   -> 파일 로드: {file_path}")

    try:
        # Header auto-detection for Excel
        if file_path.lower().endswith((".xlsx", ".xls")):
            try:
                df_raw = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
            except Exception:
                df_raw = pd.read_excel(file_path, header=None)

            header_row = 0
            for i in range(30):
                if i >= len(df_raw):
                    break
                row_str = " ".join([str(val).lower() for val in df_raw.iloc[i].values if pd.notna(val)])
                if ("s1" in row_str and ("height" in row_str or "spacing" in row_str)) or ("s1_mm" in row_str):
                    header_row = i
                    print(f"   -> 헤더 행 발견: {header_row}행")
                    break
            df = pd.read_excel(file_path, sheet_name="Sheet1", header=header_row)
        else:
            df = pd.read_csv(file_path)
            if "s1" not in [str(c).lower() for c in df.columns]:
                df = pd.read_csv(file_path, header=2)
    except Exception as e:
        print(f"   [오류] 파일 로드 실패: {e}")
        return None, None, None

    df.columns = [str(c).strip() for c in df.columns]
    print(f"   -> 컬럼: {df.columns.tolist()[:10]}...")

    def find_column(keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower()]
            if matches:
                return matches[0]
        return None

    col_map = {
        "s1": find_column(["s1_mm", "s1"]),
        "h": find_column(["fin_height", "fh_mm", "height"]),
        "s": find_column(["fin_spacing", "fs_mm", "spacing"]),
        "q": find_column(["q''", "flux", "heat flux"]),
        "dp": find_column(["delta p", "delta_p", "dp"]),
    }

    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        print(f"   [오류] 다음 컬럼을 찾을 수 없습니다: {missing}")
        print(f"   사용 가능한 컬럼: {df.columns.tolist()}")
        return None, None, None

    print(f"   -> 컬럼 매핑 완료: {col_map}")

    # Basic cleaning
    df = df.dropna(subset=[col_map["dp"], col_map["q"]])

    X = df[[col_map["s1"], col_map["h"], col_map["s"]]].values.astype(float)

    scaler_X = StandardScaler().fit(X)
    X_sc = scaler_X.transform(X)

    models = {}
    scalers_y = {}

    # Use automatic kernel selection (aligned with design_version_5)
    target_labels = {"q": "Q'' [W/m^2]", "dp": "Delta P [Pa]"}
    
    for target in ["q", "dp"]:
        print(f"   -> [{target_labels[target]}] 모델 최적화 중...")
        y = df[col_map[target]].values.astype(float)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        y_sc = scaler_y.transform(y.reshape(-1, 1)).ravel()

        # Automatic kernel selection with cross-validation
        gp, k_name = find_best_kernel(X_sc, y_sc)
        print(f"      Best Kernel: {k_name}")

        models[target] = gp
        scalers_y[target] = scaler_y

    print("   -> 대리 모델 학습 완료.")
    return models, scaler_X, scalers_y


# =============================================================================
# 2. GA optimization
# =============================================================================
global_models = None
global_scaler_X = None
global_scalers_y = None


def _inverse_mean_std(scaler_y: StandardScaler, mu_sc: np.ndarray, std_sc: np.ndarray):
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


def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness to maximize.

    Constraints:
      1) 45 <= s1 <= 200  (kept as user requested)
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

    # Model prediction
    try:
        x_sc = global_scaler_X.transform([solution])

        if USE_CONSERVATIVE_FITNESS:
            # IMPORTANT change (3): use predictive std to be conservative
            q_mu_sc, q_std_sc = global_models["q"].predict(x_sc, return_std=True)
            dp_mu_sc, dp_std_sc = global_models["dp"].predict(x_sc, return_std=True)

            q_mu, q_std = _inverse_mean_std(global_scalers_y["q"], q_mu_sc[0], q_std_sc[0])
            dp_mu, dp_std = _inverse_mean_std(global_scalers_y["dp"], dp_mu_sc[0], dp_std_sc[0])

            q_eff = q_mu - K_SIGMA * q_std
            dp_eff = dp_mu + K_SIGMA * dp_std

            # Safety floors / physical sanity
            if (dp_eff <= DP_FLOOR) or (not np.isfinite(dp_eff)) or (not np.isfinite(q_eff)):
                return -500.0
            if q_eff <= 0:
                return -500.0

            return q_eff / dp_eff

        else:
            q_sc = global_models["q"].predict(x_sc)
            dp_sc = global_models["dp"].predict(x_sc)

            pred_q = global_scalers_y["q"].inverse_transform(q_sc.reshape(-1, 1))[0][0]
            pred_dp = global_scalers_y["dp"].inverse_transform(dp_sc.reshape(-1, 1))[0][0]

            if pred_dp <= DP_FLOOR:
                return -500.0
            if pred_q <= 0:
                return -500.0

            return float(pred_q / pred_dp)

    except Exception:
        return -1000.0


def run_ga_optimization(file_path: str):
    global global_models, global_scaler_X, global_scalers_y

    # 1) Train surrogate
    models, scaler_X, scalers_y = load_and_train_model(file_path)
    if models is None:
        return

    global_models = models
    global_scaler_X = scaler_X
    global_scalers_y = scalers_y

    print("\n2. [System] GA(유전 알고리즘) 최적화 시작...")
    print(f"   - Fitness: {'Conservative (mu±k·sigma)' if USE_CONSERVATIVE_FITNESS else 'Mean (mu only)'}")
    if USE_CONSERVATIVE_FITNESS:
        print(f"   - k (sigma multiplier): {K_SIGMA}")
    print("   - 세대 수(Generations): 100")
    print("   - 인구 수(Population): 50")
    print("   - 탐색 전략: Q''/dP 최대화 (제약조건 준수)")

    # 2) GA params
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=3,
        gene_space=[
            {"low": 45, "high": 200},  # S1  (kept as user requested)
            {"low": 6, "high": 40},    # FH  (upper bound enforced by constraint)
            {"low": 2, "high": 8},     # FS
        ],
        parent_selection_type="rank",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        random_seed=30,
    )

    # 3) Run GA
    ga_instance.run()

    # 4) Best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    opt_s1, opt_fh, opt_fs = solution
    fh_limit = 0.5 * ((opt_s1 / np.sqrt(2)) - 24.0) - 0.4

    # Report both mean and conservative predictions at the optimum
    x_sc = scaler_X.transform([solution])

    q_mu_sc, q_std_sc = models["q"].predict(x_sc, return_std=True)
    dp_mu_sc, dp_std_sc = models["dp"].predict(x_sc, return_std=True)

    q_mu, q_std = _inverse_mean_std(scalers_y["q"], q_mu_sc[0], q_std_sc[0])
    dp_mu, dp_std = _inverse_mean_std(scalers_y["dp"], dp_mu_sc[0], dp_std_sc[0])

    q_eff = q_mu - K_SIGMA * q_std
    dp_eff = dp_mu + K_SIGMA * dp_std

    mean_ratio = q_mu / max(dp_mu, DP_FLOOR)
    cons_ratio = q_eff / max(dp_eff, DP_FLOOR)

    print("\n" + "=" * 70)
    print("       🧬 GA(Genetic Algorithm) 최적 설계 결과 (Surrogate-based)")
    print("=" * 70)
    print("1. 최적 설계변수 (Optimal Design Variables):")
    print(f"   - S1 (mm)          : {opt_s1:.4f}")
    print(f"   - Fin Height (mm)  : {opt_fh:.4f}  (Limit: {fh_limit:.4f})")
    print(f"   - Fin Spacing (mm) : {opt_fs:.4f}")

    print("\n2. Surrogate 예측 (Mean ± Std):")
    print(f"   - Q''  : {q_mu:.2f} ± {q_std:.2f}  (W/m^2)")
    print(f"   - ΔP   : {dp_mu:.4f} ± {dp_std:.4f}  (Pa)")

    print("\n3. 목적함수 값:")
    print(f"   - Mean ratio        (mu_Q / mu_ΔP)                  : {mean_ratio:.6f}")
    print(f"   - Conservative ratio ((mu_Q-kσ_Q)/(mu_ΔP+kσ_ΔP))    : {cons_ratio:.6f}")
    print(f"   - GA가 최대화한 fitness 값                           : {solution_fitness:.6f}")

    print("\n4. 제약조건 검증:")
    is_valid_fh = opt_fh <= (fh_limit + 1e-3)
    is_valid_basic = (45 <= opt_s1 <= 200) and (2 <= opt_fs <= 8) and (opt_fh >= 6)
    print(f"   - Height 제약 만족?  {'[PASS]' if is_valid_fh else '[FAIL]'}")
    print(f"   - 기본 범위 만족?    {'[PASS]' if is_valid_basic else '[FAIL]'}")

    # Optional: ga_instance.plot_fitness()


if __name__ == "__main__":
    file_path = "../../data/total_2D_Data.xlsx"
    run_ga_optimization(file_path)
