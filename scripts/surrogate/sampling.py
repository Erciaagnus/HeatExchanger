import math
import time
import os
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import qmc
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional

N_SAMPLES = 1000000        # 생성할 샘플 개수 (예: 100000, 1000000 등)
OUTPUT_FILENAME = "LHS_Porous_Params_Result.csv" # 저장할 파일 이름
RANDOM_SEED = 2025         # 랜덤 시드 (결과 재현용)

# 물리 해석 설정 (기본값 유지 권장)
N_POINTS_FITTING = 25      # Curve Fitting에 사용할 속도 포인트 개수 (많을수록 느려짐, 15~50 추천)
AMB_TEMP_C = 14.8          # 외기 온도 [°C]
DESIGN_VELOCITY = 2.019    # 설계 유속 [m/s]
TUBE_OD_MM = 24.0          # 튜브 외경 [mm]
FIN_THICK_MM = 0.5         # 핀 두께 [mm]

# =============================================================================
# 1. LHS 샘플링 로직 (S1, Height, Spacing 생성)
# =============================================================================
def generate_lhs_samples(n: int, seed: int) -> pd.DataFrame:
    print(f"1. LHS 샘플링 시작 (N={n})...")
    
    # 설계 변수 범위 및 제약조건 (mm)
    s_min, s_max = 45.0, 200.0
    fh_min = 6.0
    fs_min, fs_max = 2.0, 8.0
    td = TUBE_OD_MM
    margin_outside = 0.4

    # fh가 존재할 수 있는 유효 S 최소값 계산
    # fh_max(s) = 0.5*(s/sqrt(2) - td) - margin >= fh_min
    s_min_feasible = max(s_min, (2 * (fh_min + margin_outside) + td) * sqrt(2))

    if s_min_feasible >= s_max:
        raise ValueError("설정된 제약조건하에서 유효한 S범위가 없습니다.")

    # LHS 샘플러 (3차원: S, fh, fs)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    U = sampler.random(n=n)
    u_s, u_fh, u_fs = U[:, 0], U[:, 1], U[:, 2]

    # 1) S1 (Pitch) 샘플링
    s = s_min_feasible + u_s * (s_max - s_min_feasible)
    
    # 2) Fin Spacing 샘플링
    fs = fs_min + u_fs * (fs_max - fs_min)

    # 3) Fin Height 샘플링 (S값에 따른 동적 상한 적용)
    # Constraint: fh <= 0.5*(s/sqrt(2) - td) - 0.4
    fh_upper = 0.5 * (s / sqrt(2) - td) - margin_outside
    span = fh_upper - fh_min
    
    # 안전장치: 혹시라도 span이 음수면 최소값으로 고정 (위에서 s_min_feasible로 방지함)
    span = np.maximum(span, 0.0)
    
    fh = fh_min + u_fh * span

    df = pd.DataFrame({
        "S1_mm": s,
        "fin_height_fh_mm": fh,
        "fin_spacing_fs_mm": fs,
        "fh_upper_mm": fh_upper,
        "constraint_ok": True  # 생성 로직상 무조건 True
    })
    
    print(f"   -> 샘플 생성 완료. (S1: {df['S1_mm'].min():.1f}~{df['S1_mm'].max():.1f} mm)")
    return df

# =============================================================================
# 2. 물리/형상 해석 클래스 및 함수 (Porous Parameter Calc)
# =============================================================================
def air_properties(T_celsius: float) -> Dict[str, float]:
    T_K = T_celsius + 273.15
    P = 101325.0
    R = 287.05
    rho = P / (R * T_K)
    mu = (1.716e-5 * (T_K / 273.15) ** 1.5) * (384.15) / (T_K + 111.0)
    return {"rho": rho, "mu": mu}

@dataclass
class AnnularFinGeometry:
    Dc: float; delta_f: float; s1: float; Fs: float; hf: float; N: int = 4
    
    def __post_init__(self):
        self.Do = self.Dc + 2.0 * self.hf
        self.Fp = self.Fs + self.delta_f
        self.epsilon = 1.0 - (self.delta_f / self.Fp)
        self.sigma = (self.s1 - self.Dc - 2.0 * self.hf * (self.delta_f / self.Fp)) / self.s1
        
        # Areas
        self.A_fin = (2.0 * (math.pi / 4.0) * (self.Do**2 - self.Dc**2) + math.pi * self.Do * self.delta_f)
        self.A_base = math.pi * self.Dc * self.Fs
        self.A_total = self.A_fin + self.A_base
        self.A_bare = math.pi * self.Dc * self.Fp
        self.area_ratio = self.A_total / self.A_bare
        self.V_annular = (math.pi / 4.0) * (self.Do**2 - self.Dc**2) * self.Fp
        self.a_fs = self.A_total / self.V_annular
        self.s2 = self.s1  # Pitch ratio 1.0 가정

def calculate_row_physics(row, air_props, v_range, n_points):
    """한 행(Row)에 대한 물리 연산 수행"""
    try:
        # mm -> m 변환
        Dc = TUBE_OD_MM / 1000.0
        delta_f = FIN_THICK_MM / 1000.0
        s1 = row['S1_mm'] / 1000.0
        hf = row['fin_height_fh_mm'] / 1000.0
        Fs = row['fin_spacing_fs_mm'] / 1000.0
        
        geom = AnnularFinGeometry(Dc=Dc, delta_f=delta_f, s1=s1, Fs=Fs, hf=hf)
        
        # Darcy-Forchheimer Fitting (Velocity sweep)
        v_points = np.linspace(v_range[0], v_range[1], n_points)
        Y_points = []
        
        rho, mu = air_props['rho'], air_props['mu']
        
        for v in v_points:
            # Nir Correlation
            v_max = v / geom.sigma
            Re_Dc = (rho * v_max * geom.Dc) / mu
            fN = 1.1 * (Re_Dc**-0.25) * ((geom.s1/geom.Dc)**0.6) * (geom.area_ratio**0.15)
            dP_total = 4 * fN * (rho * v_max**2) / 2.0  # N=4 rows
            L = geom.s2 * 4
            Y_points.append(dP_total / L) # dP/L
            
        # Least Squares Fit (Y = A*v + B*v^2)
        X_mat = np.column_stack([v_points, v_points**2])
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, Y_points, rcond=None)
        A, B = coeffs[0], coeffs[1]
        
        inv_K = A / mu
        K = mu / A if A != 0 else np.inf
        C2 = 2.0 * B / rho
        
        # R2 score calculation
        Y_fit = A * v_points + B * v_points**2
        SS_tot = np.sum((Y_points - np.mean(Y_points))**2)
        SS_res = np.sum((Y_points - Y_fit)**2)
        R2 = 1 - (SS_res/SS_tot) if SS_tot > 0 else np.nan

        # Design Point Calculation (at v_design)
        v_des = DESIGN_VELOCITY
        v_max_des = v_des / geom.sigma
        Re_Dc_des = (rho * v_max_des * geom.Dc) / mu
        fN_des = 1.1 * (Re_Dc_des**-0.25) * ((geom.s1/geom.Dc)**0.6) * (geom.area_ratio**0.15)
        dP_tot_des = 4 * fN_des * (rho * v_max_des**2) / 2.0
        
        # Briggs & Young h calculation
        k_air = 0.0241 + 7.0e-5 * AMB_TEMP_C # Approx
        Pr = 0.71 # Approx
        Nu = 0.134 * (Re_Dc_des**0.681) * (Pr**(1/3)) * ((geom.Fs/geom.hf)**0.2) * ((geom.Fs/geom.delta_f)**0.113)
        h_fs = Nu * k_air / geom.Dc

        return pd.Series({
            'Viscous_Resistance_1_m2': inv_K,
            'Inertial_Resistance_1_m': C2,
            'K_m2': K,
            'R2_fit': R2,
            'Porosity': geom.epsilon,
            'a_fs_1_m': geom.a_fs,
            'area_ratio': geom.area_ratio,
            'sigma': geom.sigma,
            'porous_thickness_m': hf,
            'flow_depth_m': geom.s2 * 4,
            'Re_Dc': Re_Dc_des,
            'dP_total_Pa': dP_tot_des,
            'dP_per_L_Pa_m': dP_tot_des / (geom.s2 * 4),
            'h_fs_W_m2K': h_fs,
            'ok': True,
            'error': None
        })
        
    except Exception as e:
        return pd.Series({'ok': False, 'error': str(e)})

# =============================================================================
# 3. 메인 실행 함수
# =============================================================================
def main():
    start_time = time.time()
    
    # 1. 샘플 생성
    df = generate_lhs_samples(N_SAMPLES, RANDOM_SEED)
    
    print(f"2. 물성치 계산 시작 (총 {N_SAMPLES}개, Fitting Points={N_POINTS_FITTING})...")
    print("   (샘플이 많으면 시간이 다소 소요됩니다)")
    
    air_props = air_properties(AMB_TEMP_C)
    
    # Apply calculation to all rows
    # (속도를 위해 Pandas apply 대신 벡터화할 수도 있으나, 로직 복잡성을 고려해 apply 사용)
    # 진행상황 표시를 위해 청크 단위로 처리하거나 tqdm 사용 권장되나 여기서는 기본 apply 사용
    
    # 대량 데이터 처리를 위한 팁: 100만개면 여기서 시간이 꽤 걸립니다.
    # 단순화를 위해 apply를 쓰지만, 실제로는 swifter나 병렬처리가 필요할 수 있습니다.
    
    results = df.apply(
        lambda row: calculate_row_physics(row, air_props, (0.5, 3.5), N_POINTS_FITTING), 
        axis=1
    )
    
    # 결과 병합
    final_df = pd.concat([df, results], axis=1)
    
    # 샘플 넘버링 추가 (1부터 시작)
    final_df.insert(0, 'Sample_No', range(1, len(final_df) + 1))
    
    # CSV 저장 (HeatExchanger/data/에 저장)
    this_file = os.path.abspath(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(this_file), "..", "..", ".."))  # HeatExchanger
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    output_path = os.path.join(DATA_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False)

    print(f"\n[완료] 결과 저장됨: {output_path}")
    
    elapsed = time.time() - start_time
    print(f"\n[완료] 결과 저장됨: {output_path}")
    print(f"총 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")
    print(f"데이터 미리보기:\n{final_df.head(3)}")

if __name__ == "__main__":
    main()