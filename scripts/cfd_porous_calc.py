#!/usr/bin/env python3
"""
================================================================================
Annular Fin → Porous Media 파라미터 변환 계산기
================================================================================
Nir (1991) 마찰계수 상관식을 사용하여 CFD Porous Media의
점성 저항 계수(1/K)와 관성 저항 계수(C2)를 도출합니다.

다중점 피팅 방식: 0.5~3.5 m/s 범위에서 여러 속도점을 계산하여
최소제곱법으로 Darcy-Forchheimer 계수를 정확하게 근사합니다.

Reference:
    Nir, A. (1991). "Heat Transfer and Friction Factor Correlations for
    Crossflow over Staggered Finned Tube Banks", Heat Transfer Engineering,
    Vol.12, No.1, pp.43-58.

Author: Claude (Anthropic)
Date: 2026-01-17
================================================================================
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import argparse
import sys

# matplotlib 임포트 (시각화용)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이도 작동
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Visualization will be disabled.")

# =============================================================================
# 1. 물성치 계산 함수
# =============================================================================

def air_properties(T_celsius: float) -> Dict[str, float]:
    """
    주어진 온도에서 공기의 물성치를 계산합니다.

    Parameters:
        T_celsius: 온도 [°C]

    Returns:
        dict: 공기 물성치
            - rho: 밀도 [kg/m³]
            - mu: 점성계수 [Pa·s]
            - k: 열전도도 [W/(m·K)]
            - Pr: Prandtl 수 [-]
            - nu: 동점성계수 [m²/s]
    """
    T_K = T_celsius + 273.15
    P = 101325  # 대기압 [Pa]
    R = 287.058  # 공기 기체상수 [J/(kg·K)]

    # 밀도 (이상기체 상태방정식)
    rho = P / (R * T_K)

    # 점성계수 (Sutherland's Law)
    # μ = μ_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
    mu_ref = 1.716e-5  # [Pa·s] at 273.15K
    T_ref = 273.15
    S = 110.4  # Sutherland 상수 [K]
    mu = mu_ref * (T_K / T_ref)**1.5 * (T_ref + S) / (T_K + S)

    # 열전도도 (근사식)
    k = 0.0241 + 7.0e-5 * T_celsius  # 선형 근사

    # Prandtl 수
    cp = 1006  # 정압비열 [J/(kg·K)]
    Pr = mu * cp / k

    # 동점성계수
    nu = mu / rho

    return {
        'rho': rho,
        'mu': mu,
        'k': k,
        'Pr': Pr,
        'nu': nu,
        'T_K': T_K
    }


# =============================================================================
# 2. 기하학적 파라미터 계산
# =============================================================================

@dataclass
class AnnularFinGeometry:
    """Annular Fin 기하학적 파라미터를 저장하는 데이터 클래스"""
    # 입력 파라미터
    Dc: float      # 튜브 외경 [m]
    delta_f: float # 핀 두께 [m]
    s1: float      # 횡방향 피치 (Spanwise) [m]
    Fs: float      # 핀 간격 (Fin Spacing) [m]
    hf: float      # 핀 높이 [m]
    pitch_ratio: float = 1.0  # s1/s2 비율
    N: int = 4     # 튜브 열 수

    def __post_init__(self):
        """종속 파라미터 계산"""
        # 핀 외경
        self.Do = self.Dc + 2 * self.hf

        # 핀 피치 (Center-to-Center)
        self.Fp = self.Fs + self.delta_f

        # 종방향 피치
        self.s2 = self.s1 / self.pitch_ratio

        # 공극률 (Annular Zone 기준)
        self.epsilon = 1 - (self.delta_f / self.Fp)

        # 최소 자유유동 면적비 (sigma)
        # σ = (S1 - Dc - 2*hf*(δf/Fp)) / S1
        self.sigma = (self.s1 - self.Dc - 2*self.hf*(self.delta_f/self.Fp)) / self.s1

        # 면적 계산 (1 피치 구간)
        self._calculate_areas()

    def _calculate_areas(self):
        """1 피치 구간의 면적 계산"""
        # 핀 표면적 (양면 + 팁)
        # A_fin = 2 × π/4 × (Do² - Dc²) + π × Do × δf
        self.A_fin = (2 * (math.pi/4) * (self.Do**2 - self.Dc**2) +
                      math.pi * self.Do * self.delta_f)

        # 튜브 노출 면적 (핀 사이)
        self.A_base = math.pi * self.Dc * self.Fs

        # 총 표면적
        self.A_total = self.A_fin + self.A_base

        # 맨 튜브 표면적 (핀이 없을 때)
        self.A_bare = math.pi * self.Dc * self.Fp

        # 면적비 (Area Ratio) - Nir 상관식의 핵심 인자
        self.area_ratio = self.A_total / self.A_bare

        # Annular Zone 체적 (1 피치 구간)
        self.V_annular = (math.pi/4) * (self.Do**2 - self.Dc**2) * self.Fp

        # 비표면적 [1/m]
        self.a_fs = self.A_total / self.V_annular

    def get_porous_thickness(self) -> float:
        """Porous Zone의 두께 (핀 높이) 반환"""
        return self.hf

    def get_flow_depth(self) -> float:
        """유동 방향 총 깊이 반환"""
        return self.s2 * self.N


# =============================================================================
# 3. Nir (1991) 마찰계수 상관식
# =============================================================================

def nir_friction_factor(Re_Dc: float, s1_Dc: float, area_ratio: float) -> float:
    """
    Nir (1991) 마찰계수 상관식 -> Fitting/Changed Equation

    f_N = 1.1 × Re^(-0.25) × (S1/Dc)^(0.6) × (A_tot/A_bare)^(0.15)

    Parameters:
        Re_Dc: 튜브 외경 기준 Reynolds 수 (v_max 기반)
        s1_Dc: 횡방향 피치 / 튜브 외경 [-]
        area_ratio: 면적비 (A_total / A_bare) [-]

    Returns:
        f_N: Nir 마찰계수 [-]

    적용 범위:
        - Re_h = 300 ~ 10,000 (수력직경 기준)
        - Staggered 배열
    """
    term1 = 1.1 * (Re_Dc ** -0.25)
    term2 = (s1_Dc) ** 0.6
    term3 = (area_ratio) ** 0.15

    f_N = term1 * term2 * term3
    return f_N


def calculate_pressure_drop(v_inlet: float, geom: AnnularFinGeometry,
                           air_props: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Nir 상관식을 이용한 압력강하 계산

    ΔP = N × f_N × (ρ × v_max²) / 2

    Parameters:
        v_inlet: 입구 속도 (superficial velocity) [m/s]
        geom: AnnularFinGeometry 객체
        air_props: 공기 물성치 dict

    Returns:
        tuple: (ΔP_total [Pa], ΔP/L [Pa/m], Re_Dc [-])
    """
    rho = air_props['rho']
    mu = air_props['mu']

    # 최대 속도 (최소 단면적 기준)
    v_max = v_inlet / geom.sigma

    # Reynolds 수 (튜브 외경 기준, v_max 사용)
    Re_Dc = (rho * v_max * geom.Dc) / mu

    # Nir 마찰계수
    s1_Dc = geom.s1 / geom.Dc
    f_N = nir_friction_factor(Re_Dc, s1_Dc, geom.area_ratio)

    # 총 압력강하
    dP_total = geom.N * f_N * (rho * v_max**2) / 2

    # 단위 길이당 압력강하
    L = geom.get_flow_depth()
    dP_per_L = dP_total / L

    return dP_total, dP_per_L, Re_Dc


# =============================================================================
# 4. Porous Parameter Fittings
# =============================================================================

def fit_darcy_forchheimer_multipoint(
    v_range: Tuple[float, float],
    n_points: int,
    geom: AnnularFinGeometry,
    air_props: Dict[str, float]
) -> Dict[str, float]:

    rho = air_props['rho']
    mu = air_props['mu']

    v_min, v_max = v_range

    # 속도 범위 생성
    v_points = np.linspace(v_min, v_max, n_points)

    # 각 속도점에서 압력강하 계산
    Y_points = []
    Re_points = []

    for v in v_points:
        _, Y, Re = calculate_pressure_drop(v, geom, air_props)
        Y_points.append(Y)
        Re_points.append(Re)

    v_points = np.array(v_points)
    Y_points = np.array(Y_points)
    Re_points = np.array(Re_points)

    # 최소제곱법 피팅: Y = A*v + B*v²
    # 선형 시스템으로 변환: Y = [v, v²] · [A, B]^T
    # 행렬 형태: [v1, v1²]   [A]   [Y1]
    #           [v2, v2²] × [B] = [Y2]
    #           [  ...  ]         [ ...]

    # Design matrix
    X_matrix = np.column_stack([v_points, v_points**2])

    # 최소제곱법 해: (X^T X)^(-1) X^T Y
    coeffs, residuals, rank, s = np.linalg.lstsq(X_matrix, Y_points, rcond=None)

    A = coeffs[0]  # 선형 계수
    B = coeffs[1]  # 2차 계수

    # Porous 파라미터 변환
    # ΔP/L = A×v + B×v² = (μ/K)×v + (C2×ρ/2)×v²

    inv_K = A / mu        # 점성 저항 계수 [1/m²]
    K = mu / A            # 투과도 [m²]
    C2 = 2 * B / rho      # 관성 저항 계수 [1/m]

    # R² (결정계수) 계산
    Y_fit = A * v_points + B * v_points**2
    SS_res = np.sum((Y_points - Y_fit)**2)
    SS_tot = np.sum((Y_points - np.mean(Y_points))**2)
    R_squared = 1 - (SS_res / SS_tot)

    return {
        'A': A,
        'B': B,
        'inv_K': inv_K,      # Viscous Resistance (Fluent: 1/α)
        'C2': C2,            # Inertial Resistance (Fluent: C2)
        'K': K,              # Permeability
        'R_squared': R_squared,  # 피팅 품질
        'residual_sum': residuals[0] if len(residuals) > 0 else 0,
        # 피팅 데이터
        'v_points': v_points,
        'Y_points': Y_points,
        'Y_fit': Y_fit,
        'Re_points': Re_points,
        'Re_min': Re_points.min(),
        'Re_max': Re_points.max()
    }


# =============================================================================
# 5. 열전달 계수 (Briggs & Young 상관식)
# =============================================================================

def briggs_young_heat_transfer(v_inlet: float, geom: AnnularFinGeometry,
                                air_props: Dict[str, float]) -> float:
    """
    Briggs & Young (1963) 열전달 상관식

    Nu = 0.134 × Re^0.681 × Pr^(1/3) × (Fs/hf)^0.2 × (Fs/δf)^0.113

    Parameters:
        v_inlet: 입구 속도 [m/s]
        geom: AnnularFinGeometry 객체
        air_props: 공기 물성치 dict

    Returns:
        h_fs: 열전달 계수 [W/(m²·K)]
    """
    rho = air_props['rho']
    mu = air_props['mu']
    k = air_props['k']
    Pr = air_props['Pr']

    v_max = v_inlet / geom.sigma
    Re_Dc = (rho * v_max * geom.Dc) / mu

    Nu = (0.134 * (Re_Dc ** 0.681) * (Pr ** (1/3)) *
          ((geom.Fs / geom.hf) ** 0.2) *
          ((geom.Fs / geom.delta_f) ** 0.113))

    h_fs = Nu * k / geom.Dc

    return h_fs


# =============================================================================
# 6. 메인 계산 함수
# =============================================================================

def calculate_porous_parameters(
    Fs_mm: float,           # 핀 간격 [mm]
    hf_mm: float,           # 핀 높이 [mm]
    T_celsius: float,       # 대기 온도 [°C]
    v_design: float,        # 설계 풍속 [m/s]
    Dc_mm: float = 24.0,    # 튜브 외경 [mm]
    delta_f_mm: float = 0.5,# 핀 두께 [mm]
    s1_mm: float = 55.333,  # 횡방향 피치 [mm]
    pitch_ratio: float = 1.0,# s1/s2 비율
    N: int = 4,             # 튜브 열 수
    v_range: Tuple[float, float] = (0.5, 3.5),  # 피팅 속도 범위
    n_points: int = 50      # 피팅 점 개수
) -> Dict:
    """
    Annular Fin을 Porous Media로 근사하기 위한 모든 파라미터 계산

    Parameters:
        Fs_mm: 핀 간격 [mm]
        hf_mm: 핀 높이 [mm]
        T_celsius: 대기 온도 [°C]
        v_design: 설계 풍속 [m/s]
        Dc_mm: 튜브 외경 [mm] (기본값: 24)
        delta_f_mm: 핀 두께 [mm] (기본값: 0.5)
        s1_mm: 횡방향 피치 [mm] (기본값: 55.333)
        pitch_ratio: s1/s2 비율 (기본값: 1.0)
        N: 튜브 열 수 (기본값: 4)
        v_range: 피팅 속도 범위 [m/s] (기본값: 0.5~3.5)
        n_points: 피팅 점 개수 (기본값: 50)

    Returns:
        dict: 모든 계산 결과
    """
    # 단위 변환 (mm → m)
    Dc = Dc_mm / 1000
    delta_f = delta_f_mm / 1000
    s1 = s1_mm / 1000
    Fs = Fs_mm / 1000
    hf = hf_mm / 1000

    # 공기 물성치
    air_props = air_properties(T_celsius)

    # 기하학적 파라미터
    geom = AnnularFinGeometry(
        Dc=Dc, delta_f=delta_f, s1=s1, Fs=Fs, hf=hf,
        pitch_ratio=pitch_ratio, N=N
    )

    # 다중점 피팅
    porous_params = fit_darcy_forchheimer_multipoint(
        v_range, n_points, geom, air_props
    )

    # 설계점에서의 압력강하 및 Reynolds 수
    dP_total, dP_per_L, Re_design = calculate_pressure_drop(v_design, geom, air_props)

    # 열전달 계수
    h_fs = briggs_young_heat_transfer(v_design, geom, air_props)

    return {
        # 입력 조건
        'input': {
            'Fs_mm': Fs_mm,
            'hf_mm': hf_mm,
            'T_celsius': T_celsius,
            'v_design': v_design,
            'Dc_mm': Dc_mm,
            'delta_f_mm': delta_f_mm,
            's1_mm': s1_mm,
            'pitch_ratio': pitch_ratio,
            'N': N,
            'v_range': v_range,
            'n_points': n_points
        },
        # 공기 물성치
        'air': air_props,
        # 기하학적 파라미터
        'geometry': {
            'Do_mm': geom.Do * 1000,
            'Fp_mm': geom.Fp * 1000,
            's2_mm': geom.s2 * 1000,
            'epsilon': geom.epsilon,
            'sigma': geom.sigma,
            'area_ratio': geom.area_ratio,
            'a_fs': geom.a_fs,
            'porous_thickness_mm': geom.get_porous_thickness() * 1000,
            'flow_depth_mm': geom.get_flow_depth() * 1000
        },
        # Porous 파라미터 (핵심 출력)
        'porous': {
            'inv_K': porous_params['inv_K'],  # [1/m²]
            'C2': porous_params['C2'],         # [1/m]
            'K': porous_params['K'],           # [m²]
            'A': porous_params['A'],           # [Pa·s/m²]
            'B': porous_params['B'],           # [Pa·s²/m³]
            'R_squared': porous_params['R_squared']
        },
        # 설계점 결과
        'design_point': {
            'Re_Dc': Re_design,
            'dP_total_Pa': dP_total,
            'dP_per_L_Pa_m': dP_per_L,
            'h_fs_W_m2K': h_fs
        },
        # 피팅 정보
        'fitting': {
            'v_points': porous_params['v_points'],
            'Y_points': porous_params['Y_points'],
            'Y_fit': porous_params['Y_fit'],
            'Re_points': porous_params['Re_points'],
            'Re_min': porous_params['Re_min'],
            'Re_max': porous_params['Re_max']
        }
    }


# =============================================================================
# 7. 시각화 함수
# =============================================================================

def plot_fitting_results(result: Dict, save_path: str = None):
    """
    피팅 결과를 시각화

    Parameters:
        result: calculate_porous_parameters의 반환값
        save_path: 이미지 저장 경로 (None이면 자동 생성)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return

    # 파일명 자동 생성 (설계변수 기반)
    if save_path is None:
        inp = result['input']
        save_path = f"Fs{inp['Fs_mm']:.1f}_hf{inp['hf_mm']:.1f}_T{inp['T_celsius']:.1f}_v{inp['v_design']:.2f}.png"

    v_points = result['fitting']['v_points']
    Y_points = result['fitting']['Y_points']
    Y_fit = result['fitting']['Y_fit']
    R_squared = result['porous']['R_squared']
    inv_K = result['porous']['inv_K']
    C2 = result['porous']['C2']
    K = result['porous']['K']

    # Figure 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 왼쪽: 압력강하 vs 속도
    ax1.scatter(v_points, Y_points, color='blue', s=50, alpha=0.6,
                label='Nir Correlation (Calculated)', zorder=3)
    ax1.plot(v_points, Y_fit, 'r-', linewidth=2,
             label=f'Darcy-Forchheimer Fit (R²={R_squared:.6f})', zorder=2)
    ax1.set_xlabel('Velocity [m/s]', fontsize=13)
    ax1.set_ylabel('Pressure Drop per Length [Pa/m]', fontsize=13)
    ax1.set_title('Pressure Drop Fitting', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')

    # CFD 파라미터 텍스트 박스 추가
    textstr = '\n'.join((
        r'$\bf{CFD\ Porous\ Parameters:}$',
        f'1/K = {inv_K:.4e} [1/m²]',
        f'C₂ = {C2:.4f} [1/m]',
        f'K = {K:.4e} [m²]'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # 오른쪽: 잔차 플롯
    residuals = Y_points - Y_fit
    ax2.scatter(v_points, residuals, color='green', s=50, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Velocity [m/s]', fontsize=13)
    ax2.set_ylabel('Residuals [Pa/m]', fontsize=13)
    ax2.set_title('Fitting Residuals', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 잔차 통계 추가
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    textstr2 = '\n'.join((
        r'$\bf{Residual\ Statistics:}$',
        f'Mean = {residual_mean:.4f} Pa/m',
        f'Std Dev = {residual_std:.4f} Pa/m'
    ))
    props2 = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.98, 0.98, textstr2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {save_path}")


# =============================================================================
# 8. 텍스트 파일 저장 함수
# =============================================================================

def save_results_to_txt(result: Dict, save_path: str = None):
    """
    계산 결과를 텍스트 파일로 저장

    Parameters:
        result: calculate_porous_parameters의 반환값
        save_path: 텍스트 파일 저장 경로 (None이면 자동 생성)
    """
    # 파일명 자동 생성 (설계변수 기반, PNG와 동일한 이름)
    if save_path is None:
        inp = result['input']
        save_path = f"Fs{inp['Fs_mm']:.1f}_hf{inp['hf_mm']:.1f}_T{inp['T_celsius']:.1f}_v{inp['v_design']:.2f}.txt"

    inp = result['input']
    air = result['air']
    geom = result['geometry']
    porous = result['porous']
    design = result['design_point']
    fit = result['fitting']

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("       Annular Fin → Porous Media Parameter Conversion Results\n")
        f.write("       Nir (1991) Correlation with Multi-Point Least Squares Fitting\n")
        f.write("=" * 80 + "\n\n")

        f.write("[INPUT CONDITIONS]\n")
        f.write(f"  Fin Spacing (Fs)        : {inp['Fs_mm']:.1f} mm\n")
        f.write(f"  Fin Height (hf)         : {inp['hf_mm']:.1f} mm\n")
        f.write(f"  Air Temperature         : {inp['T_celsius']:.2f} °C\n")
        f.write(f"  Design Velocity         : {inp['v_design']:.4f} m/s\n")
        f.write(f"  Tube Outer Diameter (Dc): {inp['Dc_mm']:.1f} mm\n")
        f.write(f"  Transverse Pitch (s1)   : {inp['s1_mm']:.3f} mm\n")
        f.write(f"  Number of Tube Rows (N) : {inp['N']}\n")
        f.write(f"  Fin Thickness (δf)      : {inp['delta_f_mm']:.2f} mm\n")
        f.write(f"  Pitch Ratio (s1/s2)     : {inp['pitch_ratio']:.2f}\n\n")

        f.write("[AIR PROPERTIES @ {:.2f}°C]\n".format(inp['T_celsius']))
        f.write(f"  Density (ρ)             : {air['rho']:.4f} kg/m³\n")
        f.write(f"  Viscosity (μ)           : {air['mu']:.4e} Pa·s\n")
        f.write(f"  Thermal Conductivity (k): {air['k']:.4f} W/(m·K)\n")
        f.write(f"  Prandtl Number (Pr)     : {air['Pr']:.3f}\n\n")

        f.write("[GEOMETRIC PARAMETERS]\n")
        f.write(f"  Fin Outer Diameter (Do) : {geom['Do_mm']:.1f} mm\n")
        f.write(f"  Fin Pitch (Fp)          : {geom['Fp_mm']:.2f} mm\n")
        f.write(f"  Longitudinal Pitch (s2) : {geom['s2_mm']:.3f} mm\n")
        f.write(f"  Porosity (ε)            : {geom['epsilon']:.4f}\n")
        f.write(f"  Min. Flow Area Ratio (σ): {geom['sigma']:.4f}\n")
        f.write(f"  Area Ratio (Atot/Abare) : {geom['area_ratio']:.3f}\n")
        f.write(f"  Specific Surface (a_fs) : {geom['a_fs']:.2f} 1/m\n\n")

        f.write("=" * 80 + "\n")
        f.write("  ★★★ CFD POROUS MEDIA INPUT PARAMETERS (KEY OUTPUT) ★★★\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Viscous Resistance (1/K): {porous['inv_K']:.4e}  [1/m²]\n")
        f.write(f"  Inertial Resistance (C2): {porous['C2']:.4f}  [1/m]\n")
        f.write(f"  Permeability (K)        : {porous['K']:.4e}  [m²]\n")
        f.write(f"  Porosity (ε)            : {geom['epsilon']:.4f}  [-]\n")
        f.write(f"  Specific Surface (a_fs) : {geom['a_fs']:.2f}  [1/m]\n")
        f.write("=" * 80 + "\n\n")

        f.write("[DESIGN POINT PERFORMANCE]\n")
        f.write(f"  Reynolds Number (Re_Dc) : {design['Re_Dc']:.1f}\n")
        f.write(f"  Total Pressure Drop     : {design['dP_total_Pa']:.2f} Pa\n")
        f.write(f"  Pressure Drop per Length: {design['dP_per_L_Pa_m']:.2f} Pa/m\n")
        f.write(f"  Heat Transfer Coef (hfs): {design['h_fs_W_m2K']:.2f} W/(m²·K)\n\n")

        f.write("[FITTING INFORMATION]\n")
        f.write(f"  Velocity Range          : {inp['v_range'][0]:.1f} ~ {inp['v_range'][1]:.1f} m/s\n")
        f.write(f"  Number of Fitting Points: {inp['n_points']}\n")
        f.write(f"  Reynolds Number Range   : {fit['Re_min']:.1f} ~ {fit['Re_max']:.1f}\n")
        f.write(f"  Coefficient of Determination (R²): {porous['R_squared']:.8f}\n")
        f.write(f"  Darcy-Forchheimer Coef A: {porous['A']:.4e} [Pa·s/m²]\n")
        f.write(f"  Darcy-Forchheimer Coef B: {porous['B']:.4e} [Pa·s²/m³]\n\n")

        f.write("[KEY EQUATIONS]\n")
        f.write("  Nir (1991) Friction Factor:\n")
        f.write("    f_N = 1.1 × Re^(-0.25) × (S1/Dc)^(-0.4) × (Atot/Abare)^(0.15)\n\n")
        f.write("  Pressure Drop:\n")
        f.write("    ΔP = N × f_N × (ρ × v_max²) / 2\n\n")
        f.write("  Darcy-Forchheimer Equation:\n")
        f.write("    ΔP/L = (μ/K)·v + (C2·ρ/2)·v²\n")
        f.write("    ΔP/L = A·v + B·v²\n\n")
        f.write("  Multi-Point Least Squares Fitting:\n")
        f.write("    Minimize: Σ(Yi - A·vi - B·vi²)²\n")
        f.write(f"    Solution: Normal Equations (X^T X)[A B]^T = X^T Y\n\n")

        f.write("=" * 80 + "\n")
        f.write("Generated by: Nir (1991) Porous Parameter Calculator\n")
        f.write("Date: 2026-01-17\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Results saved to text file: {save_path}")


# =============================================================================
# 9. 결과 출력 함수
# =============================================================================

def print_results(result: Dict):
    """계산 결과를 보기 좋게 출력"""
    print("\n" + "=" * 80)
    print("       Annular Fin → Porous Media 파라미터 변환 결과")
    print("=" * 80)

    inp = result['input']
    print(f"\n[입력 조건]")
    print(f"  핀 간격 (Fs)        : {inp['Fs_mm']:.1f} mm")
    print(f"  핀 높이 (hf)        : {inp['hf_mm']:.1f} mm")
    print(f"  대기 온도           : {inp['T_celsius']:.2f} °C")
    print(f"  설계 풍속           : {inp['v_design']:.4f} m/s")
    print(f"  튜브 외경 (Dc)      : {inp['Dc_mm']:.1f} mm")
    print(f"  횡방향 피치 (s1)    : {inp['s1_mm']:.3f} mm")
    print(f"  튜브 열 수 (N)      : {inp['N']}")

    air = result['air']
    print(f"\n[공기 물성치 @ {inp['T_celsius']:.2f}°C]")
    print(f"  밀도 (ρ)            : {air['rho']:.4f} kg/m³")
    print(f"  점도 (μ)            : {air['mu']:.4e} Pa·s")
    print(f"  열전도도 (k)        : {air['k']:.4f} W/(m·K)")
    print(f"  Prandtl 수 (Pr)     : {air['Pr']:.3f}")

    geom = result['geometry']
    print(f"\n[기하학적 파라미터]")
    print(f"  핀 외경 (Do)        : {geom['Do_mm']:.1f} mm")
    print(f"  핀 피치 (Fp)        : {geom['Fp_mm']:.2f} mm")
    print(f"  종방향 피치 (s2)    : {geom['s2_mm']:.3f} mm")
    print(f"  공극률 (ε)          : {geom['epsilon']:.4f}")
    print(f"  최소유동면적비 (σ)  : {geom['sigma']:.4f}")
    print(f"  면적비 (Atot/Abare) : {geom['area_ratio']:.3f}")
    print(f"  비표면적 (a_fs)     : {geom['a_fs']:.2f} 1/m")

    porous = result['porous']
    print(f"\n{'='*80}")
    print(f"  ★★★ CFD Porous Media 입력값 (핵심 출력) ★★★")
    print(f"{'='*80}")
    print(f"  점성 저항 (1/K)     : {porous['inv_K']:.4e}  [1/m²]")
    print(f"  관성 저항 (C2)      : {porous['C2']:.4f}  [1/m]")
    print(f"  투과도 (K)          : {porous['K']:.4e}  [m²]")
    print(f"  공극률 (ε)          : {geom['epsilon']:.4f}  [-]")
    print(f"  비표면적 (a_fs)     : {geom['a_fs']:.2f}  [1/m]")
    print(f"{'='*80}")

    design = result['design_point']
    print(f"\n[설계점 성능]")
    print(f"  Reynolds 수 (Re_Dc) : {design['Re_Dc']:.1f}")
    print(f"  총 압력강하         : {design['dP_total_Pa']:.2f} Pa")
    print(f"  단위길이 압력강하   : {design['dP_per_L_Pa_m']:.2f} Pa/m")
    print(f"  열전달 계수 (h_fs)  : {design['h_fs_W_m2K']:.2f} W/(m²·K)")

    fit = result['fitting']
    print(f"\n[피팅 정보]")
    print(f"  피팅 속도 범위      : {inp['v_range'][0]:.1f} ~ {inp['v_range'][1]:.1f} m/s")
    print(f"  피팅 점 개수        : {inp['n_points']}")
    print(f"  Reynolds 수 범위    : {fit['Re_min']:.1f} ~ {fit['Re_max']:.1f}")
    print(f"  결정계수 (R²)       : {porous['R_squared']:.8f}")

    print("\n" + "=" * 80)


# =============================================================================
# 10. CLI 인터페이스
# =============================================================================

def get_cli_input():
    """
    CLI에서 사용자 입력을 받습니다.

    Returns:
        dict: 입력 파라미터
    """
    print("\n" + "█" * 80)
    print("  Nir (1991) 상관식 기반 Annular Fin Porous 파라미터 계산기")
    print("  다중점 피팅 방식 (0.5~3.5 m/s 범위)")
    print("█" * 80)
    print()

    try:
        Fs_mm = float(input("핀 간격 (Fs) [mm] (예: 4.0): "))
        hf_mm = float(input("핀 높이 (hf) [mm] (예: 4.0): "))
        T_celsius = float(input("대기 온도 [°C] (예: 14.8): "))
        v_design = float(input("설계 풍속 [m/s] (예: 2.0): "))

        print("\n고급 설정 (Enter로 기본값 사용)")
        Dc_mm_str = input(f"튜브 외경 (Dc) [mm] (기본값: 24.0): ")
        Dc_mm = float(Dc_mm_str) if Dc_mm_str.strip() else 24.0

        delta_f_mm_str = input(f"핀 두께 (δf) [mm] (기본값: 0.5): ")
        delta_f_mm = float(delta_f_mm_str) if delta_f_mm_str.strip() else 0.5

        s1_mm_str = input(f"횡방향 피치 (s1) [mm] (기본값: 55.333): ")
        s1_mm = float(s1_mm_str) if s1_mm_str.strip() else 55.333

        pitch_ratio_str = input(f"피치 비율 (s1/s2) (기본값: 1.0): ")
        pitch_ratio = float(pitch_ratio_str) if pitch_ratio_str.strip() else 1.0

        N_str = input(f"튜브 열 수 (기본값: 4): ")
        N = int(N_str) if N_str.strip() else 4

        v_min_str = input(f"피팅 최소 속도 [m/s] (기본값: 0.5): ")
        v_min = float(v_min_str) if v_min_str.strip() else 0.5

        v_max_str = input(f"피팅 최대 속도 [m/s] (기본값: 3.5): ")
        v_max = float(v_max_str) if v_max_str.strip() else 3.5

        n_points_str = input(f"피팅 점 개수 (기본값: 50): ")
        n_points = int(n_points_str) if n_points_str.strip() else 50

        return {
            'Fs_mm': Fs_mm,
            'hf_mm': hf_mm,
            'T_celsius': T_celsius,
            'v_design': v_design,
            'Dc_mm': Dc_mm,
            'delta_f_mm': delta_f_mm,
            's1_mm': s1_mm,
            'pitch_ratio': pitch_ratio,
            'N': N,
            'v_range': (v_min, v_max),
            'n_points': n_points
        }

    except ValueError as e:
        print(f"\n[오류] 숫자를 올바르게 입력해주세요: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
        sys.exit(0)


# =============================================================================
# 11. 메인 실행
# =============================================================================

if __name__ == "__main__":
    # argparse 설정
    parser = argparse.ArgumentParser(
        description='Annular Fin → Porous Media 파라미터 계산기 (Nir 1991)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python cfd_porous_calc.py --Fs 4.0 --hf 4.0 --T 15 --v 2.0
  python cfd_porous_calc.py --interactive
        """
    )

    parser.add_argument('--Fs', type=float, help='핀 간격 [mm]')
    parser.add_argument('--hf', type=float, help='핀 높이 [mm]')
    parser.add_argument('--T', type=float, help='대기 온도 [°C]')
    parser.add_argument('--v', type=float, help='설계 풍속 [m/s]')
    parser.add_argument('--Dc', type=float, default=24.0, help='튜브 외경 [mm] (기본: 24.0)')
    parser.add_argument('--delta_f', type=float, default=0.5, help='핀 두께 [mm] (기본: 0.5)')
    parser.add_argument('--s1', type=float, default=55.333, help='횡방향 피치 [mm] (기본: 55.333)')
    parser.add_argument('--pitch_ratio', type=float, default=1.0, help='s1/s2 비율 (기본: 1.0)')
    parser.add_argument('--N', type=int, default=4, help='튜브 열 수 (기본: 4)')
    parser.add_argument('--v_min', type=float, default=0.5, help='피팅 최소 속도 [m/s] (기본: 0.5)')
    parser.add_argument('--v_max', type=float, default=3.5, help='피팅 최대 속도 [m/s] (기본: 3.5)')
    parser.add_argument('--n_points', type=int, default=50, help='피팅 점 개수 (기본: 50)')
    parser.add_argument('--interactive', '-i', action='store_true', help='대화형 입력 모드')
    parser.add_argument('--no-plot', action='store_true', help='시각화 생성 안 함')

    args = parser.parse_args()

    # 대화형 모드 또는 필수 인자 확인
    if args.interactive or (args.Fs is None or args.hf is None or args.T is None or args.v is None):
        params = get_cli_input()
    else:
        params = {
            'Fs_mm': args.Fs,
            'hf_mm': args.hf,
            'T_celsius': args.T,
            'v_design': args.v,
            'Dc_mm': args.Dc,
            'delta_f_mm': args.delta_f,
            's1_mm': args.s1,
            'pitch_ratio': args.pitch_ratio,
            'N': args.N,
            'v_range': (args.v_min, args.v_max),
            'n_points': args.n_points
        }

    # 계산 실행
    print("\n계산 중...")
    result = calculate_porous_parameters(**params)

    # 결과 출력
    print_results(result)

    # 시각화
    if not args.no_plot:
        plot_fitting_results(result)

    # 텍스트 파일로 결과 저장
    save_results_to_txt(result)

    # JSON 출력 (선택사항)
    print("\n[JSON 형식 출력]")
    import json
    json_output = {
        'input': result['input'],
        'porous_parameters': {
            'inv_K_1_m2': result['porous']['inv_K'],
            'C2_1_m': result['porous']['C2'],
            'K_m2': result['porous']['K'],
            'porosity': result['geometry']['epsilon'],
            'a_fs_1_m': result['geometry']['a_fs'],
            'R_squared': result['porous']['R_squared']
        },
        'design_point': {
            'Re_Dc': result['design_point']['Re_Dc'],
            'dP_total_Pa': result['design_point']['dP_total_Pa'],
            'h_fs_W_m2K': result['design_point']['h_fs_W_m2K']
        }
    }
    print(json.dumps(json_output, indent=2))

    print("\n계산 완료!")