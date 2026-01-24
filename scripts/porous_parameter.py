from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class AirProps:
    """Air properties at 15°C, 1 atm (default values used in original script)."""
    rho: float = 1.225       # [kg/m^3]
    mu: float = 1.789e-5     # [Pa·s]
    k: float = 0.0253        # [W/m·K]
    pr: float = 0.71         # [-]


@dataclass(frozen=True)
class GeometryFixed:
    """Geometry constants fixed in the original script."""
    Dc: float = 0.024        # Tube outer diameter [m]
    delta_f: float = 0.0005  # Fin thickness [m]
    s1: float = 0.05533      # Spanwise pitch [m]
    N: int = 4               # Number of tube rows (used in Wang correlation)


@dataclass(frozen=True)
class DesignInput:
    """Design variables."""
    Fs_mm: float
    hf_mm: float
    pitch_ratio: float  # Ratio = s1/s2 (0.5 < ratio < 2.0)


@dataclass(frozen=True)
class FitVelocities:
    """Velocities used for 2-point Darcy-Forchheimer fitting."""
    v_design: float = 1.5  # [m/s] frontal velocity
    v_low: float = 0.5     # [m/s] frontal velocity


@dataclass(frozen=True)
class PorousResult:
    Error: bool
    Message: Optional[str]

    # Inputs (echo)
    Fs_mm: Optional[float] = None
    hf_mm: Optional[float] = None
    Ratio: Optional[float] = None

    # Derived geometry
    s2_mm: Optional[float] = None
    Porosity: Optional[float] = None
    Area_Density: Optional[float] = None  # [1/m]
    Dh_mm: Optional[float] = None
    sigma: Optional[float] = None

    # Porous coefficients
    Viscous_Res: Optional[float] = None   # 1/K [1/m^2]
    Inertial_Res: Optional[float] = None  # C2 [1/m]

    # Heat transfer (LTNE)
    h_fs: Optional[float] = None          # [W/m^2-K]

    # Reference values
    Re_Design: Optional[float] = None
    Nu: Optional[float] = None
    v_max_design: Optional[float] = None  # max/physical velocity at design point [m/s]
    dP_L_design: Optional[float] = None   # [Pa/m]


class PorousZoneCalculator:
    """
    CFD Porous Zone parameter calculator for annular finned tube bank.

    - Refactored from the provided script.
    - No CLI, no prints. Pure compute() API.
    - Keeps original math/correlations, but exposes fitting velocities.
    """

    def __init__(
        self,
        air: AirProps = AirProps(),
        geom: GeometryFixed = GeometryFixed(),
    ):
        self.air = air
        self.geom = geom

    # ---------- Public API ----------

    def compute(
        self,
        Fs_mm: float,
        hf_mm: float,
        pitch_ratio: float,
        fit: FitVelocities = FitVelocities(),
    ) -> Dict[str, Any]:
        """
        Compute porous coefficients and LTNE interfacial heat transfer coefficient.

        Returns a dict (easy integration). If you prefer a dataclass, use compute_result().
        """
        return asdict(self.compute_result(Fs_mm, hf_mm, pitch_ratio, fit))

    def compute_result(
        self,
        Fs_mm: float,
        hf_mm: float,
        pitch_ratio: float,
        fit: FitVelocities = FitVelocities(),
    ) -> PorousResult:
        # --- Validate pitch ratio ---
        if pitch_ratio <= 0.5 or pitch_ratio >= 2.0:
            return PorousResult(
                Error=True,
                Message=f"Invalid pitch_ratio ({pitch_ratio}). Must be 0.5 < Ratio < 2.0",
            )

        # --- Convert units ---
        Fs = Fs_mm / 1000.0
        hf = hf_mm / 1000.0

        # --- Unpack constants ---
        rho, mu, k_air, Pr = self.air.rho, self.air.mu, self.air.k, self.air.pr
        Dc, delta_f, s1, N = self.geom.Dc, self.geom.delta_f, self.geom.s1, self.geom.N

        # --- Derived pitches/diameters ---
        s2 = s1 / pitch_ratio
        Fp = Fs + delta_f
        Do = Dc + 2.0 * hf

        # --- Collision checks (simple in-line checks as in original) ---
        transverse_gap = s1 - Do
        longitudinal_gap = s2 - Do
        if transverse_gap < 0:
            return PorousResult(
                Error=True,
                Message=f"COLLISION! Fin OD ({Do*1000:.1f} mm) > Transverse Pitch ({s1*1000:.2f} mm)",
            )
        if longitudinal_gap < 0:
            return PorousResult(
                Error=True,
                Message=f"COLLISION! Fin OD ({Do*1000:.1f} mm) > Longitudinal Pitch ({s2*1000:.2f} mm)",
            )

        # --- Porosity (original 1D stacking approximation) ---
        epsilon = 1.0 - (delta_f / Fp)

        # --- Surface area density a_fs (original expression) ---
        area_numerator = (
            0.5 * math.pi * (Do**2 - Dc**2)
            + math.pi * Do * delta_f
            + math.pi * Dc * Fs
        )
        vol_denominator = (0.25 * math.pi * (Do**2 - Dc**2)) * Fp
        a_fs = area_numerator / vol_denominator  # [1/m]

        # --- Hydraulic diameter ---
        Dh = 4.0 * epsilon / a_fs  # [m]

        # --- Minimum flow area ratio (sigma) ---
        sigma = (s1 - Dc - 2.0 * hf * (delta_f / Fp)) / s1
        if sigma <= 0:
            return PorousResult(
                Error=True,
                Message=f"Invalid sigma (<=0). Check geometry: sigma={sigma:.6f}",
            )

        # --- Pressure drop via Wang correlation + Darcy-Forchheimer fit
