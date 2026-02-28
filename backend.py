# backend.py  ─  Computational core: Smart Charging + V2G MILP optimisation
# Project : S.KOe COOL – Reefer Trailer Bi-Directional Charging
# Author  : Kuldip Bhadreshvara, Schmitz Cargobull AG  (2025)
# Refs    : Agora Verkehrswende (2025) Bidirektionales Laden
#           Biedenbach & Strunz (2024) Multi-Use Depot Optimization WEVJ

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EVParams:
    """Battery + OBC parameters for one trailer.  Matches MATLAB EVSmartwithReefer."""
    BatteryCapacity_kWh: float
    UsableBatteryCap_kWh: float
    BatteryChargingEffi_pc: float
    OBC_Capacity_kW: float
    OBC_UsableCapacity_kW: float
    OBCEfficiency_pc: float
    SOC_arrival_winter_pc: float
    SOC_arrival_summer_pc: float
    SOC_departure_target_pc: float
    ArrivalTime_HHMM: str
    DepartureTime_HHMM: str
    MaxChargingPower_kW: float

    # Site / grid defaults
    PF_Reefer: float = 0.75
    GridPF_site: float = 0.98
    GridVoltage_V: float = 400
    GridCurrent_A: float = 32
    GridMax_kVA: float = None
    BattMaxCharge_kW: float = 14.148
    BattMaxDischarge_kW: float = 14.680
    EAxle_Efficiency_pc: float = 90
    EffectiveChargingPower_kW: float = None
    ReeferCycleInit: str = "Continuous"
    # ── Field 30 & 31: Hard grid capacity limits ─────────────────────────────
    DepotConnection_kVA: float = None   # Field 30 – total depot connection limit
    TransformerLimit_kVA: float = None  # Field 31 – on-site transformer rated kVA

    def finalize(self):
        self.GridMax_kVA = math.sqrt(3) * self.GridVoltage_V * self.GridCurrent_A / 1000.0
        self.EffectiveChargingPower_kW = min(self.OBC_UsableCapacity_kW, self.MaxChargingPower_kW)
        # Effective single-trailer kVA cap = tightest limit available
        caps = [self.GridMax_kVA]
        if self.DepotConnection_kVA is not None and self.DepotConnection_kVA > 0:
            caps.append(self.DepotConnection_kVA)
        if self.TransformerLimit_kVA is not None and self.TransformerLimit_kVA > 0:
            caps.append(self.TransformerLimit_kVA)
        self.EffectiveGridCap_kVA = min(caps)

    @property
    def binding_kva_limit(self) -> float:
        """Return the kVA value that is actually the binding constraint."""
        return getattr(self, "EffectiveGridCap_kVA", self.GridMax_kVA)


@dataclass
class V2GParams:
    """Parameters for the V2G MILP optimiser (Biedenbach & Strunz 2024 model)."""
    battery_capacity_kWh: float = 82.0
    usable_capacity_kWh: float  = 65.6      # SoC 20-95 %
    soc_min_pct: float          = 20.0       # cold-chain floor
    soc_max_pct: float          = 95.0       # cycle ceiling
    charge_power_kW: float      = 22.0
    discharge_power_kW: float   = 11.0       # V2G max
    eta_charge: float           = 0.92
    eta_discharge: float        = 0.92
    deg_cost_eur_kwh: float     = 0.07       # €/kWh cycled
    dt_h: float                 = 0.25       # 15-min slots
    n_slots: int                = 96         # 24 h × 4
    # ── Field 30 & 31: Hard grid capacity limits ─────────────────────────────
    depot_connection_kVA: float = 0.0        # 0 = no separate depot limit
    transformer_limit_kVA: float = 0.0       # 0 = no transformer limit


@dataclass
class V2GResult:
    """Output from one MILP optimisation run."""
    scenario: str
    p_charge: np.ndarray        # kW  into battery from grid
    p_discharge: np.ndarray     # kW  discharged from battery to grid
    soc: np.ndarray             # kWh stored
    cost_eur_day: float         # net daily cost (negative = revenue)
    v2g_revenue_eur_day: float
    v2g_export_kwh_day: float
    charge_cost_eur_day: float
    deg_cost_eur_day: float
    price_buy: np.ndarray
    price_v2g: np.ndarray
    plugged: np.ndarray
    tru_load: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – TARIFF & PRICE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_tariff_params() -> Dict[str, float]:
    return {
        "ConcessionFee_ct":     1.992,
        "OffshoreGridLevy_ct":  0.816,
        "CHPLevy_ct":           0.277,
        "ElectricityTax_ct":    2.05,
        "NEV19Levy_ct":         1.558,
        "NetworkUsageFees_ct":  6.63,
        "SalesMargin_ct":       0.000,
        "VAT_pc":               19.0,
    }

def compose_all_in_price(spot_eur_24: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    fixed_ct = sum([
        params["ConcessionFee_ct"], params["OffshoreGridLevy_ct"],
        params["CHPLevy_ct"],        params["ElectricityTax_ct"],
        params["NEV19Levy_ct"],      params["NetworkUsageFees_ct"],
        params["SalesMargin_ct"],
    ])
    net = np.asarray(spot_eur_24, dtype=float) + fixed_ct / 100.0
    return net * (1.0 + params["VAT_pc"] / 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════

def read_price_excel(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(path, engine="openpyxl")
    if not {"WinterWD", "SummerWD"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: WinterWD, SummerWD")
    w = np.asarray(df["WinterWD"]).reshape(-1)
    s = np.asarray(df["SummerWD"]).reshape(-1)
    if len(w) != 24 or len(s) != 24:
        raise ValueError("Price columns must each have exactly 24 values.")
    return w, s

def read_taper_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    if not {"Time", "SoC"}.issubset(df.columns):
        raise ValueError("time_soc.xlsx must contain columns: Time (min), SoC (%)")
    df = df[["Time", "SoC"]].dropna().sort_values("Time")
    return df

def read_v2g_params(path: str) -> Tuple[V2GParams, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read v2g_params.xlsx → (V2GParams, prices_15min_df, deg_sensitivity_df, dwell_df)."""
    xls = pd.ExcelFile(path, engine="openpyxl")
    df_bat  = pd.read_excel(xls, "BatteryParams").set_index("Parameter")["Value"]
    df_pr   = pd.read_excel(xls, "Prices15min")
    df_deg  = pd.read_excel(xls, "DegSensitivity")
    df_dwl  = pd.read_excel(xls, "DwellProfiles")

    v2g = V2GParams(
        battery_capacity_kWh   = float(df_bat["BatteryCapacity_kWh"]),
        usable_capacity_kWh    = float(df_bat["UsableBatteryCap_kWh"]),
        soc_min_pct            = float(df_bat["SOC_min_pct"]),
        soc_max_pct            = float(df_bat["SOC_max_pct"]),
        charge_power_kW        = float(df_bat["ChargePower_kW"]),
        discharge_power_kW     = float(df_bat["DischargePower_V2G_kW"]),
        eta_charge             = float(df_bat["ChargeEfficiency"]),
        eta_discharge          = float(df_bat["DischargeEfficiency"]),
        deg_cost_eur_kwh       = float(df_bat["DegradationCost_EUR_kWh"]),
    )
    return v2g, df_pr, df_deg, df_dwl

def read_telematics(path: str) -> pd.DataFrame:
    """
    Read real or synthetic telematics data.
    Expected columns: Timestamp, SoC_pct, TRU_Load_kW, Plugged_In
    Falls back gracefully if file is missing (uses synthetic data).
    """
    df = pd.read_excel(path, engine="openpyxl", parse_dates=["Timestamp"])
    required = {"Timestamp", "SoC_pct", "TRU_Load_kW", "Plugged_In"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"telematics file missing columns: {missing}")
    return df.sort_values("Timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – TIME & TAPER HELPERS  (existing smart-charging module)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.strip().split(":")
    return int(hh), int(mm)

def build_time_vector(arr_str: str, dep_str: str):
    base_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ah, am = _parse_hhmm(arr_str)
    dh, dm = _parse_hhmm(dep_str)
    t_arr = base_day + timedelta(hours=ah, minutes=am)
    t_dep = base_day + timedelta(hours=dh, minutes=dm)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    dt_sec = 60
    t = []
    cur = t_arr
    while cur <= t_dep - timedelta(seconds=dt_sec):
        t.append(cur)
        cur += timedelta(seconds=dt_sec)
    return t_arr, t_dep, t, dt_sec / 3600.0

def build_taper_lookup(soc_df: pd.DataFrame, EV: EVParams, eff_frac: float):
    minutes  = soc_df["Time"].to_numpy()
    soc_frac = np.clip(soc_df["SoC"].to_numpy() / 100.0, 0.0, 1.0)
    SOC_kWh  = soc_frac * EV.UsableBatteryCap_kWh
    dt_hr    = np.diff(minutes) / 60.0
    dSOC     = np.diff(SOC_kWh)
    P_batt   = np.maximum(0.0, dSOC / dt_hr)
    P_grid   = P_batt / max(eff_frac, np.finfo(float).eps)
    if P_grid.size >= 3:
        P_grid = np.convolve(P_grid, np.ones(3) / 3.0, mode="same")
    soc_raw = soc_frac[:-1]
    idx = np.argsort(soc_raw)
    soc_s, P_s = soc_raw[idx], np.maximum(0.0, P_grid[idx])
    uniq_s, uniq_P, seen = [], [], set()
    for s, p in zip(soc_s, P_s):
        if s not in seen:
            seen.add(s); uniq_s.append(s); uniq_P.append(p)
    return np.array(uniq_s), np.array(uniq_P)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – REEFER CYCLE  (existing smart-charging module)
# ═══════════════════════════════════════════════════════════════════════════════

REEFER_DEFAULTS = {
    "P_HIGH_C": 7.6, "P_LOW_C": 0.7,
    "P_HIGH_SS": 9.7, "P_MID_SS": 0.65, "P_LOW_SS": 0.0,
    "T_HIGH_C": 1717, "T_LOW_C": 292,
    "T_HIGH_SS": 975, "T_MID_SS": 295, "T_LOW_SS": 1207,
}

def get_reefer_cycle_trace(cycle_type: str, N: int, dt_sec: int = 60) -> np.ndarray:
    D = REEFER_DEFAULTS
    ct = cycle_type.replace(" ", "").replace("-", "").lower()
    if ct == "continuous":
        pattern = np.array([[D["P_HIGH_C"], D["T_HIGH_C"]], [D["P_LOW_C"], D["T_LOW_C"]]])
    elif ct in ("startstop",):
        pattern = np.array([
            [D["P_HIGH_SS"], D["T_HIGH_SS"]],
            [D["P_MID_SS"],  D["T_MID_SS"]],
            [D["P_LOW_SS"],  D["T_LOW_SS"]],
        ])
    else:
        return np.zeros(N)
    steps = np.maximum(1, np.round(pattern[:, 1] / dt_sec).astype(int))
    pw = np.repeat(pattern[:, 0], steps)
    if pw.size == 0:
        return np.zeros(N)
    full = np.tile(pw, int(np.ceil(N / len(pw))))
    return full[:N]

def _aggregate_to_minutes(t_hi, P_kW):
    bins = [dt.replace(second=0, microsecond=0) for dt in t_hi]
    bucket, t_min = {}, []
    for i, b in enumerate(bins):
        bucket.setdefault(b, []).append(i)
        if b not in dict.fromkeys(t_min):
            t_min.append(b)
    Pmin = np.array([float(np.mean(P_kW[bucket[b]])) for b in t_min])
    return t_min, Pmin

def build_reefer_stationary_minute_trace(
    cycle_type, t_minutes, t_arr, t_dep,
    dt_reefer_sec=10, PF_reefer=0.75, kVA_refr_cap=19.765,
):
    if cycle_type.strip().lower() == "noreeferstationary":
        return np.zeros(len(t_minutes)), np.zeros(len(t_minutes))
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    N10 = int(max(0, (t_dep - t_arr).total_seconds() // dt_reefer_sec))
    if N10 <= 0:
        return np.zeros(len(t_minutes)), np.zeros(len(t_minutes))
    P10 = get_reefer_cycle_trace(cycle_type, N10, dt_reefer_sec)
    kVA10 = np.minimum(P10 / max(PF_reefer, np.finfo(float).eps), kVA_refr_cap)
    t10 = [t_arr + timedelta(seconds=i * dt_reefer_sec) for i in range(N10)]
    tMin, Pmin = _aggregate_to_minutes(t10, P10)
    _,    kVAm = _aggregate_to_minutes(t10, kVA10)
    P_out = np.zeros(len(t_minutes))
    kVA_out = np.zeros(len(t_minutes))
    idx_lut = {tm: j for j, tm in enumerate(tMin)}
    for i, tm in enumerate(t_minutes):
        j = idx_lut.get(tm)
        if j is not None:
            P_out[i] = Pmin[j]; kVA_out[i] = kVAm[j]
    return P_out, kVA_out


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – SMART CHARGING SIMULATION  (existing module)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    P_trace: np.ndarray
    SOC_trace: np.ndarray
    reachedTarget: bool
    delivered_kWh: float
    kVA_grid_total: np.ndarray

def simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp_kW,
             eff_frac, th, P_refr_min_kW, kVA_refr_min) -> SimResult:
    N = len(t)
    P_trace = np.zeros(N); SOC_trace = np.zeros(N)
    SOC = getattr(EV, "CurrentSOC_kWh", EV.UsableBatteryCap_kWh * EV.SOC_arrival_winter_pc / 100.0)
    target = EV.UsableBatteryCap_kWh * (EV.SOC_departure_target_pc / 100.0)
    reached = False
    eta_batt = max(np.finfo(float).eps, EV.BatteryChargingEffi_pc / 100.0)
    eta_OBC  = max(np.finfo(float).eps, EV.OBCEfficiency_pc / 100.0)
    kVA_grid_total = np.zeros(N)
    for i in range(N):
        SOC_trace[i] = SOC
        if SOC >= target - 1e-9:
            reached = True; SOC_trace[i:] = SOC; break
        soc_frac = np.clip(SOC / EV.UsableBatteryCap_kWh, 0.0, 1.0)
        P_tap_grid = float(np.interp(soc_frac, soc_bp, Pcap_grid_bp_kW))
        P_tap_batt = P_tap_grid * eta_batt * eta_OBC
        P_eff      = min(P_tap_batt, EV.EffectiveChargingPower_kW * eta_batt * eta_OBC, EV.BattMaxCharge_kW)
        kVA_refr   = max(0.0, kVA_refr_min[i] if i < len(kVA_refr_min) else 0.0)
        # Field 30/31: use tightest grid cap (connection, transformer, or computed from V/I)
        eff_kva    = getattr(EV, "EffectiveGridCap_kVA", EV.GridMax_kVA) or EV.GridMax_kVA
        rhs        = (eff_kva / eta_OBC) - kVA_refr
        P_kVA_cap  = max(0.0, rhs / max(np.finfo(float).eps, EV.GridPF_site * eta_batt))
        if price_min[i] <= th:
            P_batt = min(P_eff, P_kVA_cap, max(0.0, (target - SOC) / dt_hr))
        else:
            P_batt = 0.0
        SOC = min(EV.UsableBatteryCap_kWh, SOC + P_batt * dt_hr)
        P_gridAC = P_batt / (eta_batt * eta_OBC)
        P_trace[i] = P_gridAC
        kVA_grid_total[i] = (kVA_refr + P_batt * eta_batt * EV.GridPF_site) * eta_OBC
    return SimResult(P_trace, SOC_trace, reached, float(np.sum(P_trace * eta_batt * eta_OBC) * dt_hr), kVA_grid_total)

def empty_sim(t, EV) -> SimResult:
    N = len(t)
    soc0 = EV.UsableBatteryCap_kWh * EV.SOC_arrival_winter_pc / 100.0
    return SimResult(np.zeros(N), np.full(N, soc0), True, 0.0, np.zeros(N))

def _finalize(sim, price_min, dt_hr, th):
    return {
        "P_trace": sim.P_trace, "SOC_trace": sim.SOC_trace,
        "reachedTarget": sim.reachedTarget, "delivered_kWh": sim.delivered_kWh,
        "price_min": price_min, "energy_kWh": float(np.sum(sim.P_trace) * dt_hr),
        "cost_EUR": float(np.sum(sim.P_trace * price_min) * dt_hr),
        "thresholdUsed": th, "kVA_grid_total": sim.kVA_grid_total,
    }

def plan_smart(t, dt_hr, price24, EV, soc_bp, Pcap_grid_bp, need_kWh, eff_frac, P_refr, kVA_refr):
    price_min = np.array([price24[dt.hour] for dt in t], dtype=float)
    if need_kWh <= 1e-12:
        return _finalize(empty_sim(t, EV), price_min, dt_hr, float("nan"))
    uniq = np.unique(price_min); best = None; usedTh = float("inf"); met = False
    for th in uniq:
        sim = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp, eff_frac, th, P_refr, kVA_refr)
        if sim.delivered_kWh >= need_kWh - 1e-9:
            best, usedTh, met = sim, float(th), True; break
        best, usedTh = sim, float(th)
    if not met:
        best = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp, eff_frac, float("inf"), P_refr, kVA_refr)
        usedTh = float("inf")
    return _finalize(best, price_min, dt_hr, usedTh)

def plan_baseline(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp, eff_frac, P_refr, kVA_refr):
    sim = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp, eff_frac, float("inf"), P_refr, kVA_refr)
    return _finalize(sim, price_min, dt_hr, float("inf"))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – V2G MILP OPTIMISER
#  Based on: Biedenbach & Strunz (2024) eFlame model + Agora V2X framework
#  Formulation: minimise cost = Σ p_buy·P_c - Σ p_v2g·P_d + Σ deg·(P_c+P_d)
#  Subject to:  SoC dynamics, power limits, mutual exclusion, plug-in windows
# ═══════════════════════════════════════════════════════════════════════════════

def _build_synthetic_inputs(v2g: V2GParams, dwell_profile: str = "Extended"):
    """
    Build synthetic 15-min price & TRU-load vectors when real telematics
    are not yet available.  Replace with read_telematics() later.
    """
    N = v2g.n_slots
    t_slots = np.arange(N)
    hours   = t_slots * v2g.dt_h

    # German all-in buy price (€/kWh) – EPEX intraday style
    buy = np.where((hours >= 6)  & (hours < 8),   0.28,
          np.where((hours >= 8)  & (hours < 11),  0.30,
          np.where((hours >= 11) & (hours < 14),  0.24,
          np.where((hours >= 14) & (hours < 16),  0.22,
          np.where((hours >= 16) & (hours < 20),  0.33,
          np.where((hours >= 20) & (hours < 22),  0.26,
                                                   0.16))))))

    # V2G price = buy + balancing premium during evening peak
    prem = np.where((hours >= 16) & (hours < 20), 0.132, 0.0)
    v2g_p = buy + prem

    # TRU load: sinusoidal between 1.5 and 4.2 kW (ambient-influenced)
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * t_slots / N + np.pi)

    # Plug-in window
    if dwell_profile == "NightOnly":
        plugged = ((hours >= 21) | (hours < 7)).astype(int)
    else:  # Extended: night + midday stop
        plugged = (
            (hours >= 21) | (hours < 7) |
            ((hours >= 12) & (hours < 18))
        ).astype(int)

    return buy, v2g_p, tru, plugged


def run_v2g_milp(
    v2g: V2GParams,
    buy_price: Optional[np.ndarray]  = None,
    v2g_price: Optional[np.ndarray]  = None,
    tru_load:  Optional[np.ndarray]  = None,
    plugged:   Optional[np.ndarray]  = None,
    dwell_profile: str = "Extended",
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 80.0,
    deg_cost_override: Optional[float] = None,
    scenario_label: str = "SmartV2G",
) -> V2GResult:
    """
    MILP optimiser for smart charging + V2G.

    Objective (per Biedenbach & Strunz 2024, eq. 2-6):
        min  Σ buy·P_c·dt  –  Σ v2g·P_d·dt  +  Σ deg·(P_c+P_d)·dt

    Decision variables per slot t:
        P_c[t]  ≥ 0   charge power  (kW from grid)
        P_d[t]  ≥ 0   discharge power (kW to grid, V2G)
        e[t]    ≥ 0   SoC (kWh)

    Key constraints:
        e[t] = e[t-1] + P_c·eta_c·dt – P_d/eta_d·dt – P_tru·dt
        E_min ≤ e[t] ≤ E_max
        0 ≤ P_c[t] ≤ P_c_max · plug[t]
        0 ≤ P_d[t] ≤ P_d_max · plug[t]
        P_c[t] + P_d[t] ≤ P_c_max           (linearised mutex)
        e[0] = soc_init;  e[N-1] ≥ soc_final
    """
    import warnings

    N   = v2g.n_slots
    dt  = v2g.dt_h
    deg = deg_cost_override if deg_cost_override is not None else v2g.deg_cost_eur_kwh

    # Default to synthetic inputs if not provided
    if buy_price is None or v2g_price is None or tru_load is None or plugged is None:
        buy_price, v2g_price, tru_load, plugged = _build_synthetic_inputs(v2g, dwell_profile)

    E_min  = v2g.usable_capacity_kWh * v2g.soc_min_pct / 100.0
    E_max  = v2g.usable_capacity_kWh * v2g.soc_max_pct / 100.0
    E_init = v2g.usable_capacity_kWh * soc_init_pct   / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct  / 100.0

    # ── Try scipy.optimize.milp (HiGHS, scipy ≥ 1.7) ──────────────────────
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix

        # Variable layout: [P_c(0..N-1), P_d(0..N-1), e(0..N-1)]  → 3N vars
        nv = 3 * N
        idx_c = np.arange(N)
        idx_d = np.arange(N, 2 * N)
        idx_e = np.arange(2 * N, 3 * N)

        # Cost vector
        c = np.zeros(nv)
        c[idx_c] =  buy_price * dt + deg * dt   # buy cost + degradation charging
        c[idx_d] = -v2g_price * dt + deg * dt   # V2G revenue (negative = revenue) + deg
        # c[idx_e] = 0  (SoC is free in objective)

        # Bounds on variables
        lb = np.zeros(nv)
        ub = np.full(nv, np.inf)

        # ── Fields 30 & 31: compute per-slot kW ceiling from kVA limits ──────
        # Convert kVA limit → kW limit (assume PF≈0.95 for EV charger at AC bus)
        _CHARGER_PF = 0.95
        _limits_kVA = []
        if v2g.depot_connection_kVA  > 0: _limits_kVA.append(v2g.depot_connection_kVA)
        if v2g.transformer_limit_kVA > 0: _limits_kVA.append(v2g.transformer_limit_kVA)
        if _limits_kVA:
            grid_kW_cap = min(_limits_kVA) * _CHARGER_PF   # binding kVA → kW
        else:
            grid_kW_cap = max(v2g.charge_power_kW, v2g.discharge_power_kW) * 10  # effectively unconstrained

        # Charge power: 0 ≤ P_c ≤ min(P_c_max, grid_kW_cap) × plug
        ub[idx_c] = np.minimum(v2g.charge_power_kW, grid_kW_cap)    * plugged
        # Discharge power: 0 ≤ P_d ≤ min(P_d_max, grid_kW_cap) × plug
        ub[idx_d] = np.minimum(v2g.discharge_power_kW, grid_kW_cap) * plugged
        # SoC: E_min ≤ e ≤ E_max
        lb[idx_e] = E_min
        ub[idx_e] = E_max

        # Constraint matrix
        ncon = N + N + 1   # SoC dynamics (N) + mutex (N) + final SoC (1)
        A = lil_matrix((ncon, nv))
        lo = np.zeros(ncon)
        hi = np.zeros(ncon)

        # 1) SoC dynamics: e[t] – e[t-1] – P_c·eta_c·dt + P_d/eta_d·dt = –tru·dt
        for t in range(N):
            row = t
            A[row, idx_e[t]]  =  1.0
            A[row, idx_c[t]]  = -v2g.eta_charge   * dt
            A[row, idx_d[t]]  =  (1.0 / v2g.eta_discharge) * dt
            if t == 0:
                lo[row] = hi[row] = E_init - tru_load[t] * dt
            else:
                A[row, idx_e[t - 1]] = -1.0
                val = -tru_load[t] * dt
                lo[row] = hi[row] = val

        # 2) Mutex (linearised): P_c[t] + P_d[t] ≤ max_power
        max_power = max(v2g.charge_power_kW, v2g.discharge_power_kW)
        for t in range(N):
            row = N + t
            A[row, idx_c[t]] = 1.0
            A[row, idx_d[t]] = 1.0
            lo[row] = -np.inf
            hi[row] = max_power

        # 3) Final SoC: e[N-1] ≥ E_fin
        row = 2 * N
        A[row, idx_e[N - 1]] = 1.0
        lo[row] = E_fin
        hi[row] = E_max

        from scipy.sparse import csc_matrix
        result = milp(
            c,
            constraints=LinearConstraint(csc_matrix(A), lo, hi),
            bounds=Bounds(lb, ub),
            options={"disp": False, "time_limit": 30},
        )

        if result.success:
            P_c = np.clip(result.x[idx_c], 0, None)
            P_d = np.clip(result.x[idx_d], 0, None)
            e   = result.x[idx_e]
        else:
            warnings.warn(f"MILP solver: {result.message} — falling back to greedy")
            P_c, P_d, e = _greedy_v2g(v2g, buy_price, v2g_price, tru_load, plugged,
                                       E_init, E_fin, E_min, E_max, deg)

    except (ImportError, Exception) as ex:
        warnings.warn(f"scipy.milp unavailable ({ex}), using greedy heuristic")
        P_c, P_d, e = _greedy_v2g(v2g, buy_price, v2g_price, tru_load, plugged,
                                   E_init, E_fin, E_min, E_max, deg)

    # ── Compute KPIs ────────────────────────────────────────────────────────
    charge_cost   = float(np.sum(P_c * buy_price) * dt)
    v2g_rev       = float(np.sum(P_d * v2g_price) * dt)
    deg_cost_tot  = float(np.sum((P_c + P_d) * deg) * dt)
    net_cost      = charge_cost - v2g_rev + deg_cost_tot
    v2g_export    = float(np.sum(P_d) * dt)

    return V2GResult(
        scenario            = scenario_label,
        p_charge            = P_c,
        p_discharge         = P_d,
        soc                 = e,
        cost_eur_day        = net_cost,
        v2g_revenue_eur_day = v2g_rev,
        v2g_export_kwh_day  = v2g_export,
        charge_cost_eur_day = charge_cost,
        deg_cost_eur_day    = deg_cost_tot,
        price_buy           = buy_price,
        price_v2g           = v2g_price,
        plugged             = plugged,
        tru_load            = tru_load,
    )


def _greedy_v2g(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, E_min, E_max, deg):
    """Greedy fallback: charge cheap, discharge expensive, respect SoC bounds + kVA limits."""
    N  = v2g.n_slots; dt = v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc = E_init

    # ── Fields 30 & 31: compute kW ceiling from kVA limits (same as MILP) ───
    _CHARGER_PF = 0.95
    _limits_kVA = []
    if v2g.depot_connection_kVA  > 0: _limits_kVA.append(v2g.depot_connection_kVA)
    if v2g.transformer_limit_kVA > 0: _limits_kVA.append(v2g.transformer_limit_kVA)
    grid_kW_cap = (min(_limits_kVA) * _CHARGER_PF) if _limits_kVA else float("inf")
    p_c_max = min(v2g.charge_power_kW,    grid_kW_cap)
    p_d_max = min(v2g.discharge_power_kW, grid_kW_cap)

    # Simple greedy: charge night valleys, discharge evening peaks
    soc = E_init
    for t in range(N):
        soc -= tru[t] * dt
        soc  = max(E_min, soc)
        if plugged[t]:
            v2g_margin = (v2g_p[t] - deg) - (buy[t] + deg)
            if v2g_margin > 0.02 and soc > E_min + p_d_max * dt / v2g.eta_discharge:
                # Discharge
                p = min(p_d_max,
                        (soc - E_min) * v2g.eta_discharge / dt)
                P_d[t] = p
                soc    = max(E_min, soc - p / v2g.eta_discharge * dt)
            elif buy[t] < 0.22 and soc < E_max - p_c_max * dt * v2g.eta_charge:
                # Charge
                p = min(p_c_max,
                        (E_max - soc) / (v2g.eta_charge * dt))
                P_c[t] = p
                soc    = min(E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    # Ensure final SoC ≥ E_fin (top up if needed)
    if soc < E_fin:
        for t in range(N - 1, -1, -1):
            if plugged[t] and P_d[t] == 0:
                deficit = E_fin - soc
                extra   = min(v2g.charge_power_kW * dt * v2g.eta_charge, deficit)
                P_c[t] += extra / (v2g.eta_charge * dt)
                soc     += extra
                if soc >= E_fin:
                    break
    return P_c, P_d, e


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – SCENARIO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_v2g_scenarios(
    v2g: V2GParams,
    df_prices: pd.DataFrame,
    df_deg_sens: pd.DataFrame,
    dwell_profile: str = "Extended",
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 80.0,
) -> Dict[str, V2GResult]:
    """
    Run the three canonical scenarios from the thesis:
      A – Dumb (uncontrolled)
      B – Smart charging only (no V2G)
      C – Smart + V2G
    Plus a degradation sensitivity sweep.
    """
    buy = df_prices["BuyPrice_EUR_kWh"].values
    v2g_p = df_prices["V2G_Price_EUR_kWh"].values

    _, _, tru, plugged = _build_synthetic_inputs(v2g, dwell_profile)

    results = {}

    # A – Dumb: always charge when plugged in regardless of price
    buy_flat = np.full_like(buy, buy.mean())
    results["A_Dumb"] = run_v2g_milp(
        v2g, buy_flat, v2g_p * 0, tru, plugged, dwell_profile,
        soc_init_pct, soc_final_pct,
        deg_cost_override=0.0, scenario_label="A – Dumb Charging",
    )

    # B – Smart no V2G: optimise charging cost, block discharge
    results["B_Smart"] = run_v2g_milp(
        v2g, buy, v2g_p * 0, tru, plugged, dwell_profile,
        soc_init_pct, soc_final_pct,
        deg_cost_override=v2g.deg_cost_eur_kwh, scenario_label="B – Smart (no V2G)",
    )

    # C – Smart + V2G: full optimisation
    results["C_V2G"] = run_v2g_milp(
        v2g, buy, v2g_p, tru, plugged, dwell_profile,
        soc_init_pct, soc_final_pct,
        deg_cost_override=v2g.deg_cost_eur_kwh, scenario_label="C – Smart + V2G",
    )

    # Degradation sensitivity
    for _, row in df_deg_sens.iterrows():
        deg_c = float(row["DegCost_EUR_kWh"])
        key   = f"DegSens_{deg_c:.3f}"
        results[key] = run_v2g_milp(
            v2g, buy, v2g_p, tru, plugged, dwell_profile,
            soc_init_pct, soc_final_pct,
            deg_cost_override=deg_c, scenario_label=f"Deg={deg_c:.3f}",
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – FLEET SCALING  (Agora Verkehrswende 2025, Fig. 3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fleet_impact(single: V2GResult, fleet_sizes: List[int]) -> pd.DataFrame:
    rows = []
    for n in fleet_sizes:
        rows.append({
            "Fleet_Size":           n,
            "Peak_Charge_kW":       float(np.max(single.p_charge)) * n,
            "Peak_Discharge_kW":    float(np.max(single.p_discharge)) * n,
            "Annual_V2G_Export_MWh": single.v2g_export_kwh_day * 365 * n / 1000,
            "Annual_V2G_Revenue_EUR": single.v2g_revenue_eur_day * 365 * n,
            "Annual_Net_Cost_EUR":    single.cost_eur_day * 365 * n,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – REEFER COST  (existing smart-charging module)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReeferCostParams:
    FixedPrice_EUR_per_kWh: float = 0.35
    DieselPrice_EUR_per_L: float  = 1.80
    Diesel_kWh_per_L: float       = 9.8
    Genset_efficiency_frac: float = 0.30
    DieselFixedCons_L_per_h: float= 2.5
    Method: str                   = "energy"

def get_reefer_cost_params() -> ReeferCostParams:
    return ReeferCostParams()

def compute_reefer_cost_scenarios(P_refr_min_kW, dt_hr, price_min, params):
    E_kWh = float(np.sum(P_refr_min_kW) * dt_hr)
    cost_dynamic = float(np.sum(P_refr_min_kW * price_min) * dt_hr)
    cost_fixed   = E_kWh * params.FixedPrice_EUR_per_kWh
    liters_energy = E_kWh / max(np.finfo(float).eps, params.Diesel_kWh_per_L * params.Genset_efficiency_frac)
    hours_total   = len(P_refr_min_kW) * dt_hr
    liters_fixed  = params.DieselFixedCons_L_per_h * hours_total
    liters_used   = liters_fixed if params.Method.lower() == "fixed" else liters_energy
    cost_diesel   = liters_used * params.DieselPrice_EUR_per_L
    return {
        "E_kWh": E_kWh, "cost_dynamic": cost_dynamic,
        "cost_fixed": cost_fixed, "cost_diesel": cost_diesel,
        "diesel_liters_energy": liters_energy,
        "diesel_liters_fixed":  liters_fixed,
        "diesel_liters_used":   liters_used,
    }

# ── Misc helpers ──────────────────────────────────────────────────────────────
def clamp(x, a, b): return max(a, min(b, x))
def fmt1(x): return f"{x:.1f}"
def fmt2(x): return f"{x:.2f}"
