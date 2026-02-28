"""
generate_data.py
================
Run this ONCE to create all synthetic data files needed by the app.
When you receive real telematics data from TrailerConnect, replace the
relevant Excel files in the /data folder – the app will automatically
pick them up without any code changes.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1.  avg_price.xlsx  – 24-hour EPEX spot prices
#     (used by smart-charging module)
# ─────────────────────────────────────────────
# Typical German EPEX day-ahead hourly averages (€/kWh, ex-VAT)
# Winter: higher prices, less solar suppression
# Summer: midday dip from solar surplus
winter_spot = np.array([
    0.055, 0.048, 0.042, 0.038, 0.037, 0.040,   # 00-05
    0.058, 0.085, 0.097, 0.088, 0.078, 0.072,   # 06-11
    0.068, 0.065, 0.063, 0.070, 0.092, 0.105,   # 12-17
    0.112, 0.108, 0.095, 0.082, 0.072, 0.060,   # 18-23
])
summer_spot = np.array([
    0.042, 0.035, 0.030, 0.027, 0.026, 0.030,   # 00-05
    0.045, 0.062, 0.055, 0.038, 0.020, 0.008,   # 06-11
    0.005, 0.012, 0.022, 0.040, 0.068, 0.085,   # 12-17
    0.092, 0.088, 0.075, 0.060, 0.050, 0.044,   # 18-23
])
df_price = pd.DataFrame({"WinterWD": winter_spot, "SummerWD": summer_spot})
df_price.to_excel(DATA_DIR / "avg_price.xlsx", index=False)
print("✓  data/avg_price.xlsx written")

# ─────────────────────────────────────────────
# 2.  time_soc.xlsx  – CC-CV charging taper curve
#     Represents how OBC power reduces as SoC rises
# ─────────────────────────────────────────────
# Time in minutes from plug-in, SoC in %
# Mimics a 22 kW OBC on a 70 kWh usable pack
time_min = np.array([
     0,  5, 10, 15, 20, 25, 30, 40, 50, 60,
    70, 80, 90,100,110,120,140,160,180,200,
   210,220,230,240,
])
soc_pct = np.array([
    40, 43, 47, 51, 55, 59, 63, 70, 76, 81,
    85, 88, 91, 93, 95, 96, 97, 98, 99, 99.3,
    99.5, 99.7, 99.9,100,
])
df_taper = pd.DataFrame({"Time": time_min, "SoC": soc_pct})
df_taper.to_excel(DATA_DIR / "time_soc.xlsx", index=False)
print("✓  data/time_soc.xlsx written")

# ─────────────────────────────────────────────
# 3.  v2g_params.xlsx  – V2G / battery parameters
#     (used by the new V2G optimisation module)
# ─────────────────────────────────────────────
# Sheet 1: battery & charging hardware
battery_params = pd.DataFrame({
    "Parameter": [
        "BatteryCapacity_kWh",
        "UsableBatteryCap_kWh",
        "SOC_min_pct",           # cold-chain reserve floor
        "SOC_max_pct",           # cycle-protection ceiling
        "ChargePower_kW",
        "DischargePower_V2G_kW",
        "ChargeEfficiency",
        "DischargeEfficiency",
        "DegradationCost_EUR_kWh",  # baseline €/kWh cycled
    ],
    "Value": [82, 65.6, 20, 95, 22, 11, 0.92, 0.92, 0.07],
    "Unit": ["kWh","kWh","%","%","kW","kW","-","-","€/kWh"],
    "Note": [
        "S.KOe COOL total",
        "80% of total (SoC 20-95%)",
        "Cold-chain reserve – cannot discharge below",
        "Cycle protection – do not charge above",
        "CCS/Type2 AC max",
        "ISO 15118-20 V2G max",
        "Round-trip half-cycle",
        "Round-trip half-cycle",
        "Baseline – see sensitivity sheet",
    ],
})

# Sheet 2: EPEX-style V2G price profile (15-min, 96 slots/day)
t_slots = np.arange(96)
hours = t_slots / 4.0

# Buy price: German all-in tariff including levies (€/kWh)
buy_price = np.where(
    (hours >= 6) & (hours < 8),   0.28,   # morning ramp
    np.where(
    (hours >= 8) & (hours < 11),  0.30,   # morning peak
    np.where(
    (hours >= 11)& (hours < 14),  0.24,   # midday
    np.where(
    (hours >= 14)& (hours < 16),  0.22,   # afternoon valley
    np.where(
    (hours >= 16)& (hours < 20),  0.33,   # evening peak  ← V2G window
    np.where(
    (hours >= 20)& (hours < 22),  0.26,   # evening ramp-down
    0.16                                   # night valley
    ))))))

# V2G sell price = buy_price + balancing premium during peak
v2g_premium = np.where((hours >= 16) & (hours < 20), 0.132, 0.0)
v2g_price = buy_price + v2g_premium

df_prices_15min = pd.DataFrame({
    "Slot":       t_slots,
    "Hour":       hours,
    "BuyPrice_EUR_kWh":  np.round(buy_price, 4),
    "V2G_Price_EUR_kWh": np.round(v2g_price, 4),
})

# Sheet 3: degradation sensitivity sweep
deg_costs   = [0.02, 0.035, 0.05, 0.07, 0.10, 0.12, 0.15]
df_deg_sens = pd.DataFrame({
    "DegCost_EUR_kWh": deg_costs,
    "Description": [
        "Optimistic (future solid-state)",
        "Mid-term improved cell",
        "Near-future target",
        "Baseline (current NMC)",
        "Conservative",
        "High degradation",
        "Break-even threshold",
    ],
})

# Sheet 4: depot dwell profiles (plug-in windows per day)
dwell_profiles = pd.DataFrame({
    "Profile":      ["NightOnly", "NightOnly", "Extended", "Extended"],
    "PlugIn_Hour":  [21.0,  0.0, 21.0, 12.0],
    "PlugOut_Hour": [31.0,  7.0, 31.0, 18.0],   # >24 wraps to next day
    "Note": [
        "Night dwell 21:00→07:00 (part 1)",
        "Night dwell 21:00→07:00 (part 2 – midnight crossing)",
        "Extended: night 21:00→07:00",
        "Extended: midday stop 12:00→18:00",
    ],
})

with pd.ExcelWriter(DATA_DIR / "v2g_params.xlsx", engine="openpyxl") as writer:
    battery_params.to_excel(writer, sheet_name="BatteryParams",   index=False)
    df_prices_15min.to_excel(writer, sheet_name="Prices15min",    index=False)
    df_deg_sens.to_excel(writer,     sheet_name="DegSensitivity", index=False)
    dwell_profiles.to_excel(writer,  sheet_name="DwellProfiles",  index=False)
print("✓  data/v2g_params.xlsx written")

# ─────────────────────────────────────────────
# 4.  telematics_template.xlsx  – placeholder for
#     real TrailerConnect data (uploaded later)
# ─────────────────────────────────────────────
n = 96  # 24 hours × 15-min slots
slots = pd.date_range("2024-01-15 00:00", periods=n, freq="15min")
np.random.seed(42)

# Synthetic TRU load: sinusoidal + noise, 1.5–4.2 kW
tru_load = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(n) / 96 + np.pi) \
           + np.random.normal(0, 0.2, n)
tru_load = np.clip(tru_load, 1.5, 4.2)

# Synthetic SoC: starts at 45%, charges overnight
soc = np.zeros(n)
soc[0] = 45.0
for i in range(1, n):
    h = slots[i].hour + slots[i].minute / 60
    if (h >= 21) or (h < 7):        # plugged in overnight
        soc[i] = min(95, soc[i-1] + 3.5)
    elif 12 <= h < 18:               # midday stop
        soc[i] = min(95, soc[i-1] + 1.8)
    else:
        soc[i] = max(20, soc[i-1] - 1.2)  # driving / standby

plugged = ((slots.hour >= 21) | (slots.hour < 7) |
           ((slots.hour >= 12) & (slots.hour < 18))).astype(int)

df_telem = pd.DataFrame({
    "Timestamp":       slots,
    "SoC_pct":         np.round(soc, 1),
    "TRU_Load_kW":     np.round(tru_load, 3),
    "Plugged_In":      plugged,                # 1 = at depot charger
    "Ambient_Temp_C":  np.round(5 + 3*np.sin(2*np.pi*np.arange(n)/96), 1),
    "Setpoint_Temp_C": np.full(n, -18.0),      # frozen goods
    "TrailerID":       "TRL-001-SYNTHETIC",
    "Source":          "SYNTHETIC – replace with TrailerConnect export",
})
df_telem.to_excel(DATA_DIR / "telematics_template.xlsx", index=False)
print("✓  data/telematics_template.xlsx written  (replace with real data later)")

print("\n✅  All data files generated in ./data/")
