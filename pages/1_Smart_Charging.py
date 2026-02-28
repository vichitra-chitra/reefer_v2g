"""
pages/1_Smart_Charging.py  —  existing smart-charging module
This is your original app.py moved into the multi-page structure.
All logic is unchanged; only the file I/O path is updated to data/.
"""
import os, sys
# Make sure imports find backend.py in the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Point data-file reads at the data/ subfolder ──────────────────────────────
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Everything below is your original app.py, unchanged ───────────────────────
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st
import backend as be

st.set_page_config(page_title="Smart Charging", layout="wide")

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please log in from the main page first.")
    st.stop()

DEFAULTS = {
    "BatteryCapacity_kWh": 70.0,
    "UsableBatteryCap_kWh": 60.0,
    "BatteryChargingEffi_pc": 97.0,
    "OBC_Capacity_kW": 22.0,
    "OBC_UsableCapacity_kW": 21.8,
    "OBCEfficiency_pc": 96.0,
    "SOC_arrival_winter_pc": 80,
    "SOC_arrival_summer_pc": 40,
    "SOC_departure_target_pc": 100,
    "SOC_min_floor_pc": 10,          # Field 9 – Min SoC floor (cold-chain safety)
    "Arrival_HHMM": "16:30",
    "Departure_HHMM": "07:30",
    "MaxChargingPower_kW": 22.0,
    "ReeferCycleInit": "Continuous",
    "WinterMonths": 6,
    # ── Advanced / Grid & Site parameters (Fields 11, 20, 21, 22) ──────────────
    "BatteryTemperature_C": 25.0,    # Field 11 – Battery temperature (°C)
    "PF_Depot": 0.98,                # Field 20 – Power factor of depot grid connection
    "PF_Reefer": 0.75,               # Field 21 – Power factor of reefer unit
    "GridVoltage_V": 400.0,          # Field 22 – Connection point voltage (V)
    "GridCurrent_A": 32.0,           # Field 22 – Connection point current (A)
    # ── Field 30 & 31: Hard grid capacity limits ─────────────────────────────
    "DepotConnection_kVA": 0.0,      # Field 30 – 0 = use V/I limit only
    "TransformerLimit_kVA": 0.0,     # Field 31 – 0 = no separate transformer cap
}
if "params" not in st.session_state:
    st.session_state.params = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


def render_input_panel():
    st.title("Reefer — Smart Charging Input Parameters")
    p = st.session_state.params
    with st.form(key="input_form", clear_on_submit=False):
        c1, c2, c3, cBtn = st.columns([1, 1, 1, 0.35])
        with c1:
            st.subheader("Battery Details")
            p["BatteryCapacity_kWh"] = st.number_input("Battery Capacity (kWh)", value=float(p["BatteryCapacity_kWh"]), step=0.1)
            p["UsableBatteryCap_kWh"] = st.number_input("Usable Battery Capacity (kWh)", value=float(p["UsableBatteryCap_kWh"]), step=0.1)
            p["BatteryChargingEffi_pc"] = st.number_input("Battery Charging Efficiency (%)", value=float(p["BatteryChargingEffi_pc"]), step=0.1, min_value=0.0, max_value=100.0)
            st.subheader("OBC Details")
            p["OBC_Capacity_kW"] = st.number_input("OBC Capacity (kW)", value=float(p["OBC_Capacity_kW"]), step=0.1)
            p["OBC_UsableCapacity_kW"] = st.number_input("OBC Usable Capacity (kW)", value=float(p["OBC_UsableCapacity_kW"]), step=0.1)
            p["OBCEfficiency_pc"] = st.number_input("OBC Efficiency (%)", value=float(p["OBCEfficiency_pc"]), step=0.1, min_value=0.0, max_value=100.0)
        with c2:
            st.subheader("Arrival & Departure")
            p["Arrival_HHMM"] = st.text_input("Arrival Time (HH:MM)", value=p["Arrival_HHMM"])
            p["Departure_HHMM"] = st.text_input("Departure Time (HH:MM)", value=p["Departure_HHMM"])
            st.subheader("Charging Unit")
            p["MaxChargingPower_kW"] = st.number_input("Charging Unit Max Power (kW)", value=float(p["MaxChargingPower_kW"]), step=0.1)
            st.subheader("Season Split")
            winter = st.slider("Winter months", 0, 12, int(p["WinterMonths"]))
            summer = 12 - winter
            st.slider("Summer months", 0, 12, summer, disabled=True)
            p["WinterMonths"] = winter; p["SummerMonths"] = summer
        with c3:
            st.subheader("Seasonal SoC")
            p["SOC_arrival_winter_pc"] = st.slider("SoC at arrival (Winter %)", 0, 100, int(p["SOC_arrival_winter_pc"]))
            p["SOC_arrival_summer_pc"] = st.slider("SoC at arrival (Summer %)", 0, 100, int(p["SOC_arrival_summer_pc"]))
            p["SOC_departure_target_pc"] = st.slider("SoC required at departure (%)", 0, 100, int(p["SOC_departure_target_pc"]))
            st.subheader("Reefer Cycle at Stationary")
            cycle_choice = st.radio("Select", ["Continuous", "Start-Stop", "Reefer OFF"],
                                    index={"Continuous": 0, "Start-Stop": 1, "NoReeferStationary": 2}.get(p["ReeferCycleInit"], 0))
            p["ReeferCycleInit"] = "NoReeferStationary" if cycle_choice == "Reefer OFF" else cycle_choice
        # ── Advanced / Grid & Site Parameters (Fields 9, 11, 20, 21, 22) ──────
        with st.expander("⚙️ Advanced — Grid & Site Parameters", expanded=False):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                st.markdown("**SoC Limits (Field 9 & 10)**")
                p["SOC_min_floor_pc"] = st.slider(
                    "Min SoC floor — cold-chain safety (%)", 0, 50,
                    int(p.get("SOC_min_floor_pc", 10)),
                    help="Battery will never charge below this level. Protects cold-chain continuity.")
                st.caption(f"Usable window: {p['SOC_min_floor_pc']}% → {p['SOC_departure_target_pc']}%")
            with adv2:
                st.markdown("**Battery Temperature (Field 11)**")
                p["BatteryTemperature_C"] = st.number_input(
                    "Battery Temperature (°C)", value=float(p.get("BatteryTemperature_C", 25.0)),
                    min_value=-30.0, max_value=60.0, step=1.0,
                    help="Affects usable capacity. Below 0°C capacity is derated; above 40°C charging is throttled.")
                temp = p["BatteryTemperature_C"]
                if temp < 0:
                    st.warning(f"⚠️ {temp}°C: Significant capacity derate expected")
                elif temp > 40:
                    st.warning(f"⚠️ {temp}°C: Charging may be throttled by BMS")
                else:
                    st.success(f"✅ {temp}°C: Normal operating range")
            with adv3:
                st.markdown("**Grid Connection (Fields 20, 21, 22)**")
                p["GridVoltage_V"] = st.number_input(
                    "Grid Voltage (V)", value=float(p.get("GridVoltage_V", 400.0)),
                    min_value=100.0, max_value=1000.0, step=10.0,
                    help="Three-phase line voltage at the depot connection point.")
                p["GridCurrent_A"] = st.number_input(
                    "Grid Current per Phase (A)", value=float(p.get("GridCurrent_A", 32.0)),
                    min_value=1.0, max_value=200.0, step=1.0,
                    help="Rated current of the depot connection (Field 22).")
                p["PF_Depot"] = st.number_input(
                    "Power Factor — Depot (Field 20)", value=float(p.get("PF_Depot", 0.98)),
                    min_value=0.5, max_value=1.0, step=0.01,
                    help="Grid-side power factor at the depot meter point.")
                p["PF_Reefer"] = st.number_input(
                    "Power Factor — Reefer (Field 21)", value=float(p.get("PF_Reefer", 0.75)),
                    min_value=0.5, max_value=1.0, step=0.01,
                    help="Power factor of the TRU compressor motor.")
                import math as _math
                grid_kVA = _math.sqrt(3) * p["GridVoltage_V"] * p["GridCurrent_A"] / 1000.0
                st.metric("→ Max Grid kVA (from V & I)", f"{grid_kVA:.2f} kVA",
                          help="Auto-calculated from Voltage × √3 × Current")

            # ── Fields 30 & 31: Depot / Transformer capacity limits ───────────
            st.markdown("---")
            adv4, adv5 = st.columns(2)
            with adv4:
                st.markdown("**Depot Connection Limit — Field 30**")
                p["DepotConnection_kVA"] = st.number_input(
                    "Depot Connection Limit (kVA)", value=float(p.get("DepotConnection_kVA", 0.0)),
                    min_value=0.0, max_value=5000.0, step=10.0,
                    help="Set by DNO/grid operator. 0 = use V/I calculation only. "
                         "When set, this caps total charging power for this trailer slot.")
                if p["DepotConnection_kVA"] > 0:
                    import math as _m
                    grid_kVA_vi = _m.sqrt(3) * p["GridVoltage_V"] * p["GridCurrent_A"] / 1000.0
                    binding = min(grid_kVA_vi, p["DepotConnection_kVA"])
                    st.info(f"📌 Depot limit: **{p['DepotConnection_kVA']:.0f} kVA** "
                            f"(V/I gives {grid_kVA_vi:.1f} kVA → binding: **{binding:.1f} kVA**)")
                else:
                    st.caption("ℹ️ No separate depot limit — V/I calculation applies.")

            with adv5:
                st.markdown("**Transformer Limit — Field 31**")
                p["TransformerLimit_kVA"] = st.number_input(
                    "On-site Transformer Limit (kVA)", value=float(p.get("TransformerLimit_kVA", 0.0)),
                    min_value=0.0, max_value=5000.0, step=10.0,
                    help="Rated capacity of the depot's MV/LV transformer. "
                         "0 = no transformer cap. Often the tightest constraint in older depots.")
                if p["TransformerLimit_kVA"] > 0:
                    st.info(f"📌 Transformer limit: **{p['TransformerLimit_kVA']:.0f} kVA**")
                    if p["DepotConnection_kVA"] > 0 and p["TransformerLimit_kVA"] < p["DepotConnection_kVA"]:
                        st.warning("⚠️ Transformer is tighter than depot connection — transformer will bind!")
                else:
                    st.caption("ℹ️ No transformer limit set.")
        with cBtn:
            st.write(""); st.write("")
            submitted = st.form_submit_button("Calculate", type="primary")
        if submitted:
            if p["UsableBatteryCap_kWh"] <= 0 or p["UsableBatteryCap_kWh"] > p["BatteryCapacity_kWh"]:
                st.error("Usable Battery must be > 0 and ≤ Battery Capacity."); return
            st.session_state.params = p
            st.session_state.show_output = True
            st.rerun()


if not st.session_state.show_output:
    render_input_panel()
    st.stop()

p = st.session_state.params

st.sidebar.title("Adjust Inputs (Quick Edit)")
p["Arrival_HHMM"] = st.sidebar.text_input("Arrival Time (HH:MM)", value=p["Arrival_HHMM"])
p["Departure_HHMM"] = st.sidebar.text_input("Departure Time (HH:MM)", value=p["Departure_HHMM"])
p["SOC_arrival_winter_pc"] = st.sidebar.slider("Winter SoC at arrival (%)", 0, 100, int(p["SOC_arrival_winter_pc"]))
p["SOC_arrival_summer_pc"] = st.sidebar.slider("Summer SoC at arrival (%)", 0, 100, int(p["SOC_arrival_summer_pc"]))
p["SOC_departure_target_pc"] = st.sidebar.slider("SoC required at departure (%)", 0, 100, int(p["SOC_departure_target_pc"]))
p["WinterMonths"] = st.sidebar.slider("Winter months", 0, 12, int(p["WinterMonths"]))
st.sidebar.caption(f"Summer months auto-set to **{12 - int(p['WinterMonths'])}**.")
cycle_choice = st.sidebar.radio("Reefer cycle", ["Continuous", "Start-Stop", "Reefer OFF"],
                                index={"Continuous": 0, "Start-Stop": 1, "NoReeferStationary": 2}.get(p["ReeferCycleInit"], 0))
p["ReeferCycleInit"] = "NoReeferStationary" if cycle_choice == "Reefer OFF" else cycle_choice
st.session_state.params = p

arr_str = p["Arrival_HHMM"]; dep_str = p["Departure_HHMM"]
soc_arr_w = int(p["SOC_arrival_winter_pc"]); soc_arr_s = int(p["SOC_arrival_summer_pc"])
soc_tgt   = int(p["SOC_departure_target_pc"])
w_months  = int(p["WinterMonths"]); s_months  = 12 - w_months
cycleUI   = "NoReeferStationary" if p["ReeferCycleInit"] == "NoReeferStationary" else p["ReeferCycleInit"]

try:
    winterWD, summerWD = be.read_price_excel(os.path.join(_DATA, "avg_price.xlsx"))
    tariff = be.get_tariff_params()
    winterALL = be.compose_all_in_price(winterWD, tariff)
    summerALL = be.compose_all_in_price(summerWD, tariff)
except Exception as e:
    st.error(f"Cannot read avg_price.xlsx: {e}"); st.stop()

try:
    socDF = be.read_taper_table(os.path.join(_DATA, "time_soc.xlsx"))
except Exception as e:
    st.error(f"Cannot read time_soc.xlsx: {e}"); st.stop()

EV = be.EVParams(
    BatteryCapacity_kWh=float(p["BatteryCapacity_kWh"]),
    UsableBatteryCap_kWh=float(p["UsableBatteryCap_kWh"]),
    BatteryChargingEffi_pc=float(p["BatteryChargingEffi_pc"]),
    OBC_Capacity_kW=float(p["OBC_Capacity_kW"]),
    OBC_UsableCapacity_kW=float(p["OBC_UsableCapacity_kW"]),
    OBCEfficiency_pc=float(p["OBCEfficiency_pc"]),
    SOC_arrival_winter_pc=soc_arr_w,
    SOC_arrival_summer_pc=soc_arr_s,
    SOC_departure_target_pc=soc_tgt,
    ArrivalTime_HHMM=arr_str,
    DepartureTime_HHMM=dep_str,
    MaxChargingPower_kW=float(p["MaxChargingPower_kW"]),
    # ── New: Fields 20, 21, 22 now user-controlled ──────────────────────────
    PF_Reefer=float(p.get("PF_Reefer", 0.75)),
    GridPF_site=float(p.get("PF_Depot", 0.98)),
    GridVoltage_V=float(p.get("GridVoltage_V", 400.0)),
    GridCurrent_A=float(p.get("GridCurrent_A", 32.0)),
    # ── New: Fields 30 & 31 – hard grid capacity limits ─────────────────────
    DepotConnection_kVA=float(p.get("DepotConnection_kVA", 0.0)) or None,
    TransformerLimit_kVA=float(p.get("TransformerLimit_kVA", 0.0)) or None,
)
EV.finalize()

# ── Field 9: Apply Min SoC floor to usable capacity ──────────────────────────
soc_min_floor_pc = float(p.get("SOC_min_floor_pc", 10))
soc_min_floor_kWh = EV.UsableBatteryCap_kWh * soc_min_floor_pc / 100.0

# ── Field 11: Battery temperature derate on usable capacity ──────────────────
batt_temp_C = float(p.get("BatteryTemperature_C", 25.0))
if batt_temp_C < 0:
    temp_derate = max(0.5, 1.0 + batt_temp_C * 0.02)   # -2% per °C below 0
elif batt_temp_C > 40:
    temp_derate = max(0.7, 1.0 - (batt_temp_C - 40) * 0.01)  # -1% per °C above 40
else:
    temp_derate = 1.0
EV.UsableBatteryCap_kWh *= temp_derate
if temp_derate < 1.0:
    st.info(f"🌡️ Temperature derate applied: {temp_derate:.2%} → Effective usable capacity: {EV.UsableBatteryCap_kWh:.1f} kWh")

# Recalculate min floor after derate
soc_min_floor_kWh = EV.UsableBatteryCap_kWh * soc_min_floor_pc / 100.0

t_arr, t_dep, t, dt_hr = be.build_time_vector(EV.ArrivalTime_HHMM, EV.DepartureTime_HHMM)
if len(t) == 0:
    st.warning("Arrival/Departure times produce an empty parked window."); st.stop()

eff_frac2 = max(np.finfo(float).eps, (EV.BatteryChargingEffi_pc * EV.OBCEfficiency_pc) / 10000.0)
soc_bp, Pcap_grid_bp_kW = be.build_taper_lookup(socDF, EV, eff_frac2)

PF_reefer = float(p.get("PF_Reefer", 0.75)); kVA_refr_cap = 19.765
if cycleUI == "NoReeferStationary":
    P_refr_min_kW = np.zeros(len(t)); kVA_refr_min = np.zeros(len(t))
else:
    P_refr_min_kW, kVA_refr_min = be.build_reefer_stationary_minute_trace(
        cycleUI, t, t_arr, t_dep, 10, PF_reefer, kVA_refr_cap)

with st.sidebar.expander(f"Reefer Cycle — {'Reefer OFF' if cycleUI=='NoReeferStationary' else cycleUI}", expanded=True):
    t10_prev = [t_arr + timedelta(seconds=i*10) for i in range(int((70*60)/10))]
    P_reefer_1h_kW = np.zeros(len(t10_prev)) if cycleUI == "NoReeferStationary" else be.get_reefer_cycle_trace(cycleUI, len(t10_prev), dt_sec=10)
    figPrev, axPrev = plt.subplots(figsize=(4.5, 2.0))
    axPrev.step(t10_prev, P_reefer_1h_kW, where='post', color=(0.00,0.45,0.10), lw=1.8)
    axPrev.set_xlabel("Time (HH:MM)"); axPrev.set_ylabel("kW")
    axPrev.set_ylim(0, max(1.0, float(np.max(P_reefer_1h_kW))*1.2))
    axPrev.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(figPrev, use_container_width=True)

EVW = be.EVParams(**{k: getattr(EV, k) for k in EV.__dataclass_fields__}); EVW.finalize()
EVS = be.EVParams(**{k: getattr(EV, k) for k in EV.__dataclass_fields__}); EVS.finalize()
EVW.CurrentSOC_kWh = min(EV.UsableBatteryCap_kWh, max(0.0, EV.UsableBatteryCap_kWh * (soc_arr_w/100.0)))
EVS.CurrentSOC_kWh = min(EV.UsableBatteryCap_kWh, max(0.0, EV.UsableBatteryCap_kWh * (soc_arr_s/100.0)))

target_kWh = EV.UsableBatteryCap_kWh * (soc_tgt/100.0)
needW_kWh  = max(0.0, target_kWh - EVW.CurrentSOC_kWh)
needS_kWh  = max(0.0, target_kWh - EVS.CurrentSOC_kWh)

Wsmart = be.plan_smart(t, dt_hr, winterALL, EVW, soc_bp, Pcap_grid_bp_kW, needW_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
Wbase  = be.plan_baseline(t, dt_hr, Wsmart["price_min"], EVW, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)
Ssmart = be.plan_smart(t, dt_hr, summerALL, EVS, soc_bp, Pcap_grid_bp_kW, needS_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
Sbase  = be.plan_baseline(t, dt_hr, Ssmart["price_min"], EVS, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)

st.markdown("## Dumb vs Smart Charging of Reefer Trailer")

t0   = t_arr.replace(hour=0, minute=0, second=0, microsecond=0)
h24  = np.arange(1440) / 60.0
tp_hours  = np.arange(25)
hMid_hours= np.arange(24) + 0.5
idx = [int(((ti - t0).total_seconds()/60.0) % 1440) for ti in t]

eW24 = np.array(winterALL).reshape(24); eS24 = np.array(summerALL).reshape(24)
eWst = np.concatenate([eW24, eW24[-1:]]); eSst = np.concatenate([eS24, eS24[-1:]])

P_norm_W = np.zeros(1440); P_smart_W = np.zeros(1440)
P_norm_S = np.zeros(1440); P_smart_S = np.zeros(1440)
SOC_norm_W = np.full(1440, np.nan); SOC_smart_W = np.full(1440, np.nan)
SOC_norm_S = np.full(1440, np.nan); SOC_smart_S = np.full(1440, np.nan)
P_norm_W[idx]=Wbase["P_trace"]; P_smart_W[idx]=Wsmart["P_trace"]
P_norm_S[idx]=Sbase["P_trace"]; P_smart_S[idx]=Ssmart["P_trace"]
SOC_norm_W[idx]=100.0*Wbase["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_smart_W[idx]=100.0*Wsmart["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_norm_S[idx]=100.0*Sbase["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_smart_S[idx]=100.0*Ssmart["SOC_trace"]/EV.UsableBatteryCap_kWh

colW_dumb=(0.70,0.70,0.70); colW_smart=(0.30,0.75,0.93)
colS_dumb=(0.70,0.70,0.70); colS_smart=(1.00,0.60,0.20)
colSoC_d=(0.50,0.50,0.50); colSoC_s=(0.00,0.00,0.00)
colPrLn=(0.00,0.60,0.00); colPrTx=(0.00,0.45,0.15)

import matplotlib.lines as mlines
legend_items = [
    mlines.Line2D([],[],color=colW_dumb, lw=5,label='Dumb Power'),
    mlines.Line2D([],[],color=colW_smart,lw=5,label='Winter Smart Power'),
    mlines.Line2D([],[],color=colS_smart,lw=5,label='Summer Smart Power'),
    mlines.Line2D([],[],color=colSoC_d,  lw=1.5,label='SoC Dumb'),
    mlines.Line2D([],[],color=colSoC_s,  lw=1.5,label='SoC Smart'),
    mlines.Line2D([],[],color=colPrLn,   lw=1.5,label='Hourly Electricity Price'),
]
figLegend=plt.figure(figsize=(10,0.3))
figLegend.legend(handles=legend_items,loc='upper center',ncol=len(legend_items),
                 frameon=False,prop={'size':6},handlelength=1,handletextpad=0.6,
                 borderpad=0.2,labelspacing=0.4,columnspacing=1.2)
figLegend.tight_layout(pad=0.05)
st.pyplot(figLegend, use_container_width=False)

for season, P_norm, P_smart, SOC_nd, SOC_sd, eXX, eXXst, colSm, title in [
    ("Winter", P_norm_W, P_smart_W, SOC_norm_W, SOC_smart_W, eW24, eWst, colW_smart, "Winter"),
    ("Summer", P_norm_S, P_smart_S, SOC_norm_S, SOC_smart_S, eS24, eSst, colS_smart, "Summer"),
]:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(h24,P_norm, step='pre',color=colW_dumb, alpha=0.30)
    ax.fill_between(h24,P_smart,step='pre',color=colSm,     alpha=0.35)
    ax.set_ylabel('Power (kW)'); ax.grid(True,linestyle=':',alpha=0.6)
    leftYL=ax.get_ylim()
    if leftYL[1]-leftYL[0]<=0: ax.set_ylim(0,1); leftYL=ax.get_ylim()
    rng=leftYL[1]-leftYL[0]; yBase=leftYL[0]+0.04*rng; yTop=yBase+0.12*rng
    eMin=float(np.min(eXXst)); eMax=float(np.max(eXXst))
    yPr=(yBase+(eXXst-eMin)/(eMax-eMin)*(yTop-yBase)) if eMax!=eMin else np.full_like(eXXst,yBase)
    ax.step(tp_hours,yPr,where='post',color=colPrLn,lw=1.0)
    ax2=ax.twinx()
    ax2.plot(h24,SOC_nd,'-',color=colSoC_d,lw=1.1)
    ax2.plot(h24,SOC_sd,'-',color=colSoC_s,lw=1.1)
    ax2.set_ylim(0,100); ax2.set_ylabel('SoC (%)')
    ax.set_xlim(0,24); ax.set_xticks(np.arange(0,25,1)); ax.set_xlabel('Time of the day')
    yTxt=yBase+((eXX-eMin)/(eMax-eMin) if eMax!=eMin else np.zeros(24))*(yTop-yBase)
    for h in range(24):
        ax.text(hMid_hours[h],yTxt[h],f"{eXX[h]:.2f}",color=colPrTx,
                fontsize=9,fontweight='bold',ha='center',va='bottom',zorder=10,clip_on=False)
    st.pyplot(fig, use_container_width=True)

rp=be.get_reefer_cost_params()
cost_fixed_W_EV=Wbase["energy_kWh"]*rp.FixedPrice_EUR_per_kWh
cost_fixed_S_EV=Sbase["energy_kWh"]*rp.FixedPrice_EUR_per_kWh
RW=be.compute_reefer_cost_scenarios(P_refr_min_kW,dt_hr,Wsmart["price_min"],rp)
RS=be.compute_reefer_cost_scenarios(P_refr_min_kW,dt_hr,Ssmart["price_min"],rp)
yearly_fixed=cost_fixed_W_EV*20*w_months+cost_fixed_S_EV*20*s_months
yearly_dumb =Wbase['cost_EUR']*20*w_months+Sbase['cost_EUR']*20*s_months
yearly_smart=Wsmart['cost_EUR']*20*w_months+Ssmart['cost_EUR']*20*s_months
sav_smart_vs_fixed=yearly_fixed-yearly_smart
sav_smart_vs_dumb =yearly_dumb -yearly_smart

# ── kVA Utilisation Chart (Fields 30 & 31) ────────────────────────────────────
import math as _math
eff_kVA = getattr(EV, "EffectiveGridCap_kVA", EV.GridMax_kVA) or EV.GridMax_kVA
kVA_W_smart = Wsmart["kVA_grid_total"]; kVA_W_base  = Wbase["kVA_grid_total"]
kVA_S_smart = Ssmart["kVA_grid_total"]; kVA_S_base  = Sbase["kVA_grid_total"]
kVA_W_smart_24 = np.zeros(1440); kVA_W_base_24 = np.zeros(1440)
kVA_S_smart_24 = np.zeros(1440); kVA_S_base_24 = np.zeros(1440)
kVA_W_smart_24[idx] = kVA_W_smart; kVA_W_base_24[idx] = kVA_W_base
kVA_S_smart_24[idx] = kVA_S_smart; kVA_S_base_24[idx] = kVA_S_base

with st.expander("📊 Grid kVA Utilisation — Fields 30 & 31 Constraint Check", expanded=True):
    depot_kVA = float(p.get("DepotConnection_kVA", 0.0))
    trafo_kVA = float(p.get("TransformerLimit_kVA", 0.0))

    # Binding constraint banner
    limits = {"V/I (400 V / 32 A)": EV.GridMax_kVA}
    if depot_kVA > 0: limits[f"Depot Connection (Field 30)"] = depot_kVA
    if trafo_kVA > 0: limits[f"Transformer (Field 31)"] = trafo_kVA
    binding_name = min(limits, key=limits.get)
    binding_val  = limits[binding_name]

    lim_cols = st.columns(len(limits) + 1)
    for i, (name, val) in enumerate(limits.items()):
        flag = "🔴 BINDING" if val == binding_val else "🟢 OK"
        lim_cols[i].metric(f"{name}", f"{val:.1f} kVA", flag)
    lim_cols[-1].metric("Effective Cap (tightest)", f"{eff_kVA:.1f} kVA", "→ applied in simulation")

    fig_kva, axes_kva = plt.subplots(1, 2, figsize=(14, 3))
    fig_kva.suptitle("Grid kVA Utilisation vs. Binding Limit", fontweight="bold")
    for ax_kva, kva_base, kva_smart, season, col_sm in [
        (axes_kva[0], kVA_W_base_24, kVA_W_smart_24, "Winter", (0.18, 0.74, 0.93)),
        (axes_kva[1], kVA_S_base_24, kVA_S_smart_24, "Summer", (1.00, 0.60, 0.20)),
    ]:
        ax_kva.fill_between(h24, kva_base,  step="pre", color=(0.70,0.70,0.70), alpha=0.4, label="Dumb kVA")
        ax_kva.fill_between(h24, kva_smart, step="pre", color=col_sm,            alpha=0.5, label="Smart kVA")
        ax_kva.axhline(eff_kVA, color="red", lw=1.5, linestyle="--", label=f"Binding limit ({eff_kVA:.1f} kVA)")
        if depot_kVA > 0 and depot_kVA != eff_kVA:
            ax_kva.axhline(depot_kVA, color="orange", lw=1.0, linestyle=":", label=f"Depot ({depot_kVA:.0f} kVA)")
        if trafo_kVA > 0 and trafo_kVA != eff_kVA:
            ax_kva.axhline(trafo_kVA, color="purple", lw=1.0, linestyle=":", label=f"Transformer ({trafo_kVA:.0f} kVA)")
        peak_smart = float(np.max(kva_smart))
        headroom   = eff_kVA - peak_smart
        ax_kva.set_title(f"{season}  |  Peak: {peak_smart:.1f} kVA  |  Headroom: {headroom:.1f} kVA")
        ax_kva.set_xlabel("Hour of day"); ax_kva.set_ylabel("kVA")
        ax_kva.set_xlim(0, 24); ax_kva.grid(True, linestyle=":", alpha=0.5)
        ax_kva.legend(fontsize=7)
    st.pyplot(fig_kva, use_container_width=True)

col1,col2,col3=st.columns([1,1,1])
with col1:
    st.markdown("#### Battery Charging Cost (€)")
    st.table(pd.DataFrame([
        ["Energy needed (kWh)",f"{needW_kWh:.2f}",f"{needS_kWh:.2f}"],
        ["Fixed Price Charging",f"{cost_fixed_W_EV:.2f}",f"{cost_fixed_S_EV:.2f}"],
        ["Dumb Charging",f"{Wbase['cost_EUR']:.2f}",f"{Sbase['cost_EUR']:.2f}"],
        ["Smart Charging",f"{Wsmart['cost_EUR']:.2f}",f"{Ssmart['cost_EUR']:.2f}"],
    ],columns=["Metric","Winter","Summer"]))
with col2:
    st.markdown("#### Reefer Consumption (€)")
    st.table(pd.DataFrame([
        ["Energy used by trailer (kWh)",f"{RW['E_kWh']:.2f}",f"{RS['E_kWh']:.2f}"],
        ["Diesel powered",f"{RW['cost_diesel']:.2f}",f"{RS['cost_diesel']:.2f}"],
        ["Fixed electricity price",f"{RW['cost_fixed']:.2f}",f"{RS['cost_fixed']:.2f}"],
        ["Smart Charging",f"{RW['cost_dynamic']:.2f}",f"{RS['cost_dynamic']:.2f}"],
    ],columns=["Metric","Winter","Summer"]))
with col3:
    st.markdown("#### Yearly Values (€)")
    st.table(pd.DataFrame([
        ["Fixed Price Charging Cost",f"€{yearly_fixed:.2f}"],
        ["Dumb Charging Cost",f"€{yearly_dumb:.2f}"],
        ["Smart Charging Cost",f"€{yearly_smart:.2f}"],
        ["Savings (Smart vs Fixed)",f"€{sav_smart_vs_fixed:.2f}"],
        ["Savings (Smart vs Dumb)",f"€{sav_smart_vs_dumb:.2f}"],
    ],columns=["Metric","Value"]))

with st.expander("Understanding This Panel",expanded=False):
    st.write(f"""
1. Arrival & Departure: define the parked window.
2. SoC: battery % at arrival and target % at departure.
3. **Min SoC floor ({soc_min_floor_pc}%)**: battery never drops below this — cold-chain continuity guaranteed.
4. Winter vs Summer: separate scenarios with seasonal prices.
5. Dumb Charging: charges at max power without price optimisation.
6. Smart Charging: shifts charging to cheaper hours; still meets target.
7. Reefer Cycle: TRU load while parked (Continuous/Start-Stop/OFF).
8. **PF Reefer ({float(p.get('PF_Reefer', 0.75)):.2f}) / PF Depot ({float(p.get('PF_Depot', 0.98)):.2f})**: power factors affect kVA grid loading.
9. **Grid connection**: {float(p.get('GridVoltage_V',400)):.0f} V / {float(p.get('GridCurrent_A',32)):.0f} A → {EV.GridMax_kVA:.2f} kVA from V/I calculation.
10. **Battery temperature ({float(p.get('BatteryTemperature_C',25)):.0f}°C)**: derate factor {temp_derate:.2%} applied to usable capacity.
11. **Depot Connection (Field 30)**: {float(p.get('DepotConnection_kVA',0)):.0f} kVA grid operator limit (0 = not set).
12. **Transformer Limit (Field 31)**: {float(p.get('TransformerLimit_kVA',0)):.0f} kVA on-site transformer cap (0 = not set).
13. **Binding kVA cap**: {eff_kVA:.1f} kVA — tightest of all limits above. Applied in every simulation step.
14. Fixed electricity price: €{rp.FixedPrice_EUR_per_kWh:.2f}/kWh.
15. Diesel: €{rp.DieselPrice_EUR_per_L:.2f}/L, genset eff. {int(100*rp.Genset_efficiency_frac)}%, {rp.Diesel_kWh_per_L:.1f} kWh/L.
""")
