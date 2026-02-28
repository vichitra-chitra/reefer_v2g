# -------------------- imports & page config --------------------
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st
import backend as be

st.set_page_config(page_title="Trailer Charging Cost — Interactive", layout="wide")


# --- Simple Authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Access Restricted")
    st.write("Please enter your credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username == "admin" and password == "dontshare":
                st.session_state.authenticated = True
                st.success("Login successful! Reloading...")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()  # Prevent loading the rest of the app

# -------------------- MATLAB-aligned defaults --------------------
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
    "Arrival_HHMM": "16:30",
    "Departure_HHMM": "07:30",
    "MaxChargingPower_kW": 22.0,
    "ReeferCycleInit": "Continuous",  # Continuous | Start-Stop | NoReeferStationary
    "WinterMonths": 6,                 # Summer = 12 - Winter
}

# Init session state once
if "params" not in st.session_state:
    st.session_state.params = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


# ---------------------- INPUT PANEL (define BEFORE use) ----------------------
def render_input_panel():
    st.title("Reefer — Input Parameters")

    p = st.session_state.params  # shorthand
    
    # --- Now start the form ---
    with st.form(key="input_form", clear_on_submit=False):
        c1, c2, c3, cBtn = st.columns([1, 1, 1, 0.35])
        
        # Column 1: Battery + OBC
        with c1:
            st.subheader("Battery Details")
            p["BatteryCapacity_kWh"] = st.number_input("Battery Capacity (kWh)", value=float(p["BatteryCapacity_kWh"]), step=0.1)
            p["UsableBatteryCap_kWh"] = st.number_input("Usable Battery Capacity (kWh)", value=float(p["UsableBatteryCap_kWh"]), step=0.1)
            p["BatteryChargingEffi_pc"] = st.number_input("Battery Charging Efficiency (%)", value=float(p["BatteryChargingEffi_pc"]), step=0.1, min_value=0.0, max_value=100.0)

            st.subheader("OBC Details")
            p["OBC_Capacity_kW"] = st.number_input("OBC Capacity (kW)", value=float(p["OBC_Capacity_kW"]), step=0.1)
            p["OBC_UsableCapacity_kW"] = st.number_input("OBC Usable Capacity (kW)", value=float(p["OBC_UsableCapacity_kW"]), step=0.1)
            p["OBCEfficiency_pc"] = st.number_input("OBC Efficiency (%)", value=float(p["OBCEfficiency_pc"]), step=0.1, min_value=0.0, max_value=100.0)

        # --- Column 2: Arrival & Departure + Charging unit + Season split ---
        with c2:
            st.subheader("Arrival & Departure")
            p["Arrival_HHMM"] = st.text_input("Arrival Time (HH:MM)", value=p["Arrival_HHMM"])
            p["Departure_HHMM"] = st.text_input("Departure Time (HH:MM)", value=p["Departure_HHMM"])
        
            st.subheader("Charging Unit")
            p["MaxChargingPower_kW"] = st.number_input(
                "Charging Unit Max Power (kW)", value=float(p["MaxChargingPower_kW"]), step=0.1
            )
            
            st.subheader("Season Split")
            winter = st.slider("Winter months", 0, 12, int(p["WinterMonths"]))
            summer = 12 - winter
            st.slider("Summer months", 0, 12, summer, disabled=True )
        
            # Update params for later calculations
            p["WinterMonths"] = winter
            p["SummerMonths"] = summer  

        # Column 3: Seasonal SoC + Reefer Cycle
        with c3:
            st.subheader("Seasonal SoC")
            p["SOC_arrival_winter_pc"] = st.slider("SoC at arrival (Winter %)", 0, 100, int(p["SOC_arrival_winter_pc"]))
            p["SOC_arrival_summer_pc"] = st.slider("SoC at arrival (Summer %)", 0, 100, int(p["SOC_arrival_summer_pc"]))
            p["SOC_departure_target_pc"] = st.slider("SoC required at departure (%)", 0, 100, int(p["SOC_departure_target_pc"]))

            st.subheader("Reefer Cycle at Stationary")
            cycle_choice = st.radio("Select", ["Continuous", "Start-Stop", "Reefer OFF"],
                                    index={"Continuous": 0, "Start-Stop": 1, "NoReeferStationary": 2}.get(p["ReeferCycleInit"], 0))
            p["ReeferCycleInit"] = "NoReeferStationary" if cycle_choice == "Reefer OFF" else cycle_choice

        # Calculate button
        with cBtn:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Calculate", type="primary")

        if submitted:
            # validations
            if p["UsableBatteryCap_kWh"] <= 0 or p["UsableBatteryCap_kWh"] > p["BatteryCapacity_kWh"]:
                st.error("Usable Battery must be > 0 and ≤ Battery Capacity.")
                return
            if any(x < 0 or x > 100 for x in [p["BatteryChargingEffi_pc"], p["OBCEfficiency_pc"],
                                              p["SOC_arrival_winter_pc"], p["SOC_arrival_summer_pc"], p["SOC_departure_target_pc"]]):
                st.error("Efficiency and SoC values must be between 0 and 100%.")
                return

            st.session_state.params = p
            st.session_state.show_output = True
            st.rerun()


# -------------------- ROUTING: input first, then output --------------------
if not st.session_state.show_output:
    render_input_panel()
    st.stop()  # do not run the output logic below until user clicks Calculate
# ---------------------------------------------------------------------------

p = st.session_state.params

# Sidebar inputs in Output GUI (optional for live tweaking)
st.sidebar.title("Adjust Inputs (Quick Edit)")
p["Arrival_HHMM"] = st.sidebar.text_input("Arrival Time (HH:MM)", value=p["Arrival_HHMM"])
p["Departure_HHMM"] = st.sidebar.text_input("Departure Time (HH:MM)", value=p["Departure_HHMM"])
p["SOC_arrival_winter_pc"] = st.sidebar.slider("Winter SoC at arrival (%)", 0, 100, int(p["SOC_arrival_winter_pc"]))
p["SOC_arrival_summer_pc"] = st.sidebar.slider("Summer SoC at arrival (%)", 0, 100, int(p["SOC_arrival_summer_pc"]))
p["SOC_departure_target_pc"] = st.sidebar.slider("SoC required at departure (%)", 0, 100, int(p["SOC_departure_target_pc"]))
p["WinterMonths"] = st.sidebar.slider("Winter months", 0, 12, int(p["WinterMonths"]))
st.sidebar.caption(f"Summer months auto-set to **{12 - int(p['WinterMonths'])}**.")
cycle_choice = st.sidebar.radio("Reefer cycle", ["Continuous", "Start-Stop", "Reefer OFF"],
                                index={"Continuous":0,"Start-Stop":1,"NoReeferStationary":2}.get(p["ReeferCycleInit"], 0))
p["ReeferCycleInit"] = "NoReeferStationary" if cycle_choice == "Reefer OFF" else cycle_choice

# Automatically update session state when any sidebar widget changes
st.session_state.params = p

# ---- After Calculate: pull values from session_state.params ----

arr_str = p["Arrival_HHMM"]
dep_str = p["Departure_HHMM"]
soc_arr_w = int(p["SOC_arrival_winter_pc"])
soc_arr_s = int(p["SOC_arrival_summer_pc"])
soc_tgt   = int(p["SOC_departure_target_pc"])
w_months  = int(p["WinterMonths"])
s_months  = 12 - w_months
cycleUI   = "NoReeferStationary" if p["ReeferCycleInit"] == "NoReeferStationary" else p["ReeferCycleInit"]

# ---------------------- Load data (same as MATLAB) -----------------------------------
try:
    winterWD, summerWD = be.read_price_excel("avg_price.xlsx")
    tariff = be.get_tariff_params()
    winterALL = be.compose_all_in_price(winterWD, tariff)
    summerALL = be.compose_all_in_price(summerWD, tariff)
except Exception as e:
    st.error(f"Cannot read avg_price.xlsx: {e}")
    st.stop()

try:
    socDF = be.read_taper_table("time_soc.xlsx")
except Exception as e:
    st.error(f"Cannot read time_soc.xlsx: {e}")
    st.stop()

# ---------------------- EV defaults ---------------------------------------------------
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
)
EV.finalize()

# ---------------------- Time vector, taper, reefer minute loads -----------------------
t_arr, t_dep, t, dt_hr = be.build_time_vector(EV.ArrivalTime_HHMM, EV.DepartureTime_HHMM)
if len(t) == 0:
    st.warning("Arrival/Departure times produce an empty parked window.")
    st.stop()

eff_frac2 = max(np.finfo(float).eps, (EV.BatteryChargingEffi_pc * EV.OBCEfficiency_pc) / 10000.0)
soc_bp, Pcap_grid_bp_kW = be.build_taper_lookup(socDF, EV, eff_frac2)

PF_reefer = 0.75
kVA_refr_cap = 19.765
if cycleUI == "NoReeferStationary":
    P_refr_min_kW = np.zeros(len(t))
    kVA_refr_min = np.zeros(len(t))
else:
    P_refr_min_kW, kVA_refr_min = be.build_reefer_stationary_minute_trace(
        cycleUI, t, t_arr, t_dep, 10, PF_reefer, kVA_refr_cap
    )

# ---------------------- Reefer preview (in sidebar) -----------------------------------
with st.sidebar.expander(f"Reefer Cycle — {'Reefer OFF' if cycleUI=='NoReeferStationary' else cycleUI}", expanded=True):
    # ≈ 2h preview from arrival (10s resolution)
    t10_prev = [t_arr + timedelta(seconds=i*10) for i in range(int((70*60)/10))]
    if cycleUI == "NoReeferStationary":
        P_reefer_1h_kW = np.zeros(len(t10_prev))
    else:
        P_reefer_1h_kW = be.get_reefer_cycle_trace(cycleUI, len(t10_prev), dt_sec=10)
    figPrev, axPrev = plt.subplots(figsize=(4.5, 2.0))
    axPrev.step(t10_prev, P_reefer_1h_kW, where='post', color=(0.00,0.45,0.10), lw=1.8)
    axPrev.set_xlabel("Time (HH:MM)"); axPrev.set_ylabel("kW")
    yMax = max(1.0, float(np.max(P_reefer_1h_kW))*1.2); axPrev.set_ylim(0, yMax)
    axPrev.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(figPrev, use_container_width=True)

# ---------------------- Season-specific arrival SOC ---------------------------
EVW = be.EVParams(**{k: getattr(EV, k) for k in EV.__dataclass_fields__}); EVW.finalize()
EVS = be.EVParams(**{k: getattr(EV, k) for k in EV.__dataclass_fields__}); EVS.finalize()
EVW.CurrentSOC_kWh = min(EV.UsableBatteryCap_kWh, max(0.0, EV.UsableBatteryCap_kWh * (soc_arr_w/100.0)))
EVS.CurrentSOC_kWh = min(EV.UsableBatteryCap_kWh, max(0.0, EV.UsableBatteryCap_kWh * (soc_arr_s/100.0)))

target_kWh = EV.UsableBatteryCap_kWh * (soc_tgt/100.0)
needW_kWh = max(0.0, target_kWh - EVW.CurrentSOC_kWh)
needS_kWh = max(0.0, target_kWh - EVS.CurrentSOC_kWh)

# ---------------------- Plans: smart vs baseline --------------------------------------
Wsmart = be.plan_smart(t, dt_hr, winterALL, EVW, soc_bp, Pcap_grid_bp_kW, needW_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
Wbase  = be.plan_baseline(t, dt_hr, Wsmart["price_min"], EVW, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)
Ssmart = be.plan_smart(t, dt_hr, summerALL, EVS, soc_bp, Pcap_grid_bp_kW, needS_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
Sbase  = be.plan_baseline(t, dt_hr, Ssmart["price_min"], EVS, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)

# ---------------------- Main panel: BIG graphs ----------------------------------------
st.markdown("## Dumb vs Smart Charging of Reefer Trailer")

# Build mapping to 24h canvas
t0 = t_arr.replace(hour=0, minute=0, second=0, microsecond=0)
t24 = [t0 + timedelta(minutes=i) for i in range(1440)]
idx = [int(((ti - t0).total_seconds()/60.0) % 1440) for ti in t]

eW24 = np.array(winterALL).reshape(24); eS24 = np.array(summerALL).reshape(24)
eWst = np.concatenate([eW24, eW24[-1:]]); eSst = np.concatenate([eS24, eS24[-1:]])
tp = [t0 + timedelta(hours=h) for h in range(25)]
hMid = [t0 + timedelta(hours=h, minutes=30) for h in range(24)]

# --- ADD: numeric hours arrays to avoid dates on X-axis
h24 = np.arange(1440) / 60.0              # 0.00 ... 23.9833 (minute resolution)
tp_hours = np.arange(25)                  # 0,1,...,24 (for hourly price band)
hMid_hours = np.arange(24) + 0.5          # mid points for hour labels (0.5,1.5,...,23.5)

P_norm_W = np.zeros(1440); P_smart_W = np.zeros(1440)
P_norm_S = np.zeros(1440); P_smart_S = np.zeros(1440)
SOC_norm_W = np.full(1440, np.nan); SOC_smart_W = np.full(1440, np.nan)
SOC_norm_S = np.full(1440, np.nan); SOC_smart_S = np.full(1440, np.nan)

P_norm_W[idx] = Wbase["P_trace"]; P_smart_W[idx] = Wsmart["P_trace"]
P_norm_S[idx] = Sbase["P_trace"]; P_smart_S[idx] = Ssmart["P_trace"]
SOC_norm_W[idx] = 100.0 * Wbase["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_smart_W[idx] = 100.0 * Wsmart["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_norm_S[idx] = 100.0 * Sbase["SOC_trace"]/EV.UsableBatteryCap_kWh
SOC_smart_S[idx] = 100.0 * Ssmart["SOC_trace"]/EV.UsableBatteryCap_kWh

colW_dumb = (0.70, 0.70, 0.70); colW_smart = (0.30, 0.75, 0.93)
colS_dumb = (0.70, 0.70, 0.70); colS_smart = (1.00, 0.60, 0.20)
colSoC_d = (0.50, 0.50, 0.50); colSoC_s = (0.00, 0.00, 0.00)
colPrLn = (0.00, 0.60, 0.00); colPrTx = (0.00, 0.45, 0.15)



import matplotlib.lines as mlines
# Thinner legend strokes to save space
legend_items = [
    mlines.Line2D([], [], color=colW_dumb,  lw=5, label='Dumb Power'),
    mlines.Line2D([], [], color=colW_smart, lw=5, label='Winter Smart Power'),
    mlines.Line2D([], [], color=colS_smart, lw=5, label='Summer Smart Power'),
    mlines.Line2D([], [], color=colSoC_d,   lw=1.5, label='SoC Dumb'),
    mlines.Line2D([], [], color=colSoC_s,   lw=1.5, label='SoC Smart'),
    mlines.Line2D([], [], color=colPrLn,    lw=1.5, label='Hourly Electricity Price'),
]

# Smaller figure height and tighter legend spacing
figLegend = plt.figure(figsize=(10, 0.3))  # width, height (inches)
figLegend.legend(
    handles=legend_items,
    loc='upper center',
    ncol=len(legend_items),
    frameon=False,
    prop={'size': 6},        # smaller font
    handlelength=1,        # shorter line samples
    handletextpad=0.6,       # less gap between line and text
    borderpad=0.2,           # tighter box padding
    labelspacing=0.4,        # less vertical spacing
    columnspacing=1.2        # tighter gap between columns
)

# Trim extra margins
figLegend.tight_layout(pad=0.05)

st.pyplot(figLegend, use_container_width=False)



# ---- Winter (full-width)
figW, axW = plt.subplots(figsize=(12, 3))
axW.fill_between(h24, P_norm_W, step='pre', color=colW_dumb, alpha=0.30, label='Dumb Power')
axW.fill_between(h24, P_smart_W, step='pre', color=colW_smart, alpha=0.35, label='Winter Smart Power')
axW.set_ylabel('Power (kW)')
axW.grid(True, linestyle=':', alpha=0.6)

leftYL = axW.get_ylim()
if leftYL[1]-leftYL[0] <= 0:
    axW.set_ylim(0, 1)
    leftYL = axW.get_ylim()
padFrac = 0.04; bandFrac = 0.12
rngW = leftYL[1]-leftYL[0]; yBase = leftYL[0]+padFrac*rngW; yTop = yBase+bandFrac*rngW
eMinW = float(np.min(eWst)); eMaxW = float(np.max(eWst))
yPriceW = (yBase + (eWst-eMinW)/(eMaxW-eMinW)*(yTop-yBase)) if eMaxW!=eMinW else np.full_like(eWst, yBase)

# PRICE BAND using hours on X
axW.step(tp_hours, yPriceW, where='post', color=colPrLn, lw=1.0, label='Hourly Electricity Price')

# SoC on twin axis (X still hours)
ax2 = axW.twinx()
ax2.plot(h24, SOC_norm_W, '-', color=colSoC_d, lw=1.1, label='SoC Dumb')
ax2.plot(h24, SOC_smart_W, '-', color=colSoC_s, lw=1.1, label='SoC Smart')
ax2.set_ylim(0, 100); ax2.set_ylabel('SoC (%)')

# X-axis: hours 0..24
axW.set_xlim(0, 24)
axW.set_xticks(np.arange(0, 25, 1))
axW.set_xlabel('Time of the day')

# Hourly price labels WITHOUT € using mid-hour positions
yTxtW = yBase + ((eW24 - eMinW)/(eMaxW-eMinW) if eMaxW!=eMinW else np.zeros(24))*(yTop-yBase)
for h in range(24):
    axW.text(hMid_hours[h], yTxtW[h], f"{eW24[h]:.2f}", color=colPrTx,
             fontsize=9, fontweight='bold', ha='center', va='bottom')

st.pyplot(figW, use_container_width=True)

# ---- Summer (full-width)
figS, axS = plt.subplots(figsize=(12, 3))

# Ensure left axis draws above the twin axis
axS.set_zorder(2)
ax2S = axS.twinx()
ax2S.set_zorder(1)
axS.patch.set_alpha(0)

# Areas below everything else so labels can sit above
axS.fill_between(h24, P_norm_S,  step='pre', color=colS_dumb,  alpha=0.30, zorder=0)
axS.fill_between(h24, P_smart_S, step='pre', color=colS_smart, alpha=0.30, zorder=0)

axS.set_ylabel('Power (kW)')
axS.grid(True, linestyle=':', alpha=0.6)

leftYL = axS.get_ylim()
if leftYL[1]-leftYL[0] <= 0:
    axS.set_ylim(0, 1)
    leftYL = axS.get_ylim()

rngS   = leftYL[1]-leftYL[0]
yBaseS = leftYL[0] + padFrac * rngS
yTopS  = yBaseS + bandFrac * rngS

eMinS = float(np.min(eSst)); eMaxS = float(np.max(eSst))
yPriceS = (yBaseS + (eSst - eMinS)/(eMaxS - eMinS) * (yTopS - yBaseS)) if eMaxS != eMinS else np.full_like(eSst, yBaseS)

# PRICE BAND (hours on X)
axS.step(tp_hours, yPriceS, where='post', color=colPrLn, lw=1.0)

# SoC twin axis (hours on X)
ax2S.plot(h24, SOC_norm_S,  '-', color=colSoC_d, lw=1.1)
ax2S.plot(h24, SOC_smart_S, '-', color=colSoC_s, lw=1.1)
ax2S.set_ylim(0, 100); ax2S.set_ylabel('SoC (%)')

# X-axis hours 0..24
axS.set_xlim(0, 24)
axS.set_xticks(np.arange(0, 25, 1))
axS.set_xlabel('Time of the day')

# Hourly price labels WITHOUT € (high zorder, unclipped)
yTxtS = yBaseS + ((eS24 - eMinS)/(eMaxS - eMinS) if eMaxS != eMinS else np.zeros(24)) * (yTopS - yBaseS)
# Optional small upward nudge: yTxtS = yTxtS + (yTopS - yBaseS) * 0.02
for h in range(24):
    axS.text(
        hMid_hours[h], yTxtS[h], f"{eS24[h]:.2f}",
        color=colPrTx, fontsize=9, fontweight='bold',
        ha='center', va='bottom', zorder=10, clip_on=False
    )

# Render figure
st.pyplot(figS, use_container_width=True)


rp = be.get_reefer_cost_params()
cost_fixed_W_EV = Wbase["energy_kWh"] * rp.FixedPrice_EUR_per_kWh
cost_fixed_S_EV = Sbase["energy_kWh"] * rp.FixedPrice_EUR_per_kWh
RW = be.compute_reefer_cost_scenarios(P_refr_min_kW, dt_hr, Wsmart["price_min"], rp)
RS = be.compute_reefer_cost_scenarios(P_refr_min_kW, dt_hr, Ssmart["price_min"], rp)
yearly_fixed = cost_fixed_W_EV*20*w_months + cost_fixed_S_EV*20*s_months
yearly_dumb  = Wbase['cost_EUR']*20*w_months + Sbase['cost_EUR']*20*s_months
yearly_smart = Wsmart['cost_EUR']*20*w_months + Ssmart['cost_EUR']*20*s_months
sav_smart_vs_fixed = yearly_fixed - yearly_smart
sav_smart_vs_dumb  = yearly_dumb - yearly_smart

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### Battery Charging Cost (€)")
    df_charge = pd.DataFrame([
        ["Energy needed (kWh)", f"{needW_kWh:.2f}", f"{needS_kWh:.2f}"],
        ["Fixed Price Charging", f"{cost_fixed_W_EV:.2f}", f"{cost_fixed_S_EV:.2f}"],
        ["Dumb Charging", f"{Wbase['cost_EUR']:.2f}", f"{Sbase['cost_EUR']:.2f}"],
        ["Smart Charging", f"{Wsmart['cost_EUR']:.2f}", f"{Ssmart['cost_EUR']:.2f}"],
    ], columns=["Metric", "Winter", "Summer"])
    st.table(df_charge)

with col2:
    st.markdown("#### Reefer Consumption (€)")
    df_trailer = pd.DataFrame([
        ["Energy used by trailer (kWh)", f"{RW['E_kWh']:.2f}", f"{RS['E_kWh']:.2f}"],
        ["Diesel powered", f"{RW['cost_diesel']:.2f}", f"{RS['cost_diesel']:.2f}"],
        ["Fixed electricity price", f"{RW['cost_fixed']:.2f}", f"{RS['cost_fixed']:.2f}"],
        ["Dumb Charging", f"{RW['cost_dynamic']:.2f}", f"{RS['cost_dynamic']:.2f}"],
        ["Smart Charging", f"{RW['cost_dynamic']:.2f}", f"{RS['cost_dynamic']:.2f}"],
    ], columns=["Metric", "Winter", "Summer"])
    st.table(df_trailer)

with col3:
    st.markdown("#### Yearly Values (€)")
    df_yearly = pd.DataFrame([
        ["Fixed Price Charging Cost", f"€{yearly_fixed:.2f}"],
        ["Dumb Charging Cost", f"€{yearly_dumb:.2f}"],
        ["Smart Charging Cost", f"€{yearly_smart:.2f}"],
        ["Savings (Smart vs Fixed)", f"€{sav_smart_vs_fixed:.2f}"],
        ["Savings (Smart vs Dumb)", f"€{sav_smart_vs_dumb:.2f}"],
    ], columns=["Metric", "Value"])
    st.table(df_yearly)



with st.expander("Understanding This Panel", expanded=False):
    st.write(
        f"""
1. Arrival & Departure: Times define the parked window.  
2. SoC: Battery % at arrival and the target % at departure.  
3. Winter vs Summer: Separate scenarios with seasonal prices.  
4. Dumb Charging: Charges at max power without price optimization.  
5. Smart Charging: Shifts charging to cheaper hours; still meets target.  
6. Reefer Cycle: Trailer refrigeration load while parked (Continuous/Start-Stop/OFF).  
7. Trailer Energy: kWh consumed by the reefer during the parked window.  
8. Cost Comparison: Charging & reefer costs under fixed vs dynamic pricing.  
9. Yearly Savings: Assumes 20 parked days/month; multiplies seasonal costs.  
10. Graph Colors: Blue = Winter Smart Power, Orange = Summer Smart Power; Gray = Dumb.  
11. Fixed electricity price: €{rp.FixedPrice_EUR_per_kWh:.2f} per kWh.  
12. Diesel price: €{rp.DieselPrice_EUR_per_L:.2f}/L; DG efficiency: {int(100*rp.Genset_efficiency_frac)}%; Energy density: {rp.Diesel_kWh_per_L:.1f} kWh/L.
"""
    )
