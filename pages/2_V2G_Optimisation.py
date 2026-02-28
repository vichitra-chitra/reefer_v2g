# pages/2_V2G_Optimisation.py
# ──────────────────────────────────────────────────────────────────────────────
# Vehicle-to-Grid MILP Optimisation  (Scenario A / B / C + Sensitivity)
# References:
#   Biedenbach & Strunz (2024) Multi-Use Depot Optimization, WEVJ 15, 84
#   Agora Verkehrswende (2025) Bidirektionales Laden
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backend as be

st.set_page_config(page_title="V2G Optimisation", layout="wide")

# ── Auth (reuse session state from app.py) ────────────────────────────────────
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please log in from the main page first.")
    st.stop()

st.title("⚡ V2G Optimisation  —  S.KOe COOL Reefer Trailer")
st.caption(
    "MILP model based on Biedenbach & Strunz (2024) | "
    "Regulatory framework: Agora Verkehrswende (2025)"
)

# ── Sidebar: V2G parameters ───────────────────────────────────────────────────
st.sidebar.title("V2G Parameters")
st.sidebar.caption("Adjust and press **Run Optimisation**")

with st.sidebar.expander("🔋 Battery (S.KOe COOL)", expanded=True):
    batt_cap   = st.number_input("Total capacity (kWh)",    value=82.0, step=1.0)
    usable_pct = st.slider("Usable window (% of total)", 50, 95, 80)
    soc_min    = st.slider("SoC floor — cold-chain reserve (%)", 5, 40,  20)
    soc_max    = st.slider("SoC ceiling — cycle protection (%)", 80, 99, 95)
    soc_init   = st.slider("SoC at plug-in (%)", 10, 90, 45)
    soc_final  = st.slider("SoC required at departure (%)", 60, 99, 80)

with st.sidebar.expander("⚙️ Charging Hardware", expanded=True):
    p_charge    = st.number_input("Max charge power (kW) — ISO 15118",  value=22.0, step=1.0)
    p_discharge = st.number_input("Max V2G discharge power (kW)",        value=11.0, step=1.0)
    eta_c       = st.slider("Charge efficiency (%)", 80, 99, 92) / 100.0
    eta_d       = st.slider("Discharge efficiency (%)", 80, 99, 92) / 100.0

with st.sidebar.expander("🔌 Grid Limits — Fields 30 & 31", expanded=False):
    depot_kVA = st.number_input(
        "Depot Connection Limit (kVA) — Field 30", value=0.0, step=10.0, min_value=0.0,
        help="Grid operator / DNO limit on total depot kVA. 0 = no separate limit.")
    trafo_kVA = st.number_input(
        "Transformer Limit (kVA) — Field 31", value=0.0, step=10.0, min_value=0.0,
        help="On-site MV/LV transformer rated capacity. 0 = no constraint.")
    import math as _m
    if depot_kVA > 0 or trafo_kVA > 0:
        active_limits = []
        if depot_kVA > 0: active_limits.append(f"Depot: {depot_kVA:.0f} kVA")
        if trafo_kVA > 0: active_limits.append(f"Trafo: {trafo_kVA:.0f} kVA")
        binding = min([x for x in [depot_kVA, trafo_kVA] if x > 0])
        kw_cap  = binding * 0.95
        st.sidebar.warning(
            f"⚡ Active limits: {' | '.join(active_limits)}\n\n"
            f"→ Binding: **{binding:.0f} kVA** ≈ **{kw_cap:.1f} kW**\n\n"
            f"Charge & discharge power capped at {kw_cap:.1f} kW in MILP."
        )
    else:
        st.sidebar.caption("ℹ️ No grid limits — hardware power rating applies.")

with st.sidebar.expander("💸 Economics", expanded=True):
    deg_cost = st.number_input(
        "Degradation cost (€/kWh cycled)", value=0.07, step=0.01, format="%.3f",
        help="Baseline: 0.07 €/kWh (NMC cell). Target future: 0.035 €/kWh"
    )
    fleet_n = st.number_input("Fleet size (trailers)", value=10, min_value=1, max_value=500, step=5)

dwell_profile = st.sidebar.selectbox(
    "Depot dwell profile",
    ["Extended (night + midday stop)", "Night Only (21:00–07:00)"],
    help="Extended enables V2G during evening price peak (16–20h)"
)
dwell_key = "Extended" if "Extended" in dwell_profile else "NightOnly"

run_btn = st.sidebar.button("▶️ Run Optimisation", type="primary")

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@st.cache_data(show_spinner="Loading V2G parameters…")
def load_v2g_data(data_dir):
    path = os.path.join(data_dir, "v2g_params.xlsx")
    if not os.path.exists(path):
        st.error("❌  data/v2g_params.xlsx not found. Run `python generate_data.py` first.")
        st.stop()
    return be.read_v2g_params(path)

try:
    v2g_base, df_prices, df_deg_sens, df_dwell = load_v2g_data(DATA_DIR)
except Exception as e:
    st.error(f"Data load error: {e}"); st.stop()

# ── Check for real telematics ─────────────────────────────────────────────────
telem_path = os.path.join(DATA_DIR, "telematics_real.xlsx")
telem_note = "🔬 Using **synthetic** data. Upload `telematics_real.xlsx` to data/ for real analysis."
has_real_telem = os.path.exists(telem_path)
if has_real_telem:
    telem_note = "✅ Real telematics data detected and loaded."
st.info(telem_note)

# ── Build V2GParams from sidebar ──────────────────────────────────────────────
v2g_params = be.V2GParams(
    battery_capacity_kWh  = batt_cap,
    usable_capacity_kWh   = batt_cap * usable_pct / 100.0,
    soc_min_pct           = soc_min,
    soc_max_pct           = soc_max,
    charge_power_kW       = p_charge,
    discharge_power_kW    = p_discharge,
    eta_charge            = eta_c,
    eta_discharge         = eta_d,
    deg_cost_eur_kwh      = deg_cost,
    # ── Fields 30 & 31: hard grid capacity limits ────────────────────────────
    depot_connection_kVA  = depot_kVA,
    transformer_limit_kVA = trafo_kVA,
)

# ── Run on button OR show placeholder ─────────────────────────────────────────
if not run_btn and "v2g_results" not in st.session_state:
    st.markdown(
        """
        ### 👈 Configure parameters in the sidebar and press **Run Optimisation**

        This page implements the MILP model from **Biedenbach & Strunz (2024)**,
        adapted for the S.KOe COOL reefer trailer:

        | Scenario | Description |
        |---|---|
        | **A – Dumb** | Uncontrolled: charge at max power whenever plugged in |
        | **B – Smart** | Price-optimal charging only (no V2G discharge) |
        | **C – Smart + V2G** | Full bidirectional: buy cheap, sell peak (V2G) |

        **Battery degradation** is modelled as an opportunity cost per kWh cycled
        (Agora, 2025 / Biedenbach eq. 6).  Sensitivity sweep covers 0.02 – 0.15 €/kWh.
        """
    )
    st.stop()

# Run optimisation
if run_btn or "v2g_results" in st.session_state:
    if run_btn:
        with st.spinner("Running MILP optimisation…"):
            results = be.run_all_v2g_scenarios(
                v2g_params, df_prices, df_deg_sens,
                dwell_profile = dwell_key,
                soc_init_pct  = soc_init,
                soc_final_pct = soc_final,
            )
            st.session_state.v2g_results = results
            st.session_state.v2g_params  = v2g_params
    else:
        results = st.session_state.v2g_results

    A = results["A_Dumb"]
    B = results["B_Smart"]
    C = results["C_V2G"]
    hours = np.arange(v2g_params.n_slots) * v2g_params.dt_h

    # ── Grid Constraint Status Banner ─────────────────────────────────────────
    import math as _m
    if depot_kVA > 0 or trafo_kVA > 0:
        _limits = {k: v for k, v in [
            ("Depot Connection (F30)", depot_kVA),
            ("Transformer (F31)",      trafo_kVA),
        ] if v > 0}
        binding_kVA = min(_limits.values())
        binding_kW  = binding_kVA * 0.95
        peak_c = float(np.max(C.p_charge))
        peak_d = float(np.max(C.p_discharge))
        st.info(
            f"**Grid Limits Active** — "
            + "  |  ".join(f"{k}: **{v:.0f} kVA**" for k, v in _limits.items())
            + f"  →  Binding: **{binding_kVA:.0f} kVA** ({binding_kW:.1f} kW)\n\n"
            f"Peak charge: **{peak_c:.1f} kW**  |  Peak V2G discharge: **{peak_d:.1f} kW**  "
            + ("✅ Within limits" if max(peak_c, peak_d) <= binding_kW + 0.5 else "⚠️ Exceeds limit — check solver")
        )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown("## 📊 Daily Economics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = C.cost_eur_day - A.cost_eur_day
        st.metric("Net Cost – Dumb",   f"€{A.cost_eur_day:.2f}/day",  "Reference")
    with c2:
        st.metric("Net Cost – Smart",  f"€{B.cost_eur_day:.2f}/day",
                  f"€{B.cost_eur_day-A.cost_eur_day:.2f} vs Dumb")
    with c3:
        st.metric("Net Cost – V2G",    f"€{C.cost_eur_day:.2f}/day",
                  f"€{C.cost_eur_day-A.cost_eur_day:.2f} vs Dumb")
    with c4:
        st.metric("V2G Revenue",       f"€{C.v2g_revenue_eur_day:.2f}/day",
                  f"{C.v2g_export_kwh_day:.1f} kWh exported")

    # Annualised
    days = 365
    st.markdown("#### Annualised (365 days)")
    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1: st.metric("Annual – Dumb",  f"€{A.cost_eur_day*days:,.0f}")
    with ca2: st.metric("Annual – Smart", f"€{B.cost_eur_day*days:,.0f}",
                         f"Save €{(A.cost_eur_day-B.cost_eur_day)*days:,.0f}")
    with ca3: st.metric("Annual – V2G",   f"€{C.cost_eur_day*days:,.0f}",
                         f"Save €{(A.cost_eur_day-C.cost_eur_day)*days:,.0f}")
    with ca4: st.metric("Annual V2G Rev.", f"€{C.v2g_revenue_eur_day*days:,.0f}",
                         f"{C.v2g_export_kwh_day*days/1000:.1f} MWh/yr")

    st.divider()

    # ── 6-panel figure ────────────────────────────────────────────────────────
    st.markdown("## 📈 Optimisation Charts")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("S.KOe COOL  —  V2G MILP Optimisation Results", fontsize=14, fontweight="bold")

    COL = {
        "dumb":   "#AAAAAA",
        "smart":  "#30BDE0",
        "v2g":    "#FF7700",
        "price":  "#008000",
        "soc_d":  "#888888",
        "soc_s":  "#000000",
        "tru":    "#AA0000",
        "plug":   "#CCEECC",
    }

    # ── Panel 1: Charging schedules ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.fill_between(hours, A.p_charge, step="pre", color=COL["dumb"],  alpha=0.5, label="A – Dumb")
    ax.fill_between(hours, B.p_charge, step="pre", color=COL["smart"], alpha=0.5, label="B – Smart")
    ax.fill_between(hours, C.p_charge, step="pre", color=COL["v2g"],   alpha=0.4, label="C – V2G")
    ax.step(hours, A.tru_load, where="post", color=COL["tru"], lw=1.0, linestyle="--", label="TRU Load")
    ax.set_title("(1) Charging Power by Scenario")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # ── Panel 2: V2G discharge vs price ──────────────────────────────────────
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.bar(hours, C.p_discharge, width=0.22, color=COL["v2g"], alpha=0.75, label="V2G Discharge")
    ax2.step(hours, C.price_v2g, where="post", color=COL["price"], lw=1.5, label="V2G Price")
    ax.fill_between(hours, 0, C.plugged * max(C.p_discharge.max(), 0.1),
                    step="pre", color=COL["plug"], alpha=0.2, label="Plugged In")
    ax.set_title("(2) V2G Discharge vs. Price Signal")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Discharge (kW)")
    ax2.set_ylabel("Price (€/kWh)"); ax2.tick_params(colors=COL["price"])
    ax.legend(loc="upper left",  fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # ── Panel 3: SoC trajectories ────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(hours, A.soc / v2g_params.usable_capacity_kWh * 100,
            color=COL["dumb"],  lw=1.5, label="A – Dumb", linestyle="--")
    ax.plot(hours, B.soc / v2g_params.usable_capacity_kWh * 100,
            color=COL["smart"], lw=2.0, label="B – Smart")
    ax.plot(hours, C.soc / v2g_params.usable_capacity_kWh * 100,
            color=COL["v2g"],   lw=2.0, label="C – V2G")
    ax.axhline(soc_min, color="red",    lw=1.0, linestyle=":", label=f"Floor {soc_min}%")
    ax.axhline(soc_max, color="orange", lw=1.0, linestyle=":", label=f"Ceiling {soc_max}%")
    ax.set_title("(3) State of Charge Trajectories")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("SoC (%)")
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # ── Panel 4: Policy scenario bar ────────────────────────────────────────
    ax = axes[1, 0]
    labels = ["A – Dumb", "B – Smart\n(no V2G)", "C – Smart\n+ V2G"]
    costs  = [A.cost_eur_day * days, B.cost_eur_day * days, C.cost_eur_day * days]
    revs   = [0, 0, C.v2g_revenue_eur_day * days]
    bars   = ax.bar(labels, costs, color=[COL["dumb"], COL["smart"], COL["v2g"]], alpha=0.85)
    ax.bar(labels, [-r for r in revs], bottom=costs,
           color="#006600", alpha=0.6, label="V2G Revenue (negative)")
    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"€{val:,.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_title("(4) Annual Net Cost per Trailer")
    ax.set_ylabel("€ / year"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: Fleet scaling ───────────────────────────────────────────────
    ax = axes[1, 1]
    fleet_sizes = [1, 5, 10, 25, 50, 100]
    df_fleet = be.compute_fleet_impact(C, fleet_sizes)
    ax.plot(df_fleet["Fleet_Size"], df_fleet["Peak_Charge_kW"],
            "o-", color=COL["smart"], label="Peak Charge (kW)", lw=2)
    ax.plot(df_fleet["Fleet_Size"], df_fleet["Peak_Discharge_kW"],
            "s--", color=COL["v2g"], label="Peak V2G (kW)", lw=2)
    ax2f = ax.twinx()
    ax2f.bar(df_fleet["Fleet_Size"], df_fleet["Annual_V2G_Revenue_EUR"] / 1000,
             width=3, color=COL["price"], alpha=0.4, label="Annual V2G Rev. (k€)")
    ax.set_title("(5) Fleet-Level Grid Impact")
    ax.set_xlabel("Number of trailers"); ax.set_ylabel("Power (kW)")
    ax2f.set_ylabel("Annual Revenue (k€)"); ax.set_xscale("log")
    ax.legend(loc="upper left",  fontsize=7)
    ax2f.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

    # ── Panel 6: Degradation sensitivity ─────────────────────────────────────
    ax = axes[1, 2]
    deg_keys  = [k for k in results if k.startswith("DegSens")]
    deg_costs_list = [float(k.replace("DegSens_", "")) for k in deg_keys]
    net_costs_list = [results[k].cost_eur_day * days for k in deg_keys]
    exports_list   = [results[k].v2g_export_kwh_day for k in deg_keys]
    ax.plot(deg_costs_list, net_costs_list, "o-", color=COL["v2g"], lw=2, label="Annual Net Cost")
    ax.axhline(B.cost_eur_day * days, color=COL["smart"], lw=1.5, linestyle="--",
               label="Smart-only reference")
    ax2d = ax.twinx()
    ax2d.bar(deg_costs_list, exports_list, width=0.008, color="#006600", alpha=0.5,
             label="V2G Export (kWh/day)")
    ax.set_title("(6) Battery Degradation Sensitivity")
    ax.set_xlabel("Degradation cost (€/kWh cycled)"); ax.set_ylabel("Annual Net Cost (€)")
    ax2d.set_ylabel("V2G Export (kWh/day)")
    ax.legend(loc="upper left",  fontsize=7)
    ax2d.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── Detailed tables ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📋 Detailed Results")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Scenario Comparison (Daily)")
        df_compare = pd.DataFrame([
            {
                "Scenario":               r.scenario,
                "Charge Cost (€/day)":    f"{r.charge_cost_eur_day:.2f}",
                "V2G Revenue (€/day)":    f"{r.v2g_revenue_eur_day:.2f}",
                "Deg. Cost (€/day)":      f"{r.deg_cost_eur_day:.2f}",
                "Net Cost (€/day)":       f"{r.cost_eur_day:.2f}",
                "V2G Export (kWh/day)":   f"{r.v2g_export_kwh_day:.1f}",
            }
            for r in [A, B, C]
        ])
        st.table(df_compare)

    with col_t2:
        st.markdown("#### Fleet Impact (Single Trailer × Fleet Size)")
        df_fleet_disp = be.compute_fleet_impact(C, [1, 10, 50, 100])
        df_fleet_disp["Annual_V2G_Revenue_EUR"] = df_fleet_disp["Annual_V2G_Revenue_EUR"].map("€{:,.0f}".format)
        df_fleet_disp["Annual_V2G_Export_MWh"]  = df_fleet_disp["Annual_V2G_Export_MWh"].map("{:.1f} MWh".format)
        st.table(df_fleet_disp[["Fleet_Size","Peak_Charge_kW","Peak_Discharge_kW",
                                 "Annual_V2G_Export_MWh","Annual_V2G_Revenue_EUR"]])

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💾 Export Results")
    c_dl1, c_dl2 = st.columns(2)

    # Timeseries CSV
    df_ts = pd.DataFrame({
        "Hour":        hours,
        "Plugged":     C.plugged,
        "TRU_kW":      C.tru_load,
        "BuyPrice":    C.price_buy,
        "V2G_Price":   C.price_v2g,
        "A_Charge_kW": A.p_charge,
        "B_Charge_kW": B.p_charge,
        "C_Charge_kW": C.p_charge,
        "C_Discharge_kW": C.p_discharge,
        "A_SoC_pct":   A.soc / v2g_params.usable_capacity_kWh * 100,
        "B_SoC_pct":   B.soc / v2g_params.usable_capacity_kWh * 100,
        "C_SoC_pct":   C.soc / v2g_params.usable_capacity_kWh * 100,
    })
    with c_dl1:
        st.download_button(
            "⬇️ Download Timeseries CSV",
            df_ts.to_csv(index=False).encode(),
            file_name="v2g_schedules.csv",
            mime="text/csv",
        )

    # Sensitivity CSV
    df_sens_out = pd.DataFrame([
        {"DegCost": float(k.replace("DegSens_","")),
         "NetCost_EUR_day": results[k].cost_eur_day,
         "V2G_Rev_EUR_day": results[k].v2g_revenue_eur_day,
         "V2G_kWh_day":     results[k].v2g_export_kwh_day}
        for k in results if k.startswith("DegSens")
    ])
    with c_dl2:
        st.download_button(
            "⬇️ Download Sensitivity CSV",
            df_sens_out.to_csv(index=False).encode(),
            file_name="degradation_sensitivity.csv",
            mime="text/csv",
        )

    # ── Theory box ────────────────────────────────────────────────────────────
    with st.expander("📖 Model Details & Assumptions", expanded=False):
        st.markdown(f"""
**MILP Formulation** *(Biedenbach & Strunz 2024, eq. 2-6)*

Objective:
```
min  Σₜ buy[t]·P_c[t]·Δt  −  Σₜ v2g[t]·P_d[t]·Δt  +  Σₜ deg·(P_c[t]+P_d[t])·Δt
```

Constraints:
- **SoC dynamics**: `e[t] = e[t-1] + P_c·η_c·Δt − P_d/η_d·Δt − P_tru·Δt`
- **SoC bounds**: `E_min ≤ e[t] ≤ E_max`  (cold-chain floor + cycle ceiling)
- **Power limits**: `0 ≤ P_c[t] ≤ {p_charge} kW × plug[t]`  |  `0 ≤ P_d[t] ≤ {p_discharge} kW × plug[t]`
- **Mutual exclusion** (linearised): `P_c[t] + P_d[t] ≤ max_power`
- **Final SoC**: `e[N-1] ≥ E_final` ({soc_final}%)
- **Grid kVA cap (Fields 30 & 31)**: `P_c[t] ≤ min(P_c_max, kVA_limit × PF)` and `P_d[t] ≤ min(P_d_max, kVA_limit × PF)` — active limits: {f"Depot {depot_kVA:.0f} kVA" if depot_kVA > 0 else "none"} / {f"Transformer {trafo_kVA:.0f} kVA" if trafo_kVA > 0 else "none"}

**V2G Revenue Streams** *(Agora Verkehrswende 2025, Table 1)*:
- Arbitrage (EPEX intraday): buy at night (0.16 €/kWh), sell at peak (0.33 €/kWh)
- Balancing market premium: +0.132 €/kWh during 16:00–20:00

**Battery Degradation** *(Agora 2025 §3.2 / Biedenbach eq. 6)*:
- Modelled as linear cost per kWh cycled: {deg_cost} €/kWh (baseline)
- Break-even: ~0.23 €/kWh under current DE tariff spread
- V2G viable if degradation cost ≤ 0.10 €/kWh (current NMC)
        """)
