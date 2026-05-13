"""
dashboard.py — NNN Media Budget Planner

Built for Growth Marketing teams. Uses a trained NNN transformer model
under the hood, surfacing channel efficiency, budget scenarios, DEP impact,
and cross-channel synergies in actionable terms.

Run:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"

CHANNEL_ICONS = {
    "Google Non-Brand": "🔍", "Google Brand": "🏷️", "Google Shopping": "🛒",
    "YouTube": "▶️", "Facebook": "📘", "LinkedIn": "💼",
    "Display": "🖼️", "Bing": "🅱️", "Reddit": "💬",
    "CTV": "📺", "Podcast": "🎙️", "Affiliates": "🤝",
    "Review Sites": "⭐",
}


# =========================================================================
# Cached loaders
# =========================================================================

@st.cache_resource
def load_model():
    from model import NNNModel
    with open(OUTPUT / "cv_results.json") as f:
        cv = json.load(f)
    probe = json.load(open(OUTPUT / "probe_results.json"))
    synergy_df = pd.read_csv(OUTPUT / "synergy_matrix.csv", index_col=0)
    channels = list(synergy_df.columns)
    model = NNNModel(
        n_channels=len(channels), input_dim=258,
        d_model=cv["config"]["d_model"], n_heads=cv["config"]["n_heads"],
        n_funnel_stages=cv["config"]["n_funnel_stages"],
    )
    model.load_state_dict(torch.load(OUTPUT / "nnn_model.pt", weights_only=True))
    model.eval()
    return model, channels, cv, probe, synergy_df


@st.cache_data
def load_data():
    from data_prep import RealDataConfig, prepare_real_data
    config = RealDataConfig(csv_path=str(ROOT / "data" / "real_marketing_data.csv"))
    data = prepare_real_data(config)
    meta = data["metadata"]
    X = torch.tensor(data["tensor"], dtype=torch.float32)
    y_raw = torch.tensor(data["targets"]["Closed_Won"], dtype=torch.float32)
    y_log = torch.log1p(y_raw)

    csv = pd.read_csv(ROOT / "data" / "real_marketing_data.csv", keep_default_na=False, na_values=[""])
    channel_names = meta["channels"]
    current_spend = np.array([csv.groupby("Channel")["Spend"].mean().get(ch, 0) for ch in channel_names])
    current_dep = np.array([csv.groupby("Channel")["DEP"].mean().get(ch, 0) for ch in channel_names])

    # Weekly totals for backtest
    dep_by_week_geo = csv.pivot_table(
        index="Date", columns="Geography", values="DEP", aggfunc="sum", fill_value=0,
    )
    spend_by_week_geo = csv.pivot_table(
        index="Date", columns="Geography", values="Spend", aggfunc="sum", fill_value=0,
    )

    return {
        "X": X, "y_raw": y_raw,
        "y_mean": y_log.mean().item(), "y_std": y_log.std().item(),
        "spend_max": meta["normalization"]["max"][0],
        "current_spend": current_spend,
        "current_dep": current_dep,
        "dep_weekly": dep_by_week_geo,
        "spend_weekly": spend_by_week_geo,
        "geos": meta["geos"],
        "dates": [str(d).split(" ")[0] for d in meta["dates"]],
        "channels": channel_names,
        "csv": csv,
    }


@st.cache_data
def load_optim():
    with open(OUTPUT / "optimization_results.json") as f:
        return json.load(f)


def pred_to_cw(pred, y_mean, y_std):
    return torch.expm1(pred * y_std + y_mean)


def run_model(model, X_base, spend_multipliers, d):
    X_mod = X_base.clone()
    for i, mult in enumerate(spend_multipliers):
        X_mod[:, :, i, 0] = X_base[:, :, i, 0] * mult
    with torch.no_grad():
        pred = model(X_mod).squeeze(-1)
        cw = pred_to_cw(pred, d["y_mean"], d["y_std"])
    return cw.sum().item(), cw.sum(dim=1).numpy()


# =========================================================================
# Scorecard builder
# =========================================================================

def build_scorecard(channels, current_spend, current_dep, optim, synergy_df):
    sens = optim["sensitivity"]
    total_spend = current_spend.sum()
    mrois = [sens[ch]["marginal_roi"] for ch in channels]
    max_mroi, min_mroi = max(mrois), min(mrois)
    mroi_range = max_mroi - min_mroi if max_mroi != min_mroi else 1

    syn_matrix = synergy_df.values

    rows = []
    for i, ch in enumerate(channels):
        spend = current_spend[i]
        dep = current_dep[i]
        spend_pct = spend / total_spend * 100
        mroi = sens[ch]["marginal_roi"]
        eff = (mroi - min_mroi) / mroi_range * 100
        dep_per_1k = dep / (spend / 1000 + 1e-8)

        # Top synergy partner
        syn_row = syn_matrix[i].copy()
        syn_row[i] = 0
        top_syn_idx = syn_row.argmax()
        top_syn_name = channels[top_syn_idx]
        top_syn_weight = syn_row[top_syn_idx]

        # Who feeds this channel most
        syn_col = syn_matrix[:, i].copy()
        syn_col[i] = 0
        top_feeder_idx = syn_col.argmax()
        top_feeder = channels[top_feeder_idx]

        if eff >= 65:
            signal = "🟢 Invest More"
        elif eff >= 35:
            signal = "🟡 Hold"
        else:
            signal = "🔴 Reduce"

        rows.append({
            "Channel": ch, "Icon": CHANNEL_ICONS.get(ch, "📊"),
            "Weekly Spend": spend, "Budget Share": spend_pct,
            "Efficiency": eff, "mROI": mroi,
            "DEP/wk": dep, "DEP per $1K": dep_per_1k,
            "Signal": signal,
            "Synergy Target": f"Boosts {top_syn_name}",
            "Fed By": top_feeder,
            "Synergy Weight": top_syn_weight,
        })

    return pd.DataFrame(rows).sort_values("Efficiency", ascending=False)


# =========================================================================
# Scenario engine
# =========================================================================

SCENARIOS = {
    "Current Plan": "No changes — your current weekly allocation.",
    "Conservative (+10%)": "Shift 10% from bottom-3 efficiency channels to top-3.",
    "Moderate (+20%)": "Shift 20% — a meaningful test without major risk.",
    "Aggressive (+30%)": "Shift 30% toward highest-efficiency channels.",
    "Custom": "Set your own spend adjustments per channel.",
}


def compute_scenario(key, scorecard, channels, current_spend):
    if key == "Current Plan":
        return np.ones(len(channels))
    if key == "Custom":
        return None

    pct = {"Conservative (+10%)": 0.10, "Moderate (+20%)": 0.20, "Aggressive (+30%)": 0.30}[key]
    sc = scorecard.sort_values("Efficiency", ascending=False)
    top3 = sc.head(3)["Channel"].tolist()
    bot3 = sc.tail(3)["Channel"].tolist()

    total = current_spend.sum()
    shift = total * pct
    bot_spend = sum(current_spend[channels.index(ch)] for ch in bot3)
    top_spend = sum(current_spend[channels.index(ch)] for ch in top3)

    mults = np.ones(len(channels))
    for ch in bot3:
        idx = channels.index(ch)
        cut = shift * (current_spend[idx] / (bot_spend + 1e-8))
        mults[idx] = max(0.1, (current_spend[idx] - cut) / (current_spend[idx] + 1e-8))
    for ch in top3:
        idx = channels.index(ch)
        add = shift * (current_spend[idx] / (top_spend + 1e-8))
        mults[idx] = (current_spend[idx] + add) / (current_spend[idx] + 1e-8)
    return mults


# =========================================================================
# Page sections
# =========================================================================

def render_header(scorecard, d):
    model, channels, cv, _, _ = load_model()
    X = d["X"]

    # Inline backtest sparkline as trust signal
    with torch.no_grad():
        pred_norm = model(X).squeeze(-1)
        pred = pred_to_cw(pred_norm, d["y_mean"], d["y_std"])
    actual = d["y_raw"].sum(dim=0).numpy()
    predicted = pred.sum(dim=0).numpy()
    r2 = 1 - ((actual - predicted)**2).sum() / ((actual - actual.mean())**2).sum()

    col_main, col_trust = st.columns([3, 1])
    with col_main:
        biggest = scorecard.sort_values("Weekly Spend", ascending=False).iloc[0]
        rank = list(scorecard["Channel"]).index(biggest["Channel"]) + 1
        top = scorecard.iloc[0]
        st.markdown(
            f"**{biggest['Channel']}** takes **{biggest['Budget Share']:.0f}%** of budget "
            f"(${biggest['Weekly Spend']:,.0f}/wk) but ranks **#{rank}** in efficiency. "
            f"**{top['Channel']}** has the highest ROI at only {top['Budget Share']:.1f}% of budget."
        )
    with col_trust:
        confidence = "High" if r2 > 0.85 else "Medium" if r2 > 0.7 else "Low"
        st.metric("Model Confidence", confidence, help=f"R²={r2:.2f} on 71 weeks of historical data")


def render_scorecard(scorecard):
    st.markdown("## Channel Scorecard")
    st.caption("Each channel rated on efficiency (deals per dollar) with cross-channel synergy context.")

    for row_start in range(0, len(scorecard), 3):
        cols = st.columns(3)
        for col_i in range(3):
            idx = row_start + col_i
            if idx >= len(scorecard):
                break
            r = scorecard.iloc[idx]
            with cols[col_i]:
                with st.container(border=True):
                    st.markdown(f"**{r['Icon']} {r['Channel']}** &ensp; {r['Signal']}")
                    m1, m2 = st.columns(2)
                    m1.metric("Spend/wk", f"${r['Weekly Spend']:,.0f}", delta=f"{r['Budget Share']:.1f}% of budget", delta_color="off")
                    m2.metric("DEP/wk", f"${r['DEP/wk']:,.0f}", delta=f"${r['DEP per $1K']:,.0f} per $1K", delta_color="off")
                    st.caption(f"mROI: {r['mROI']:.5f} · {r['Synergy Target']} · Fed by {r['Fed By']}")

    # Efficiency vs Budget Share scatter
    fig = px.scatter(
        scorecard, x="Budget Share", y="DEP per $1K",
        size="Weekly Spend", color="Signal",
        text="Channel",
        color_discrete_map={
            "🟢 Invest More": "#2ecc71", "🟡 Hold": "#f39c12", "🔴 Reduce": "#e74c3c",
        },
        labels={"Budget Share": "Budget Share (%)", "DEP per $1K": "DEP per $1K Spend"},
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bubble size = weekly spend. Top-left = underfunded efficient channels. Bottom-right = overfunded inefficient channels.")


def render_planner(d, scorecard):
    st.markdown("## Budget Planner")

    model, channels, _, _, synergy_df = load_model()
    X = d["X"]
    B, T, C, D = X.shape
    X_base = X[:, max(0, T - 8):, :, :]
    current_spend = d["current_spend"]
    current_dep = d["current_dep"]

    baseline_cw, baseline_geo = run_model(model, X_base, np.ones(len(channels)), d)

    scenario = st.radio("Select scenario", list(SCENARIOS.keys()), horizontal=True, key="scenario")
    st.caption(SCENARIOS[scenario])

    if scenario == "Custom":
        st.markdown("**Adjust each channel:**")
        slider_cols = st.columns(4)
        mults = np.ones(len(channels))
        for i, ch in enumerate(channels):
            with slider_cols[i % 4]:
                pct = st.slider(
                    f"{CHANNEL_ICONS.get(ch,'')} {ch}", min_value=-50, max_value=100,
                    value=0, step=5, format="%+d%%", key=f"p_{ch}",
                )
                mults[i] = 1.0 + pct / 100.0
    else:
        mults = compute_scenario(scenario, scorecard, channels, current_spend)

    # Run simulation
    sim_cw, sim_geo = run_model(model, X_base, mults, d)
    delta_cw = sim_cw - baseline_cw
    delta_pct = delta_cw / (baseline_cw + 1e-8) * 100

    new_spend = current_spend * mults
    old_total = current_spend.sum()
    new_total = new_spend.sum()

    # DEP impact (proportional estimate based on current DEP/spend ratio)
    old_dep_total = current_dep.sum()
    new_dep = current_dep * mults
    new_dep_total = new_dep.sum()
    dep_delta = new_dep_total - old_dep_total

    # DEP/Spend
    old_dep_per_spend = old_dep_total / (old_total + 1e-8)
    new_dep_per_spend = new_dep_total / (new_total + 1e-8)

    # Results
    st.markdown("### Projected Impact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed-Won Deals", f"{sim_cw:,.0f}", delta=f"{delta_cw:+,.0f} ({delta_pct:+.1f}%)")
    c2.metric("Weekly DEP", f"${new_dep_total:,.0f}", delta=f"${dep_delta:+,.0f}")
    c3.metric("DEP / Spend", f"${new_dep_per_spend:,.2f}", delta=f"${new_dep_per_spend - old_dep_per_spend:+,.3f}")
    c4.metric("Weekly Budget", f"${new_total:,.0f}", delta=f"${new_total - old_total:+,.0f}" if abs(new_total - old_total) > 100 else "No change")

    # Before/after + response curves side by side
    chart_left, chart_right = st.columns([3, 2])

    with chart_left:
        plan_rows = []
        for i, ch in enumerate(channels):
            plan_rows.append({"Channel": ch, "Allocation": "Current", "Spend": current_spend[i]})
            plan_rows.append({"Channel": ch, "Allocation": "Planned", "Spend": new_spend[i]})
        plan_df = pd.DataFrame(plan_rows)

        fig = px.bar(
            plan_df, x="Spend", y="Channel", color="Allocation", barmode="group",
            orientation="h",
            color_discrete_map={"Current": "lightslategray", "Planned": "#e74c3c"},
            labels={"Spend": "$/week"},
        )
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.08), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        # Show response curves for channels being changed
        changed = [(ch, mults[i]) for i, ch in enumerate(channels) if abs(mults[i] - 1.0) > 0.05]
        if changed:
            st.caption("Response curves for adjusted channels")
            test_mults = np.linspace(0.5, 1.5, 11)
            curve_rows = []
            for ch, _ in changed[:4]:
                c_idx = channels.index(ch)
                for tm in test_mults:
                    ms = np.ones(len(channels))
                    ms[c_idx] = tm
                    cw, _ = run_model(model, X_base, ms, d)
                    curve_rows.append({"Channel": ch, "Multiplier": tm, "CW": cw})
            if curve_rows:
                cdf = pd.DataFrame(curve_rows)
                fig2 = px.line(cdf, x="Multiplier", y="CW", color="Channel", labels={"CW": "Deals"})
                fig2.add_vline(x=1.0, line_dash="dash", line_color="gray")
                fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Adjust a scenario to see response curves for changed channels.")

    # Synergy alerts
    if scenario != "Current Plan":
        syn_matrix = synergy_df.values
        cut_channels = [ch for i, ch in enumerate(channels) if mults[i] < 0.8]
        if cut_channels:
            alerts = []
            for ch in cut_channels:
                c_idx = channels.index(ch)
                syn_row = syn_matrix[c_idx].copy()
                syn_row[c_idx] = 0
                top_target_idx = syn_row.argmax()
                top_target = channels[top_target_idx]
                if syn_row[top_target_idx] > 0.082:
                    alerts.append(f"Cutting **{ch}** may reduce **{top_target}** effectiveness (synergy: {syn_row[top_target_idx]:.4f})")
            if alerts:
                st.warning("**Synergy alerts:**  \n" + "  \n".join(alerts))

    # Geo breakdown
    with st.expander("Impact by geography"):
        geo_df = pd.DataFrame({
            "Geography": d["geos"],
            "Current CW": baseline_geo, "Projected CW": sim_geo,
            "Delta": sim_geo - baseline_geo,
        })
        st.dataframe(geo_df.style.format({
            "Current CW": "{:,.0f}", "Projected CW": "{:,.0f}", "Delta": "{:+,.0f}",
        }), hide_index=True, use_container_width=True)

    # Export
    st.markdown("---")
    export_rows = []
    for i, ch in enumerate(channels):
        sc_row = scorecard[scorecard["Channel"] == ch].iloc[0]
        export_rows.append({
            "Channel": ch,
            "Current_Weekly_Spend": round(current_spend[i], 2),
            "Planned_Weekly_Spend": round(new_spend[i], 2),
            "Change_Pct": f"{(mults[i]-1)*100:+.0f}%",
            "Current_DEP": round(current_dep[i], 2),
            "Planned_DEP": round(new_dep[i], 2),
            "DEP_per_1K_Spend": round(sc_row["DEP per $1K"], 2),
            "Efficiency_Signal": sc_row["Signal"],
            "Projected_CW_Delta": round(delta_cw),
            "Scenario": scenario,
        })
    csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode()
    st.download_button(
        "📥 Download Budget Plan", data=csv_bytes,
        file_name=f"nnn_budget_plan_{scenario.lower().replace(' ','_').replace('+','')}.csv",
        mime="text/csv",
    )


def render_deep_dive(d, scorecard):
    st.markdown("## Channel Deep Dive")
    model, channels, _, _, synergy_df = load_model()

    selected = st.selectbox(
        "Pick a channel", channels,
        format_func=lambda ch: f"{CHANNEL_ICONS.get(ch,'')} {ch}",
        key="dd_ch",
    )

    X = d["X"]
    B, T, C, D = X.shape
    X_base = X[:, max(0, T - 8):, :, :]
    c_idx = channels.index(selected)
    sc = scorecard[scorecard["Channel"] == selected].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Weekly Spend", f"${sc['Weekly Spend']:,.0f}")
    c2.metric("Budget Share", f"{sc['Budget Share']:.1f}%")
    c3.metric("DEP/wk", f"${sc['DEP/wk']:,.0f}")
    c4.metric("DEP per $1K", f"${sc['DEP per $1K']:,.0f}")
    c5.metric("Signal", sc["Signal"].split(" ", 1)[1])

    left, right = st.columns(2)

    with left:
        st.markdown("#### Spend vs. Deals")
        mults_range = np.linspace(0.0, 2.0, 21)
        curve = []
        for mult in mults_range:
            ms = np.ones(len(channels))
            ms[c_idx] = mult
            cw, _ = run_model(model, X_base, ms, d)
            curve.append({"Spend ($/wk)": d["current_spend"][c_idx] * mult, "Mult": mult, "Deals": cw})
        cdf = pd.DataFrame(curve)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cdf["Spend ($/wk)"], y=cdf["Deals"], mode="lines+markers", line=dict(color="#e74c3c", width=2.5), marker=dict(size=5)))
        fig.add_vline(x=d["current_spend"][c_idx], line_dash="dash", line_color="gray", annotation_text="Current")
        fig.update_layout(xaxis_title="$/week", yaxis_title="Projected Deals", height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Cross-Channel Synergies")
        syn_matrix = synergy_df.values

        outgoing = [(channels[j], syn_matrix[c_idx, j]) for j in range(len(channels)) if j != c_idx]
        outgoing.sort(key=lambda x: -x[1])
        incoming = [(channels[j], syn_matrix[j, c_idx]) for j in range(len(channels)) if j != c_idx]
        incoming.sort(key=lambda x: -x[1])

        syn_df = pd.DataFrame({
            "Direction": [f"{selected} → {ch}" for ch, _ in outgoing[:5]] + [f"{ch} → {selected}" for ch, _ in incoming[:5]],
            "Weight": [w for _, w in outgoing[:5]] + [w for _, w in incoming[:5]],
            "Type": ["Boosts"] * 5 + ["Boosted by"] * 5,
        })
        fig2 = px.bar(syn_df, x="Weight", y="Direction", color="Type", orientation="h",
                       color_discrete_map={"Boosts": "#3498db", "Boosted by": "#2ecc71"})
        fig2.update_layout(height=350, legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)


def render_backtest(d):
    st.markdown("## Historical Accuracy")
    model, channels, cv, _, _ = load_model()
    X = d["X"]

    with torch.no_grad():
        pred_norm = model(X).squeeze(-1)
        pred_raw = pred_to_cw(pred_norm, d["y_mean"], d["y_std"])

    y_raw = d["y_raw"]
    dates = d["dates"]
    geos = d["geos"]

    geo_select = st.selectbox("Geography", ["All regions"] + geos, key="bt_geo")

    if geo_select == "All regions":
        actual = y_raw.sum(dim=0).numpy()
        predicted = pred_raw.sum(dim=0).numpy()
    else:
        g_idx = geos.index(geo_select)
        actual = y_raw[g_idx].numpy()
        predicted = pred_raw[g_idx].numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers", name="Actual", line=dict(color="steelblue", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines", name="Predicted", line=dict(color="#e74c3c", width=2, dash="dot")))

    if len(dates) > 59:
        fig.add_vrect(x0=dates[50], x1=dates[min(59, len(dates)-1)], fillcolor="rgba(255,165,0,0.12)", line_width=0,
                      annotation_text="Fold 3 — high error period", annotation_position="top left", annotation_font_color="darkorange")

    fig.update_layout(height=400, yaxis_title="Closed-Won Deals", legend=dict(orientation="h", y=-0.12), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    error = np.abs(actual - predicted) / (np.abs(actual) + 1.0) * 100
    r2 = 1 - ((actual - predicted)**2).sum() / ((actual - actual.mean())**2).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Error (MAPE)", f"{error.mean():.1f}%")
    c2.metric("R²", f"{r2:.2%}")
    c3.metric("Model Confidence", "High" if r2 > 0.85 else "Medium" if r2 > 0.7 else "Low")

    # DEP trend
    st.markdown("#### DEP Trend by Channel")
    csv = d["csv"]
    dep_trend = csv.groupby(["Date", "Channel"])["DEP"].sum().reset_index()
    dep_trend = dep_trend[dep_trend["DEP"] > 0]
    if len(dep_trend) > 0:
        fig_dep = px.area(dep_trend, x="Date", y="DEP", color="Channel", labels={"DEP": "Weekly DEP ($)"})
        fig_dep.update_layout(height=350, legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_dep, use_container_width=True)


# =========================================================================
# Main
# =========================================================================

def main():
    st.set_page_config(page_title="NNN Media Budget Planner", page_icon="📊", layout="wide")

    st.markdown("# 📊 Media Budget Planner")
    st.caption("Powered by NNN — trained on 71 weeks of real spend, CRM, and DEP data across 5 regions and 13 channels.")

    d = load_data()
    optim = load_optim()
    _, channels, _, _, synergy_df = load_model()

    scorecard = build_scorecard(channels, d["current_spend"], d["current_dep"], optim, synergy_df)

    render_header(scorecard, d)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Channel Scorecard",
        "💰 Budget Planner",
        "🔎 Channel Deep Dive",
        "📈 Historical View",
    ])

    with tab1:
        render_scorecard(scorecard)
    with tab2:
        render_planner(d, scorecard)
    with tab3:
        render_deep_dive(d, scorecard)
    with tab4:
        render_backtest(d)


if __name__ == "__main__":
    main()
