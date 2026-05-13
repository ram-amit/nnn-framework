"""
dashboard.py — NNN Media Budget Planner

Built for Growth Marketing teams to plan and optimize media budgets.
Uses a trained NNN (transformer) model under the hood, but surfaces
only the decisions and evidence marketers need.

Run:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Cached loaders (same engine, friendlier surface)
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

    spend_csv = pd.read_csv(
        ROOT / "data" / "real_marketing_data.csv",
        keep_default_na=False, na_values=[""],
    )
    current_spend = spend_csv.groupby("Channel")["Spend"].mean()
    channel_names = meta["channels"]
    current_spend_vec = np.array([current_spend.get(ch, 0) for ch in channel_names])

    return {
        "X": X, "y_raw": y_raw,
        "y_mean": y_log.mean().item(), "y_std": y_log.std().item(),
        "spend_max": meta["normalization"]["max"][0],
        "current_spend": current_spend_vec,
        "geos": meta["geos"],
        "dates": [str(d).split(" ")[0] for d in meta["dates"]],
        "channels": channel_names,
    }


@st.cache_data
def load_optim():
    with open(OUTPUT / "optimization_results.json") as f:
        return json.load(f)


def pred_to_cw(pred, y_mean, y_std):
    return torch.expm1(pred * y_std + y_mean)


def run_model(model, X_base, spend_multipliers, d):
    """Run model with per-channel spend multipliers. Returns total CW and per-geo CW."""
    X_mod = X_base.clone()
    for i, mult in enumerate(spend_multipliers):
        X_mod[:, :, i, 0] = X_base[:, :, i, 0] * mult
    with torch.no_grad():
        pred = model(X_mod).squeeze(-1)
        cw = pred_to_cw(pred, d["y_mean"], d["y_std"])
    return cw.sum().item(), cw.sum(dim=1).numpy()


def build_channel_scorecard(channels, current_spend, optim):
    """Build a DataFrame scoring each channel for the scorecard view."""
    sens = optim["sensitivity"]
    total_spend = current_spend.sum()

    mrois = [sens[ch]["marginal_roi"] for ch in channels]
    max_mroi = max(mrois)
    min_mroi = min(mrois)
    mroi_range = max_mroi - min_mroi if max_mroi != min_mroi else 1

    rows = []
    for i, ch in enumerate(channels):
        spend = current_spend[i]
        spend_pct = spend / total_spend * 100
        mroi = sens[ch]["marginal_roi"]

        # Efficiency score: 0-100 based on mROI ranking
        eff_score = (mroi - min_mroi) / mroi_range * 100

        # Budget fairness: does this channel's budget share match its efficiency?
        mroi_share = mroi / sum(mrois) * 100
        budget_gap = mroi_share - spend_pct

        if eff_score >= 65:
            signal = "🟢 Invest More"
            reason = "High efficiency — each dollar here drives more deals than average"
        elif eff_score >= 35:
            signal = "🟡 Hold"
            reason = "Average efficiency — maintain current levels"
        else:
            signal = "🔴 Reduce"
            reason = "Below-average efficiency — consider reallocating to higher-performing channels"

        rows.append({
            "Channel": ch,
            "Icon": CHANNEL_ICONS.get(ch, "📊"),
            "Weekly Spend": spend,
            "Budget Share": spend_pct,
            "Efficiency Score": eff_score,
            "Marginal ROI": mroi,
            "Signal": signal,
            "Reason": reason,
            "Budget Gap": budget_gap,
        })

    return pd.DataFrame(rows).sort_values("Efficiency Score", ascending=False)


# =========================================================================
# Scenario engine
# =========================================================================

SCENARIOS = {
    "Current Plan": {
        "description": "No changes — your current weekly budget allocation.",
        "multipliers": None,
    },
    "Conservative Shift": {
        "description": "Shift 10% of budget from the 3 lowest-efficiency channels to the 3 highest.",
        "shift_pct": 0.10,
    },
    "Moderate Rebalance": {
        "description": "Shift 20% from underperformers to top performers. A meaningful test without major risk.",
        "shift_pct": 0.20,
    },
    "Aggressive Growth": {
        "description": "Shift 30% toward highest-efficiency channels. Higher potential upside, needs close monitoring.",
        "shift_pct": 0.30,
    },
    "Custom": {
        "description": "Set your own spend levels per channel.",
        "multipliers": None,
    },
}


def compute_scenario_multipliers(scenario_key, scorecard, channels, current_spend):
    """Turn a scenario into per-channel spend multipliers."""
    if scenario_key == "Current Plan":
        return np.ones(len(channels))

    if scenario_key == "Custom":
        return None  # handled by sliders

    shift_pct = SCENARIOS[scenario_key]["shift_pct"]
    sorted_sc = scorecard.sort_values("Efficiency Score", ascending=False)
    top3 = sorted_sc.head(3)["Channel"].tolist()
    bot3 = sorted_sc.tail(3)["Channel"].tolist()

    total = current_spend.sum()
    shift_amount = total * shift_pct

    # Take from bottom 3 proportionally
    bot3_spend = sum(current_spend[channels.index(ch)] for ch in bot3)
    top3_spend = sum(current_spend[channels.index(ch)] for ch in top3)

    multipliers = np.ones(len(channels))
    for ch in bot3:
        idx = channels.index(ch)
        ch_share = current_spend[idx] / (bot3_spend + 1e-8)
        cut = shift_amount * ch_share
        multipliers[idx] = max(0.1, (current_spend[idx] - cut) / (current_spend[idx] + 1e-8))

    for ch in top3:
        idx = channels.index(ch)
        ch_share = current_spend[idx] / (top3_spend + 1e-8)
        add = shift_amount * ch_share
        multipliers[idx] = (current_spend[idx] + add) / (current_spend[idx] + 1e-8)

    return multipliers


# =========================================================================
# Page sections
# =========================================================================

def render_headline(scorecard, d):
    """Top-of-page executive insight."""
    top = scorecard.iloc[0]
    worst = scorecard.iloc[-1]

    top_spend_ch = scorecard.sort_values("Weekly Spend", ascending=False).iloc[0]

    st.markdown(
        f"#### Your biggest spend — **{top_spend_ch['Channel']}** "
        f"at **${top_spend_ch['Weekly Spend']:,.0f}/wk** "
        f"({top_spend_ch['Budget Share']:.0f}% of budget) — "
        f"ranks **#{int(scorecard[scorecard['Channel']==top_spend_ch['Channel']].index[0])+1}** "
        f"in efficiency out of {len(scorecard)} channels."
    )
    st.caption(
        f"Meanwhile, **{top['Channel']}** has the highest marginal ROI "
        f"but only gets {top['Budget Share']:.1f}% of budget."
    )


def render_scorecard(scorecard):
    """Channel scorecard with traffic-light signals."""
    st.markdown("## Channel Scorecard")
    st.caption("Each channel rated by how efficiently it drives Closed-Won deals at its current spend level.")

    cols_per_row = 3
    rows_needed = (len(scorecard) + cols_per_row - 1) // cols_per_row

    for row_i in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_i in range(cols_per_row):
            idx = row_i * cols_per_row + col_i
            if idx >= len(scorecard):
                break
            r = scorecard.iloc[idx]
            with cols[col_i]:
                st.markdown(
                    f"**{r['Icon']} {r['Channel']}**  \n"
                    f"{r['Signal']}  \n"
                    f"${r['Weekly Spend']:,.0f}/wk · {r['Budget Share']:.1f}% of budget  \n"
                    f"Efficiency: **{r['Efficiency Score']:.0f}/100**"
                )
                st.caption(r["Reason"])
                st.markdown("---")


def render_budget_planner(d, scorecard):
    """Scenario selector + results."""
    st.markdown("## Budget Planner")
    st.caption("Pick a reallocation scenario to see its projected impact on Closed-Won deals.")

    model, channels, _, _, _ = load_model()
    X = d["X"]
    B, T, C, D = X.shape
    X_base = X[:, max(0, T - 8):, :, :]
    current_spend = d["current_spend"]

    # Baseline
    baseline_cw, baseline_geo = run_model(model, X_base, np.ones(len(channels)), d)

    # Scenario picker
    scenario_key = st.radio(
        "Select a scenario",
        list(SCENARIOS.keys()),
        horizontal=True,
        key="scenario_picker",
    )
    st.caption(SCENARIOS[scenario_key]["description"])

    # Compute multipliers
    if scenario_key == "Custom":
        st.markdown("**Adjust each channel:**")
        slider_cols = st.columns(3)
        multipliers = np.ones(len(channels))
        for i, ch in enumerate(channels):
            with slider_cols[i % 3]:
                pct = st.slider(
                    f"{CHANNEL_ICONS.get(ch,'')} {ch}",
                    min_value=-50, max_value=100, value=0, step=5,
                    format="%+d%%", key=f"custom_{ch}",
                )
                multipliers[i] = 1.0 + pct / 100.0
    else:
        multipliers = compute_scenario_multipliers(
            scenario_key, scorecard, channels, current_spend,
        )

    # Run simulation
    sim_cw, sim_geo = run_model(model, X_base, multipliers, d)
    delta_cw = sim_cw - baseline_cw
    delta_pct = delta_cw / (baseline_cw + 1e-8) * 100

    new_spend = current_spend * multipliers
    old_total = current_spend.sum()
    new_total = new_spend.sum()

    # Results
    st.markdown("### Projected Impact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Deals (8wk)", f"{baseline_cw:,.0f}")
    c2.metric(
        "Projected Deals",
        f"{sim_cw:,.0f}",
        delta=f"{delta_cw:+,.0f} ({delta_pct:+.1f}%)",
    )
    c3.metric(
        "Weekly Budget",
        f"${new_total:,.0f}",
        delta=f"${new_total - old_total:+,.0f} vs. current" if abs(new_total - old_total) > 100 else "No change",
    )

    # Before/after chart
    plan_rows = []
    for i, ch in enumerate(channels):
        plan_rows.append({"Channel": ch, "Type": "Current", "Spend": current_spend[i]})
        plan_rows.append({"Channel": ch, "Type": "Planned", "Spend": new_spend[i]})
    plan_df = pd.DataFrame(plan_rows)
    plan_df = plan_df.sort_values("Spend", ascending=True)

    fig = px.bar(
        plan_df, x="Spend", y="Channel", color="Type", barmode="group",
        orientation="h",
        color_discrete_map={"Current": "lightslategray", "Planned": "#e74c3c"},
        labels={"Spend": "Weekly Spend ($)"},
    )
    fig.update_layout(
        height=420, legend=dict(orientation="h", y=-0.08),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Geo breakdown
    with st.expander("Impact by geography"):
        geo_df = pd.DataFrame({
            "Geography": d["geos"],
            "Current Deals": baseline_geo,
            "Projected Deals": sim_geo,
            "Change": sim_geo - baseline_geo,
        })
        st.dataframe(geo_df.style.format({
            "Current Deals": "{:,.0f}",
            "Projected Deals": "{:,.0f}",
            "Change": "{:+,.0f}",
        }), hide_index=True, use_container_width=True)

    # Export
    st.markdown("---")
    export_rows = []
    for i, ch in enumerate(channels):
        sc_row = scorecard[scorecard["Channel"] == ch].iloc[0]
        export_rows.append({
            "Channel": ch,
            "Current Weekly Spend": round(current_spend[i], 2),
            "Planned Weekly Spend": round(new_spend[i], 2),
            "Change (%)": f"{(multipliers[i]-1)*100:+.0f}%",
            "Efficiency Score": round(sc_row["Efficiency Score"]),
            "Recommendation": sc_row["Signal"],
            "Projected CW Impact (total)": round(delta_cw),
            "Scenario": scenario_key,
        })
    export_df = pd.DataFrame(export_rows)
    csv = export_df.to_csv(index=False).encode()

    st.download_button(
        "📥 Download Budget Plan",
        data=csv,
        file_name=f"nnn_budget_plan_{scenario_key.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )


def render_channel_deep_dive(d, scorecard):
    """Single-channel deep dive with response curve and synergy connections."""
    st.markdown("## Channel Deep Dive")
    model, channels, _, _, synergy_df = load_model()

    selected = st.selectbox(
        "Pick a channel to explore",
        channels,
        format_func=lambda ch: f"{CHANNEL_ICONS.get(ch,'')} {ch}",
        key="deep_dive_ch",
    )

    X = d["X"]
    B, T, C, D = X.shape
    X_base = X[:, max(0, T - 8):, :, :]
    c_idx = channels.index(selected)

    sc_row = scorecard[scorecard["Channel"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly Spend", f"${sc_row['Weekly Spend']:,.0f}")
    c2.metric("Budget Share", f"{sc_row['Budget Share']:.1f}%")
    c3.metric("Efficiency Score", f"{sc_row['Efficiency Score']:.0f}/100")
    c4.metric("Recommendation", sc_row["Signal"].split(" ", 1)[1])

    # Response curve
    st.markdown("#### Spend vs. Deals")
    st.caption("What happens if you scale this channel's spend up or down?")
    multipliers = np.linspace(0.0, 2.0, 21)
    curve_rows = []
    for mult in multipliers:
        mults = np.ones(len(channels))
        mults[c_idx] = mult
        cw_total, _ = run_model(model, X_base, mults, d)
        spend_dollar = d["current_spend"][c_idx] * mult
        curve_rows.append({
            "Spend ($/wk)": spend_dollar,
            "Multiplier": mult,
            "Projected Deals": cw_total,
        })
    curve_df = pd.DataFrame(curve_rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["Spend ($/wk)"], y=curve_df["Projected Deals"],
        mode="lines+markers", line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=5),
    ))
    current_dollar = d["current_spend"][c_idx]
    fig.add_vline(
        x=current_dollar, line_dash="dash", line_color="gray",
        annotation_text="Current spend", annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Weekly Spend ($)", yaxis_title="Projected Closed-Won Deals (total)",
        height=350, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Synergy connections
    st.markdown("#### Cross-Channel Synergies")
    st.caption(f"Which channels does {selected} influence — and which influence it?")

    syn_matrix = synergy_df.values
    ch_idx = channels.index(selected)

    outgoing = [(channels[j], syn_matrix[ch_idx, j])
                for j in range(len(channels)) if j != ch_idx]
    outgoing.sort(key=lambda x: -x[1])

    incoming = [(channels[i], syn_matrix[i, ch_idx])
                for i in range(len(channels)) if i != ch_idx]
    incoming.sort(key=lambda x: -x[1])

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"**{selected} boosts →**")
        for ch, w in outgoing[:5]:
            bar_len = int((w - 0.065) / 0.03 * 20)
            bar = "█" * max(1, bar_len)
            st.text(f"  {ch:20s} {bar}")
    with sc2:
        st.markdown(f"**← Boosted by**")
        for ch, w in incoming[:5]:
            bar_len = int((w - 0.065) / 0.03 * 20)
            bar = "█" * max(1, bar_len)
            st.text(f"  {ch:20s} {bar}")


def render_backtest(d):
    """Historical view — simplified for non-technical audience."""
    st.markdown("## Historical Accuracy")
    st.caption(
        "How well does the model predict actual deal closings? "
        "Strong historical fit = more trustworthy budget recommendations."
    )

    model, channels, cv, _, _ = load_model()
    X = d["X"]

    with torch.no_grad():
        pred_norm = model(X).squeeze(-1)
        pred_raw = pred_to_cw(pred_norm, d["y_mean"], d["y_std"])

    y_raw = d["y_raw"]
    dates = d["dates"]
    geos = d["geos"]

    geo_select = st.selectbox("Geography", ["All regions combined"] + geos, key="bt_geo")

    if geo_select == "All regions combined":
        actual = y_raw.sum(dim=0).numpy()
        predicted = pred_raw.sum(dim=0).numpy()
    else:
        g_idx = geos.index(geo_select)
        actual = y_raw[g_idx].numpy()
        predicted = pred_raw[g_idx].numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual, mode="lines+markers", name="Actual Deals",
        line=dict(color="steelblue", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted, mode="lines", name="Model Prediction",
        line=dict(color="#e74c3c", width=2, dash="dot"),
    ))
    fig.update_layout(
        height=400, yaxis_title="Closed-Won Deals",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    overall_error = np.mean(np.abs(actual - predicted) / (np.abs(actual) + 1.0)) * 100
    r2 = 1 - ((actual - predicted)**2).sum() / ((actual - actual.mean())**2).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Error", f"{overall_error:.1f}%")
    c2.metric("Prediction Accuracy (R²)", f"{r2:.1%}")
    c3.metric("Model Confidence", "High" if r2 > 0.85 else "Medium" if r2 > 0.7 else "Low")

    st.caption(
        "R² = how much of the weekly variation the model explains. "
        "Above 80% is considered strong for marketing measurement."
    )


# =========================================================================
# Main
# =========================================================================

def main():
    st.set_page_config(
        page_title="NNN Media Budget Planner",
        page_icon="📊",
        layout="wide",
    )

    st.markdown("# 📊 Media Budget Planner")
    st.caption("Powered by NNN — a neural network trained on 71 weeks of real spend and CRM data across 5 regions and 13 channels.")

    d = load_data()
    optim = load_optim()
    _, channels, _, _, _ = load_model()

    scorecard = build_channel_scorecard(channels, d["current_spend"], optim)

    # Headline insight
    render_headline(scorecard, d)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Channel Scorecard",
        "💰 Budget Planner",
        "🔎 Channel Deep Dive",
        "📈 Historical Accuracy",
    ])

    with tab1:
        render_scorecard(scorecard)

    with tab2:
        render_budget_planner(d, scorecard)

    with tab3:
        render_channel_deep_dive(d, scorecard)

    with tab4:
        render_backtest(d)


if __name__ == "__main__":
    main()
