"""
dashboard.py — NNN Strategy Command Center

Streamlit app for visualizing NNN model insights, running what-if simulations,
and exporting optimized budget plans.

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
        n_channels=len(channels),
        input_dim=258,
        d_model=cv["config"]["d_model"],
        n_heads=cv["config"]["n_heads"],
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
    y_mean = y_log.mean().item()
    y_std = y_log.std().item()

    spend_max = meta["normalization"]["max"][0]

    csv = pd.read_csv(
        ROOT / "data" / "real_marketing_data.csv",
        keep_default_na=False, na_values=[""],
    )
    current_spend = csv.groupby("Channel")["Spend"].mean()
    channel_names = meta["channels"]
    current_spend_vec = np.array([current_spend.get(ch, 0) for ch in channel_names])

    dates = [str(d).split(" ")[0] for d in meta["dates"]]

    return {
        "X": X, "y_raw": y_raw,
        "y_mean": y_mean, "y_std": y_std,
        "spend_max": spend_max,
        "current_spend": current_spend_vec,
        "geos": meta["geos"],
        "dates": dates,
        "channels": channel_names,
    }


@st.cache_data
def load_optim_results():
    with open(OUTPUT / "optimization_results.json") as f:
        return json.load(f)


def pred_to_cw(pred, y_mean, y_std):
    return torch.expm1(pred * y_std + y_mean)


# =========================================================================
# Synergy Network Graph
# =========================================================================

def render_synergy_network(synergy_df, channels):
    st.subheader("Cross-Channel Synergy Network")
    st.caption(
        "Edge thickness = attention weight between channels. "
        "The model learned these relationships from 71 weeks of real spend data."
    )

    threshold = st.slider(
        "Edge threshold (show only synergies above this weight)",
        min_value=0.075, max_value=0.095, value=0.082, step=0.001,
        format="%.3f", key="syn_thresh",
    )

    G = nx.DiGraph()
    G.add_nodes_from(channels)

    matrix = synergy_df.values
    for i, src in enumerate(channels):
        for j, tgt in enumerate(channels):
            if i != j and matrix[i, j] > threshold:
                G.add_edge(src, tgt, weight=float(matrix[i, j]))

    pos = nx.spring_layout(G, seed=42, k=2.5)

    edge_traces = []
    annotations = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = d["weight"]
        width = max(1, (w - threshold) / 0.005 * 3)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color="rgba(100,100,200,0.4)"),
            hoverinfo="text",
            text=f"{u} → {v}: {w:.4f}",
            showlegend=False,
        ))

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = (dx**2 + dy**2)**0.5 + 1e-8
        ax, ay = x1 - dx / length * 0.08, y1 - dy / length * 0.08
        annotations.append(dict(
            ax=mx, ay=my, x=ax, y=ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=width * 0.6, arrowcolor="rgba(80,80,180,0.6)",
        ))

    optim = load_optim_results()
    sens = optim.get("sensitivity", {})
    mrois = [sens.get(ch, {}).get("marginal_roi", 0) for ch in channels]
    max_mroi = max(mrois) if mrois else 1
    node_sizes = [15 + 25 * (m / max_mroi) for m in mrois]
    node_colors = mrois

    node_x = [pos[ch][0] for ch in channels]
    node_y = [pos[ch][1] for ch in channels]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=channels,
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="YlOrRd",
            colorbar=dict(title="mROI", thickness=15),
            line=dict(width=1.5, color="white"),
        ),
        hovertext=[
            f"{ch}<br>mROI: {mrois[i]:.5f}<br>Spend: ${sens.get(ch,{}).get('current_spend',0):,.0f}/wk"
            for i, ch in enumerate(channels)
        ],
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top synergy flows (above threshold):**")
    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    edges.sort(key=lambda x: -x[2])
    if edges:
        syn_table = pd.DataFrame(edges[:10], columns=["Source", "Target", "Weight"])
        syn_table["Weight"] = syn_table["Weight"].map("{:.4f}".format)
        st.dataframe(syn_table, hide_index=True, use_container_width=True)
    else:
        st.info("No edges above threshold. Lower the slider to reveal weaker synergies.")


# =========================================================================
# Response Curve Explorer
# =========================================================================

def render_response_curves(data_bundle):
    st.subheader("Spend Response Curves")
    st.caption("Each curve shows predicted Closed-Won deals as channel spend varies from 50% to 150% of current.")

    model, channels, cv, probe, _ = load_model()
    d = data_bundle
    X = d["X"]
    B, T, C, D = X.shape
    base_start = max(0, T - 8)
    X_base = X[:, base_start:, :, :]

    multipliers = np.linspace(0.5, 1.5, 21)

    selected = st.multiselect(
        "Select channels to compare",
        channels,
        default=["Google Non-Brand", "YouTube", "CTV", "Affiliates", "Facebook"],
        key="resp_channels",
    )

    if not selected:
        st.warning("Select at least one channel.")
        return

    rows = []
    for ch in selected:
        c_idx = channels.index(ch)

        for mult in multipliers:
            X_mod = X_base.clone()
            X_mod[:, :, c_idx, 0] = X_base[:, :, c_idx, 0] * mult
            with torch.no_grad():
                pred = model(X_mod).squeeze(-1)
                cw = pred_to_cw(pred, d["y_mean"], d["y_std"]).sum().item()
            rows.append({"Channel": ch, "Multiplier": mult, "Closed_Won": cw})

    df = pd.DataFrame(rows)

    fig = px.line(
        df, x="Multiplier", y="Closed_Won", color="Channel",
        labels={"Multiplier": "Spend Multiplier (1.0 = current)", "Closed_Won": "Predicted Closed-Won"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Current")
    fig.update_layout(height=450, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Marginal ROI bar chart
    optim = load_optim_results()
    sens = optim["sensitivity"]
    mroi_df = pd.DataFrame([
        {"Channel": ch, "Marginal ROI": sens[ch]["marginal_roi"]}
        for ch in channels
    ]).sort_values("Marginal ROI", ascending=True)

    fig2 = px.bar(
        mroi_df, x="Marginal ROI", y="Channel", orientation="h",
        color="Marginal ROI", color_continuous_scale="YlOrRd",
    )
    fig2.update_layout(height=400, showlegend=False, margin=dict(l=10))
    st.plotly_chart(fig2, use_container_width=True)


# =========================================================================
# Budget What-If Simulator
# =========================================================================

def render_whatif(data_bundle):
    st.subheader("Budget What-If Simulator")
    st.caption("Adjust channel spend below and see the predicted impact on Closed-Won deals.")

    model, channels, cv, probe, _ = load_model()
    d = data_bundle
    X = d["X"]
    B, T, C, D = X.shape
    base_start = max(0, T - 8)
    X_base = X[:, base_start:, :, :]

    with torch.no_grad():
        base_pred = model(X_base).squeeze(-1)
        baseline_cw = pred_to_cw(base_pred, d["y_mean"], d["y_std"]).sum().item()

    # Sliders inside the tab — two-column grid
    slider_col, result_col = st.columns([1, 2])

    adjustments = {}
    with slider_col:
        st.markdown("**Channel Spend Adjustments**")
        for ch in channels:
            current = d["current_spend"][channels.index(ch)]
            adj = st.slider(
                f"{ch} (${current:,.0f}/wk)",
                min_value=-50, max_value=100, value=0, step=5,
                format="%d%%", key=f"wif_{ch}",
            )
            adjustments[ch] = adj

    X_mod = X_base.clone()
    total_new_spend = 0
    total_old_spend = 0
    spend_details = []

    for i, ch in enumerate(channels):
        mult = 1.0 + adjustments[ch] / 100.0
        X_mod[:, :, i, 0] = X_base[:, :, i, 0] * mult

        current_dollar = d["current_spend"][i]
        new_dollar = current_dollar * mult
        total_old_spend += current_dollar
        total_new_spend += new_dollar
        spend_details.append({
            "Channel": ch,
            "Current $/wk": current_dollar,
            "Adjusted $/wk": new_dollar,
            "Change": f"{adjustments[ch]:+d}%",
        })

    with torch.no_grad():
        new_pred = model(X_mod).squeeze(-1)
        new_cw = pred_to_cw(new_pred, d["y_mean"], d["y_std"]).sum().item()

    delta_cw = new_cw - baseline_cw
    delta_pct = delta_cw / (baseline_cw + 1e-8) * 100

    with result_col:
        m1, m2 = st.columns(2)
        m1.metric("Baseline CW", f"{baseline_cw:,.0f}")
        m2.metric(
            "Simulated CW", f"{new_cw:,.0f}",
            delta=f"{delta_cw:+,.0f} ({delta_pct:+.1f}%)",
        )

        m3, m4 = st.columns(2)
        m3.metric("Current Budget", f"${total_old_spend:,.0f}/wk")
        m4.metric("Simulated Budget", f"${total_new_spend:,.0f}/wk")

        # Per-geo breakdown
        with torch.no_grad():
            base_geo = pred_to_cw(base_pred, d["y_mean"], d["y_std"]).sum(dim=1)
            new_geo = pred_to_cw(new_pred, d["y_mean"], d["y_std"]).sum(dim=1)

        geo_df = pd.DataFrame({
            "Geography": d["geos"],
            "Baseline CW": base_geo.numpy(),
            "Simulated CW": new_geo.numpy(),
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", x=geo_df["Geography"],
            y=geo_df["Baseline CW"], marker_color="lightslategray",
        ))
        fig.add_trace(go.Bar(
            name="Simulated", x=geo_df["Geography"],
            y=geo_df["Simulated CW"], marker_color="crimson",
        ))
        fig.update_layout(
            barmode="group", height=320,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Spend detail table
        with st.expander("Spend detail table"):
            st.dataframe(
                pd.DataFrame(spend_details),
                hide_index=True, use_container_width=True,
            )

        # Export
        export_df = pd.DataFrame(spend_details)
        export_df["Predicted_CW_Impact"] = delta_cw
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "Download Optimized Plan (CSV)",
            data=csv_bytes,
            file_name="nnn_budget_plan.csv",
            mime="text/csv",
        )


# =========================================================================
# Historical Backtesting
# =========================================================================

def render_backtest(data_bundle):
    st.subheader("Historical Backtest: Actual vs. Predicted Closed-Won")
    st.caption("71-week time series with CV Fold 3 outlier period highlighted.")

    model, channels, cv, probe, _ = load_model()
    d = data_bundle
    X = d["X"]

    with torch.no_grad():
        pred_norm = model(X).squeeze(-1)
        pred_raw = pred_to_cw(pred_norm, d["y_mean"], d["y_std"])

    y_raw = d["y_raw"]
    dates = d["dates"]
    geos = d["geos"]

    geo_select = st.selectbox("Geography", ["All (summed)"] + geos, key="bt_geo")

    if geo_select == "All (summed)":
        actual = y_raw.sum(dim=0).numpy()
        predicted = pred_raw.sum(dim=0).numpy()
    else:
        g_idx = geos.index(geo_select)
        actual = y_raw[g_idx].numpy()
        predicted = pred_raw[g_idx].numpy()

    bt_df = pd.DataFrame({
        "Week": dates,
        "Actual": actual,
        "Predicted": predicted,
        "Error": actual - predicted,
        "APE": np.abs(actual - predicted) / (np.abs(actual) + 1.0) * 100,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df["Week"], y=bt_df["Actual"],
        mode="lines+markers", name="Actual",
        line=dict(color="steelblue", width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=bt_df["Week"], y=bt_df["Predicted"],
        mode="lines+markers", name="Predicted",
        line=dict(color="crimson", width=2, dash="dot"),
        marker=dict(size=4),
    ))

    # Highlight Fold 3 validation period (weeks 50-59)
    if len(dates) > 59:
        fold3_start = dates[50]
        fold3_end = dates[min(59, len(dates) - 1)]
        fig.add_vrect(
            x0=fold3_start, x1=fold3_end,
            fillcolor="rgba(255,165,0,0.15)", line_width=0,
            annotation_text="Fold 3 (high error)",
            annotation_position="top left",
            annotation_font_color="darkorange",
        )

    fig.update_layout(
        height=450,
        yaxis_title="Closed-Won Deals",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    col1, col2, col3 = st.columns(3)
    overall_mape = bt_df["APE"].mean()
    r2_val = 1 - ((bt_df["Error"]**2).sum() / ((bt_df["Actual"] - bt_df["Actual"].mean())**2).sum())
    col1.metric("MAPE", f"{overall_mape:.1f}%")
    col2.metric("R²", f"{r2_val:.4f}")
    col3.metric("Mean Error", f"{bt_df['Error'].mean():+.1f} deals")

    if len(dates) > 59:
        st.markdown("#### Fold 3 Deep Dive (Weeks 50-59)")
        fold3_df = bt_df.iloc[50:60]

        fc1, fc2 = st.columns(2)
        with fc1:
            f3_mape = fold3_df["APE"].mean()
            st.metric("Fold 3 MAPE", f"{f3_mape:.1f}%")
        with fc2:
            f3_bias = fold3_df["Error"].mean()
            direction = "over-predicting" if f3_bias < 0 else "under-predicting"
            st.metric("Fold 3 Bias", f"{f3_bias:+.1f} ({direction})")

        st.markdown(
            "The model shows elevated error during this period. "
            "Common causes: seasonal budget shifts (Q4 freeze/thaw), "
            "campaign strategy changes, or CRM data lag."
        )

    # Residual plot
    with st.expander("Residual analysis"):
        fig_resid = go.Figure()
        colors = ["green" if abs(e) < bt_df["APE"].quantile(0.75) else "red" for e in bt_df["APE"]]
        fig_resid.add_trace(go.Bar(
            x=bt_df["Week"], y=bt_df["Error"],
            marker_color=colors,
            name="Residual",
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_resid.update_layout(height=300, yaxis_title="Actual - Predicted")
        st.plotly_chart(fig_resid, use_container_width=True)


# =========================================================================
# Model Summary
# =========================================================================

def render_summary():
    _, channels, cv, probe, _ = load_model()
    optim = load_optim_results()

    st.subheader("Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CV Mean MAPE", f"{cv['mean_mape']:.1f}%")
    col2.metric("CV Mean R²", f"{cv['mean_r2']:.4f}")
    col3.metric("Final R²", f"{cv['final_r2']:.4f}")
    col4.metric("Final MAPE", f"{cv['final_mape']:.1f}%")

    # Fold-by-fold
    fold_df = pd.DataFrame([
        {
            "Fold": f["fold"],
            "MAPE": f"{f['val_metrics']['mape']:.1f}%",
            "R²": f"{f['val_metrics']['r2']:.4f}",
            "Epochs": f["epochs_trained"],
        }
        for f in cv["folds"]
    ])
    st.dataframe(fold_df, hide_index=True, use_container_width=True)

    # Contributions
    contribs = probe["channel_contributions"]
    contrib_df = pd.DataFrame([
        {"Channel": ch, "Contribution": v}
        for ch, v in contribs.items()
    ]).sort_values("Contribution", ascending=True)

    fig = px.bar(
        contrib_df, x="Contribution", y="Channel", orientation="h",
        color="Contribution", color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Sparsity
    sparsity = probe["channel_sparsity"]
    stage_summary = probe["stage_summary"]

    st.markdown("**Attention Sparsity by Funnel Stage**")
    stage_df = pd.DataFrame([
        {"Stage": s.upper(), "Sparsity": f"{info['channel_attn_sparsity']:.1%}"}
        for s, info in stage_summary.items()
        if info["channel_attn_sparsity"] is not None
    ])
    st.dataframe(stage_df, hide_index=True, use_container_width=True)


# =========================================================================
# Main
# =========================================================================

def main():
    st.set_page_config(
        page_title="NNN Strategy Command Center",
        page_icon="::chart_with_upwards_trend::",
        layout="wide",
    )

    st.title("NNN Strategy Command Center")
    st.markdown("*Next-Generation Neural Network for Marketing Measurement*")

    data_bundle = load_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Summary",
        "Synergy Network",
        "Response Curves",
        "What-If Simulator",
        "Historical Backtest",
    ])

    with tab1:
        render_summary()

    with tab2:
        _, channels, _, _, synergy_df = load_model()
        render_synergy_network(synergy_df, channels)

    with tab3:
        render_response_curves(data_bundle)

    with tab4:
        render_whatif(data_bundle)

    with tab5:
        render_backtest(data_bundle)


if __name__ == "__main__":
    main()
