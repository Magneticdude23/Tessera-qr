"""
Vol-surface market-making research console.

A Streamlit front end over the src/ engine: pick a data source, fit an
arbitrage-free SVI surface, inspect the smile / butterfly / density per maturity,
view the 3D surface, scan for put-call-parity and box dislocations, and (if an
OpenRouter key is configured server-side) generate a desk-style commentary that
phrases the computed metrics.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud or Hugging Face Spaces (see README).

The key is read server-side from st.secrets / env and never reaches the browser.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ingest import synthetic_chain, fetch_chain_yfinance
from src.surface import fit_surface, surface_metrics
from src.svi import (svi_total_variance_to_vol, butterfly_g,
                     risk_neutral_density)
from src.llm import generate_desk_note

st.set_page_config(page_title="Tessera", layout="wide",
                   page_icon="📈")


# --------------------------------------------------------------------------- #
# Data + fit (cached on the controls, not the unhashable chain dict)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_and_fit(source, ticker, r, q, seed):
    if source == "Synthetic (offline)":
        chain = synthetic_chain(r=r, q=q, seed=seed, two_sided=True,
                                parity_dislocation_bps=8.0)
        underlying = "SYNTHETIC"
    else:
        chain = fetch_chain_yfinance(ticker, r=r, q=q)
        underlying = ticker.upper()
    res = fit_surface(chain, underlying=underlying)
    return res, surface_metrics(res)


def smile_density_figure(s):
    k = np.linspace(s.k.min() - 0.1, s.k.max() + 0.1, 240)
    iv_fit = svi_total_variance_to_vol(k, s.params, s.T) * 100
    g = butterfly_g(k, s.params)
    dens = risk_neutral_density(k, s.params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.k, y=s.iv * 100, mode="markers",
                             name="market IV", marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=k, y=iv_fit, mode="lines", name="SVI fit"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Smile  T={s.T:.2f}y", yaxis_title="IV (%)",
                      xaxis_title="log-moneyness k")
    return fig, k, g, dens


def gk_density_figure(k, g, dens):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=g, mode="lines", name="g(k)"))
    fig.add_trace(go.Scatter(x=k, y=dens, mode="lines", name="RN density",
                             yaxis="y2"))
    fig.add_hline(y=0, line_dash="dash", line_color="#c0392b")
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        title="Butterfly g(k) & risk-neutral density",
        xaxis_title="log-moneyness k",
        yaxis=dict(title="g(k)"),
        yaxis2=dict(title="density", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def surface_3d_figure(res):
    Ts = sorted(s.T for s in res.slices)
    k_grid = np.linspace(-0.4, 0.4, 40)
    Z = []
    for T in Ts:
        s = res.slice_at(T)
        Z.append(svi_total_variance_to_vol(k_grid, s.params, T) * 100)
    fig = go.Figure(data=[go.Surface(
        x=k_grid, y=Ts, z=np.array(Z), colorscale="Viridis",
        colorbar=dict(title="IV %"))])
    fig.update_layout(
        height=480, margin=dict(l=0, r=0, t=30, b=0),
        title="Implied-volatility surface",
        scene=dict(xaxis_title="log-moneyness k", yaxis_title="T (years)",
                   zaxis_title="IV (%)"))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Tessera")
st.sidebar.caption("Arbitrage-free SVI surface & market-making diagnostics")

source = st.sidebar.radio("Data source",
                          ["Synthetic (offline)", "Live (yfinance)"])
ticker = st.sidebar.text_input("Ticker", value="SPY",
                               disabled=(source == "Synthetic (offline)"))
r = st.sidebar.number_input("Risk-free r", value=0.04, step=0.005, format="%.3f")
q = st.sidebar.number_input("Dividend yield q", value=0.01, step=0.005,
                            format="%.3f")
seed = st.sidebar.number_input("Synthetic seed", value=7, step=1)

if source == "Live (yfinance)":
    st.sidebar.info("Live mode needs `yfinance` and outbound internet; run "
                    "locally or on a host that allows it.")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Tessera — volatility surface console")

try:
    res, metrics = load_and_fit(source, ticker, r, q, int(seed))
except Exception as e:  # live data can fail; keep the app alive
    st.error(f"Could not build the surface: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Underlying", metrics["underlying"])
c2.metric("Maturities fitted", metrics["n_maturities"])
c3.metric("Calendar arb-free", "yes" if metrics["calendar_arbitrage_free"] else "NO")
all_bf = all(v["free"] for v in metrics["butterfly_by_T"].values())
c4.metric("Butterfly arb-free", "all" if all_bf else "VIOLATION")

st.plotly_chart(surface_3d_figure(res), width='stretch')

st.subheader("Per-maturity diagnostics")
labels = [f"{s.T:.2f}y" for s in res.slices]
tabs = st.tabs(labels)
for tab, s in zip(tabs, res.slices):
    with tab:
        fig1, k, g, dens = smile_density_figure(s)
        fig2 = gk_density_figure(k, g, dens)
        left, right = st.columns(2)
        left.plotly_chart(fig1, width='stretch')
        right.plotly_chart(fig2, width='stretch')
        a, b, c = st.columns(3)
        a.metric("ATM vol", f"{metrics['atm_vol_by_T'][round(s.T,3)]*100:.1f}%")
        b.metric("Skew (down−up, 10%)",
                 f"{metrics['skew_by_T'][round(s.T,3)]*100:.2f} pts")
        c.metric("Fit RMSE", f"{metrics['fit_rmse_vol_bps'][round(s.T,3)]:.1f} bps")
        if not s.arb["butterfly_free"]:
            st.warning(f"Butterfly arbitrage: min g(k) = {s.arb['min_g']:.4f}")

# Arbitrage policing
st.subheader("Static arbitrage scan")
colp, colb = st.columns(2)
with colp:
    st.markdown("**Put-call parity dislocations** (gross, pre-cost)")
    if metrics["parity_top"]:
        st.dataframe(metrics["parity_top"], width='stretch')
        st.caption("Most apparent violations close once spread, borrow and "
                   "dividends are netted out.")
    else:
        st.success("No parity dislocations beyond threshold.")
with colb:
    st.markdown("**Box-spread implied financing rate**")
    if metrics["box_implied_financing"]:
        st.metric("Median implied rate",
                  f"{metrics['box_implied_financing']['median_implied_rate']*100:.2f}%")
        st.caption(f"Assumed r = {r*100:.2f}%. A wide spread of box-implied "
                   "rates flags a mispriced strike pair.")
    else:
        st.info("Box rates need two-sided quotes (synthetic mode supplies them).")

# LLM desk note
st.subheader("Desk commentary")
st.caption("The model only *phrases* the computed metrics above — it never "
           "computes or invents a figure. Key stays server-side.")
if st.button("Generate desk note"):
    with st.spinner("Drafting…"):
        out = generate_desk_note(metrics)
    if out.ok:
        st.markdown(out.text)
    elif out.reason == "no_api_key":
        st.info("Commentary disabled: no OPENROUTER_API_KEY configured. "
                "Add it to .streamlit/secrets.toml or the host's secrets.")
    elif out.reason == "rate_limited":
        st.warning("Rate limit reached — try again shortly.")
    else:
        st.error(f"Commentary unavailable ({out.reason}).")

with st.expander("Raw metrics passed to the model"):
    st.json(metrics)
