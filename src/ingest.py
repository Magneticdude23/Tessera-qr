"""
Option-chain ingestion.

Two sources:

1. `fetch_chain_yfinance` -- live equity / ETF / index option chains from Yahoo
   via yfinance. Free, snapshot frequency (not tick), perfectly adequate for
   surface fitting and the static no-arbitrage checks. Run this on your own
   machine; Yahoo is not reachable from the sandbox this repo was built in.

2. `synthetic_chain` -- generates a chain from a *known* SVI surface plus quote
   noise and a bid/ask spread. This exists so the whole pipeline (invert ->
   fit -> check arb -> plot) can run deterministically offline, and so you have
   a ground-truth fixture to unit-test the inversion and calibration against.
   When the recovered SVI params are close to the ones you fed in, the stack works.

Both return a uniform dict the rest of the repo consumes:
    {
      "spot": float, "r": float, "q": float,
      "expiries": { T (years) : {
          "F": forward, "df": discount factor,
          "K": strikes[], "call": bool[], "mid": prices[]
      }}
    }
"""
from __future__ import annotations

import numpy as np

from .black_scholes import bs_price, forward_from_spot
from .svi import SVIParams, svi_total_variance_to_vol


# ---------------------------------------------------------------------------
# Synthetic (offline) chain
# ---------------------------------------------------------------------------
def synthetic_chain(
    spot=100.0, r=0.04, q=0.01,
    expiries_years=(0.08, 0.25, 0.5, 1.0),
    n_strikes=21, strike_width=0.45,
    noise_vol_bps=15.0, spread_bps=20.0, seed=0,
    two_sided=False, parity_dislocation_bps=0.0,
):
    """
    Build an arbitrage-free-ish chain from per-maturity SVI 'truth' params.

    The truth slices are deliberately equity-like: negative rho (downside skew),
    total variance rising with maturity (calendar-arb-free by construction), and
    the level term-structure roughly flat-to-upward. Quote noise is added in vol
    space (noise_vol_bps) and converted to price; spread_bps sets a symmetric
    bid/ask whose mid we return.
    """
    rng = np.random.default_rng(seed)

    # truth SVI params per maturity (a scales with T so total variance grows)
    def truth(T):
        return SVIParams(a=0.02 * T + 0.005, b=0.15 * np.sqrt(T) + 0.02,
                         rho=-0.35, m=0.0, sigma=0.12)

    out = {"spot": spot, "r": r, "q": q, "expiries": {}, "_truth": {}}
    for T in expiries_years:
        F = forward_from_spot(spot, r, q, T)
        df = np.exp(-r * T)
        p = truth(T)
        out["_truth"][T] = p

        k = np.linspace(-strike_width, strike_width, n_strikes)  # log-moneyness grid
        K = F * np.exp(k)
        vol_true = svi_total_variance_to_vol(k, p, T)
        vol_noisy = vol_true + rng.normal(0.0, noise_vol_bps * 1e-4, size=k.shape)
        vol_noisy = np.maximum(vol_noisy, 1e-3)

        # quote OTM options on each side (standard market convention): calls for
        # k > 0, puts for k < 0. Reduces the deep-ITM low-information quotes.
        call = k >= 0.0
        mid = bs_price(F, K, vol_noisy, T, df=df, call=call)

        # apply a symmetric spread and return the mid (so the spread is realistic
        # but we still hand back a single clean price the inverter consumes)
        half = mid * (spread_bps * 1e-4)
        bid = np.maximum(mid - half, 1e-6)
        ask = mid + half
        mid_quoted = 0.5 * (bid + ask)

        out["expiries"][float(T)] = {
            "F": float(F), "df": float(df),
            "K": K, "call": call, "mid": mid_quoted,
        }

        if two_sided:
            # full call AND put price at every strike, from the same noisy vol
            call_mid = bs_price(F, K, vol_noisy, T, df=df, call=True)
            put_mid = bs_price(F, K, vol_noisy, T, df=df, call=False)
            # optionally inject a put-call parity dislocation at one strike so the
            # scanner has a real violation to find (demonstration only)
            if parity_dislocation_bps:
                j = n_strikes // 2 + 2
                put_mid[j] += F * parity_dislocation_bps * 1e-4
            out["expiries"][float(T)]["call_mid"] = call_mid
            out["expiries"][float(T)]["put_mid"] = put_mid
    return out


# ---------------------------------------------------------------------------
# Live chain (run on your machine)
# ---------------------------------------------------------------------------
def fetch_chain_yfinance(ticker, r=0.04, q=0.0, max_expiries=6,
                         min_volume=0, drop_zero_bid=True):
    """
    Pull live option chains from Yahoo. Requires `pip install yfinance`.

    We back out the forward per expiry from put-call parity using the most
    at-the-money strike (more robust than trusting Yahoo's spot + a guessed
    carry). r is the assumed risk-free; q is solved implicitly via the parity
    forward, so the q argument is only a fallback if parity recovery fails.
    """
    import yfinance as yf  # local import so the sandbox demo never needs it

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    out = {"spot": spot, "r": r, "q": q, "expiries": {}}

    import datetime as dt
    today = dt.date.today()
    expiries = tk.options[:max_expiries]

    for exp in expiries:
        exp_date = dt.date.fromisoformat(exp)
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue
        chain = tk.option_chain(exp)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        for dfq in (calls, puts):
            dfq["mid"] = 0.5 * (dfq["bid"] + dfq["ask"])
            if drop_zero_bid:
                dfq.drop(dfq[dfq["bid"] <= 0].index, inplace=True)
            if min_volume:
                dfq.drop(dfq[dfq["volume"].fillna(0) < min_volume].index, inplace=True)

        # recover forward via put-call parity at the strike with smallest |C-P|
        merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
        if merged.empty:
            continue
        merged["parity_gap"] = (merged["mid_c"] - merged["mid_p"]).abs()
        atm = merged.iloc[merged["parity_gap"].argmin()]
        df_disc = np.exp(-r * T)
        # C - P = df*(F - K)  =>  F = K + (C - P)/df
        F = float(atm["strike"] + (atm["mid_c"] - atm["mid_p"]) / df_disc)

        # quote OTM side only: calls above F, puts below F
        c_otm = calls[calls["strike"] >= F]
        p_otm = puts[puts["strike"] < F]
        K = np.concatenate([p_otm["strike"].values, c_otm["strike"].values])
        mid = np.concatenate([p_otm["mid"].values, c_otm["mid"].values])
        is_call = np.concatenate([
            np.zeros(len(p_otm), dtype=bool), np.ones(len(c_otm), dtype=bool)
        ])
        order = np.argsort(K)
        out["expiries"][float(T)] = {
            "F": F, "df": float(df_disc),
            "K": K[order], "call": is_call[order], "mid": mid[order],
        }
    return out
