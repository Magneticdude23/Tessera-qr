"""
Static no-arbitrage scanners that police a chain for "free money".

These are the relationships a derivatives desk monitors continuously. The honest
finding is almost always that apparent violations are tiny and vanish once you
account for the bid/ask spread, borrow cost, and discrete dividends -- so each
scanner reports the *gross* dislocation and leaves the net judgement explicit.

Put-call parity (forward form):     C - P = df * (F - K)
A long box spread (K1 < K2) locks a known payoff (K2 - K1) at expiry, so its
cost must equal df * (K2 - K1); inverting gives the market-implied financing rate.
"""
from __future__ import annotations

import numpy as np


def put_call_parity_scan(K, call_mid, put_mid, F, df, flag_bps=5.0):
    """
    For every strike with both a call and a put, measure the parity gap

        gap = (C - P) - df * (F - K)

    Returned in price and in basis points of the forward. Rows with |gap| above
    flag_bps are marked as candidate dislocations (gross, pre-cost).

    Returns a list of dicts sorted by descending |gap_bps|.
    """
    K = np.asarray(K, float)
    C = np.asarray(call_mid, float)
    P = np.asarray(put_mid, float)
    gap = (C - P) - df * (F - K)
    gap_bps = gap / F * 1e4
    rows = []
    for i in range(K.size):
        rows.append({
            "K": float(K[i]),
            "gap": float(gap[i]),
            "gap_bps": float(gap_bps[i]),
            "flagged": bool(abs(gap_bps[i]) >= flag_bps),
        })
    rows.sort(key=lambda r: -abs(r["gap_bps"]))
    return rows


def box_spread_implied_rates(K, call_mid, put_mid, T, max_pairs=12):
    """
    Back out the market-implied financing rate from box spreads.

    For a strike pair (K1 < K2), a long box costs
        cost = (C(K1) - C(K2)) + (P(K2) - P(K1))
    and pays (K2 - K1) with certainty, so
        cost = (K2 - K1) * exp(-r_implied * T)  =>  r_implied = -ln(cost/(K2-K1)) / T

    A coherent surface returns a tight cluster of implied rates; a wide spread of
    implied rates across pairs is the signal that some box is mispriced (or that
    your discounting / dividend assumption is off). Returns (rows, summary).
    """
    K = np.asarray(K, float)
    C = np.asarray(call_mid, float)
    P = np.asarray(put_mid, float)
    order = np.argsort(K)
    K, C, P = K[order], C[order], P[order]

    rows = []
    n = K.size
    # sample evenly spaced pairs so we don't return O(n^2) near-duplicates
    lefts = np.linspace(0, n - 2, min(max_pairs, max(n - 1, 1)), dtype=int)
    for i in lefts:
        j = n - 1 - i
        if j <= i:
            continue
        width = K[j] - K[i]
        cost = (C[i] - C[j]) + (P[j] - P[i])
        if width <= 0 or cost <= 0 or cost >= width:
            continue  # outside the no-arb band -> rate undefined / degenerate
        r_imp = -np.log(cost / width) / T
        rows.append({
            "K_low": float(K[i]), "K_high": float(K[j]),
            "cost": float(cost), "width": float(width),
            "implied_rate": float(r_imp),
        })
    if rows:
        rates = np.array([r["implied_rate"] for r in rows])
        summary = {
            "n_boxes": len(rows),
            "min_rate": float(rates.min()),
            "max_rate": float(rates.max()),
            "median_rate": float(np.median(rates)),
            "rate_spread_bps": float((rates.max() - rates.min()) * 1e4),
        }
    else:
        summary = {"n_boxes": 0, "min_rate": np.nan, "max_rate": np.nan,
                   "median_rate": np.nan, "rate_spread_bps": np.nan}
    return rows, summary
