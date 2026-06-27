"""
Implied-volatility inversion.

Black-Scholes is not invertible in closed form, so we bracket-and-solve. We use
Brent's method (scipy.optimize.brentq) rather than Newton-Raphson because Newton
divides by vega, and vega collapses to ~0 for deep in/out-of-the-money options --
exactly the strikes where a smile fit needs clean data. Brent needs no derivative
and is globally convergent on a bracket, so it degrades gracefully.

Before inverting we filter quotes that violate the static no-arbitrage price
bounds, because feeding an out-of-bounds price to the solver yields a garbage vol
(or no root) and poisons the SVI fit downstream. For a call in forward terms the
price must satisfy   max(F - K, 0) * df  <=  C  <=  F * df.
The mirror bound holds for puts. We also drop quotes whose vega is below a floor,
since their implied vol is informationally useless (the price barely moves with vol).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import bs_price, bs_vega


def _within_arb_bounds(price, F, K, df, call):
    intrinsic = np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0)) * df
    upper = np.where(call, F, K) * df
    # strict-ish: allow a tiny epsilon so exactly-at-bound quotes survive rounding
    eps = 1e-9
    return (price > intrinsic - eps) & (price < upper + eps)


def implied_vol_one(price, F, K, T, df=1.0, call=True, lo=1e-4, hi=5.0):
    """Single-quote inversion. Returns np.nan if no root in [lo, hi]."""
    if not _within_arb_bounds(price, F, K, df, call):
        return np.nan

    def objective(sigma):
        return bs_price(F, K, sigma, T, df=df, call=call) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return np.nan  # price not bracketed -> no admissible vol
    try:
        return brentq(objective, lo, hi, xtol=1e-8, rtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol(price, F, K, T, df=1.0, call=True, vega_floor=1e-4):
    """
    Vectorised inversion with bound + vega filtering.

    Returns an array of implied vols (np.nan where the quote was rejected). The
    caller should drop the nans before fitting; the reason a quote is nan is
    either an arbitrage-bound violation or a degenerate-vega strike.
    """
    price = np.atleast_1d(np.asarray(price, dtype=float))
    F = np.broadcast_to(np.asarray(F, dtype=float), price.shape)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    T = np.broadcast_to(np.asarray(T, dtype=float), price.shape)
    call = np.broadcast_to(np.asarray(call, dtype=bool), price.shape)
    df = float(df)

    out = np.full(price.shape, np.nan)
    for i in range(price.size):
        iv = implied_vol_one(price[i], F[i], K[i], T[i], df=df, call=bool(call[i]))
        if np.isnan(iv):
            continue
        # reject strikes whose vega is too small for the vol to be meaningful
        if bs_vega(F[i], K[i], iv, T[i], df=df) < vega_floor:
            continue
        out[i] = iv
    return out
