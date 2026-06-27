"""
Black-Scholes-Merton pricing and Greeks, vectorised over numpy arrays.

We work in forward terms throughout, because an options market maker prices
off the forward F = S * exp((r - q) * T), not the spot. Quoting in forward
space makes the dividend / financing assumption explicit and keeps the
implied-vol inversion clean (the forward absorbs r and q, so the smile is a
function of log-moneyness k = log(K / F) alone).

Convention used everywhere in this repo:
    k   = log(K / F)           log-moneyness (0 = at-the-money-forward)
    w   = sigma**2 * T         total implied variance
    vol = sigma                annualised Black-Scholes implied vol
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


SQRT_2PI = np.sqrt(2.0 * np.pi)


def d1_d2(F, K, sigma, T):
    """Black-Scholes d1, d2 in forward terms."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    vol_sqrt_t = sigma * np.sqrt(T)
    # guard against zero vol / zero T producing nan instead of a sane limit
    vol_sqrt_t = np.where(vol_sqrt_t <= 0.0, np.nan, vol_sqrt_t)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(F, K, sigma, T, df=1.0, call=True):
    """
    Undiscounted-forward Black price scaled by the discount factor df = exp(-rT).

    Passing df=1.0 returns the *forward* (undiscounted) option value, which is
    what you compare against forward-quoted mids. Pass the real discount factor
    to get the present value.
    """
    d1, d2 = d1_d2(F, K, sigma, T)
    call = np.asarray(call, dtype=bool)
    fwd_call = F * norm.cdf(d1) - K * norm.cdf(d2)
    fwd_put = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return df * np.where(call, fwd_call, fwd_put)


def bs_vega(F, K, sigma, T, df=1.0):
    """dPrice/dSigma. Identical for calls and puts. Per 1.00 of vol (not per 1%)."""
    d1, _ = d1_d2(F, K, sigma, T)
    return df * F * norm.pdf(d1) * np.sqrt(T)


def bs_delta(F, K, sigma, T, df=1.0, call=True):
    """Spot delta in forward terms (dPrice/dF). Multiply by dF/dS = exp((r-q)T) for spot delta."""
    d1, _ = d1_d2(F, K, sigma, T)
    call = np.asarray(call, dtype=bool)
    return df * np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def bs_gamma(F, K, sigma, T, df=1.0):
    """d2Price/dF2. Identical for calls and puts."""
    d1, _ = d1_d2(F, K, sigma, T)
    return df * norm.pdf(d1) / (F * sigma * np.sqrt(T))


def bs_theta(F, K, sigma, T, df=1.0, r=0.0, call=True):
    """Per-year theta. Simple forward-space approximation (carry term folded into F)."""
    d1, d2 = d1_d2(F, K, sigma, T)
    call = np.asarray(call, dtype=bool)
    term = -df * F * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    rate_call = -r * K * df * norm.cdf(d2)
    rate_put = r * K * df * norm.cdf(-d2)
    return term + np.where(call, rate_call, rate_put)


def forward_from_spot(S, r, q, T):
    """F = S * exp((r - q) * T)."""
    return S * np.exp((r - q) * T)
