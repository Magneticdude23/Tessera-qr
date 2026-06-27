"""
Raw SVI (Stochastic Volatility Inspired) slice, after Gatheral (2004) and the
arbitrage-free treatment in Gatheral & Jacquier (2014), "Arbitrage-free SVI
volatility surfaces", Quantitative Finance.

A single-maturity slice models *total implied variance* w(k) = sigma(k)^2 * T as

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

with k = log(K / F). The five parameters have direct geometric meaning:
    a      vertical level (overall variance floor)
    b      wing slope / how fast the smile opens up
    rho    skew / asymmetry, in (-1, 1); equity index smiles have rho < 0
    m      horizontal shift of the smile minimum
    sigma  curvature near the minimum (the ATM "smoothness")

NOTE the unfortunate name clash: SVI's 'sigma' parameter is curvature, NOT vol.

We carry analytic first and second derivatives so the butterfly-arbitrage
function g(k) is exact rather than finite-differenced -- a market maker checking
arb should not introduce numerical noise into the check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self):
        return (self.a, self.b, self.rho, self.m, self.sigma)


def svi_w(k, p: SVIParams):
    """Total implied variance w(k)."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def svi_w_prime(k, p: SVIParams):
    """dw/dk, analytic."""
    k = np.asarray(k, dtype=float)
    root = np.sqrt((k - p.m) ** 2 + p.sigma**2)
    return p.b * (p.rho + (k - p.m) / root)


def svi_w_double_prime(k, p: SVIParams):
    """d2w/dk2, analytic."""
    k = np.asarray(k, dtype=float)
    root = np.sqrt((k - p.m) ** 2 + p.sigma**2)
    return p.b * p.sigma**2 / root**3


def svi_total_variance_to_vol(k, p: SVIParams, T):
    """Convert the fitted slice back to Black-Scholes implied vol."""
    return np.sqrt(np.maximum(svi_w(k, p), 1e-12) / T)


# ---------------------------------------------------------------------------
# No-arbitrage diagnostics
# ---------------------------------------------------------------------------
def butterfly_g(k, p: SVIParams):
    """
    Gatheral's g(k). A slice is free of *butterfly* arbitrage iff g(k) >= 0 for
    all k (and call prices vanish as k -> +inf). g(k) is proportional to the
    risk-neutral density, so g(k) < 0 means the implied density goes negative --
    a butterfly spread with negative cost, i.e. free money / mispricing.

        g(k) = (1 - k w' / (2 w))^2  -  (w'^2 / 4) (1/w + 1/4)  +  w'' / 2
    """
    w = svi_w(k, p)
    wp = svi_w_prime(k, p)
    wpp = svi_w_double_prime(k, p)
    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = (wp**2 / 4.0) * (1.0 / w + 0.25)
    term3 = wpp / 2.0
    return term1 - term2 + term3


def risk_neutral_density(k, p: SVIParams):
    """
    Implied risk-neutral density of log-moneyness, p(k). Positive iff g(k) >= 0.

        p(k) = g(k) / sqrt(2 pi w(k)) * exp( -d_minus(k)^2 / 2 )
        d_minus(k) = -k / sqrt(w) - sqrt(w) / 2
    """
    w = svi_w(k, p)
    g = butterfly_g(k, p)
    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2.0
    return g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus**2)


def calendar_arbitrage_ok(slices, k_grid=None):
    """
    Calendar no-arb across maturities: total variance must be non-decreasing in T
    at every log-moneyness. `slices` is a list of (T, SVIParams) sorted by T.
    Returns (ok: bool, worst_violation: float). A negative worst_violation means
    an earlier maturity has higher total variance than a later one somewhere.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 201)
    slices = sorted(slices, key=lambda s: s[0])
    worst = np.inf
    for (T_lo, p_lo), (T_hi, p_hi) in zip(slices[:-1], slices[1:]):
        diff = svi_w(k_grid, p_hi) - svi_w(k_grid, p_lo)  # want >= 0 everywhere
        worst = min(worst, float(diff.min()))
    return (worst >= 0.0), worst


def arb_report(p: SVIParams, k_grid=None):
    """Convenience summary of butterfly arb for a single slice."""
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    g = butterfly_g(k_grid, p)
    min_g = float(g.min())
    return {
        "butterfly_free": bool(min_g >= 0.0),
        "min_g": min_g,
        "min_g_at_k": float(k_grid[np.argmin(g)]),
        "n_violation_points": int((g < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _initial_guess(k, w):
    """Cheap data-driven starting point for the optimiser."""
    a0 = max(float(np.min(w)) * 0.5, 1e-4)
    b0 = 0.1
    rho0 = -0.3  # equity skew prior
    m0 = float(k[np.argmin(w)])
    sigma0 = 0.1
    return np.array([a0, b0, rho0, m0, sigma0])


def fit_svi_slice(k, w, weights=None, enforce_no_butterfly=False):
    """
    Calibrate raw SVI to one maturity by weighted least squares on total variance.

    k, w   : log-moneyness and market total variance (sigma_mkt^2 * T) arrays.
    weights: optional per-quote weights (vega is a sensible choice -- it weights
             the fit toward strikes a maker actually trades). Defaults to equal.
    enforce_no_butterfly: if True, add a soft penalty pushing g(k) >= 0. We keep
             this OFF by default so the *checker* is demonstrated on a raw fit;
             turning it on shows how you'd repair an arbitrageable slice.

    Returns (SVIParams, fit_info_dict).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if weights is None:
        weights = np.ones_like(w)
    sqrt_w_obs = np.sqrt(weights)

    x0 = _initial_guess(k, w)
    # bounds: a>0, b>0, rho in (-1,1), m free-ish, sigma>0
    lb = np.array([1e-6, 1e-4, -0.999, -2.0, 1e-4])
    ub = np.array([np.max(w) + 1.0, 5.0, 0.999, 2.0, 2.0])

    k_dense = np.linspace(k.min() - 0.5, k.max() + 0.5, 200)

    def residuals(x):
        p = SVIParams(*x)
        base = sqrt_w_obs * (svi_w(k, p) - w)
        if not enforce_no_butterfly:
            return base
        g = butterfly_g(k_dense, p)
        penalty = 50.0 * np.minimum(g, 0.0)  # only negative g is penalised
        return np.concatenate([base, penalty])

    sol = least_squares(
        residuals, x0, bounds=(lb, ub), method="trf",
        xtol=1e-12, ftol=1e-12, max_nfev=5000,
    )
    p = SVIParams(*sol.x)
    fitted_w = svi_w(k, p)
    rmse_vol = float(np.sqrt(np.mean((np.sqrt(np.maximum(fitted_w, 1e-12))
                                      - np.sqrt(np.maximum(w, 1e-12))) ** 2)))
    info = {
        "success": bool(sol.success),
        "cost": float(sol.cost),
        "rmse_total_variance": float(np.sqrt(np.mean((fitted_w - w) ** 2))),
        "rmse_vol_proxy": rmse_vol,
        "params": p.as_tuple(),
    }
    return p, info
