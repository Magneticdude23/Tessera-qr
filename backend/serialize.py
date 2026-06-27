"""
Serialize a fitted SurfaceResult into plot-ready JSON for the browser.

The frontend never recomputes anything: every curve it draws (market smile points,
the dense SVI fit, g(k), the risk-neutral density, the 3D surface mesh) is computed
here, server-side, and shipped as plain numbers. This keeps the numerics in one
place (scipy) and keeps the client a thin rendering layer.
"""
from __future__ import annotations

import numpy as np

from src.svi import (svi_total_variance_to_vol, butterfly_g,
                     risk_neutral_density)


def _round_list(arr, nd):
    return [round(float(x), nd) for x in np.asarray(arr).ravel()]


def serialize_surface(res, metrics):
    slices = []
    for s in res.slices:
        kd = np.linspace(float(s.k.min()) - 0.05, float(s.k.max()) + 0.05, 121)
        vol_fit = svi_total_variance_to_vol(kd, s.params, s.T) * 100.0
        g = butterfly_g(kd, s.params)
        dens = risk_neutral_density(kd, s.params)
        p = s.params
        slices.append({
            "T": round(s.T, 4),
            "label": f"{s.T:.2f}y",
            "market_k": _round_list(s.k, 5),
            "market_iv": _round_list(s.iv * 100.0, 4),
            "fit_k": _round_list(kd, 5),
            "fit_vol": _round_list(vol_fit, 4),
            "g": _round_list(g, 6),
            "density": _round_list(dens, 6),
            "butterfly_free": bool(s.arb["butterfly_free"]),
            "min_g": round(float(s.arb["min_g"]), 5),
            "atm_vol": round(metrics["atm_vol_by_T"][round(s.T, 3)] * 100, 2),
            "skew": round(metrics["skew_by_T"][round(s.T, 3)] * 100, 2),
            "rmse_bps": metrics["fit_rmse_vol_bps"][round(s.T, 3)],
            "params": {"a": round(p.a, 5), "b": round(p.b, 5),
                       "rho": round(p.rho, 4), "m": round(p.m, 4),
                       "sigma": round(p.sigma, 4)},
        })

    Ts = sorted(s.T for s in res.slices)
    ks = np.linspace(-0.4, 0.4, 40)
    Z = []
    for T in Ts:
        s = res.slice_at(T)
        Z.append(_round_list(svi_total_variance_to_vol(ks, s.params, T) * 100, 4))

    return {
        "metrics": metrics,
        "slices": slices,
        "surface3d": {"k": _round_list(ks, 4),
                      "T": [round(float(t), 4) for t in Ts],
                      "z": Z},
        "parity": metrics["parity_top"],
        "box": metrics["box_implied_financing"],
        "calendar_free": metrics["calendar_arbitrage_free"],
    }
