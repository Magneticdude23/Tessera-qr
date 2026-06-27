"""
Surface orchestration. One call takes a raw chain (synthetic or live) all the way
to a fitted, arbitrage-checked surface plus a clean metrics dict. The metrics dict
is deliberately the *only* thing the LLM report layer ever sees -- it contains
finished numbers, never raw quotes, so the model phrases and never computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .black_scholes import bs_vega
from .iv import implied_vol
from .svi import (SVIParams, fit_svi_slice, arb_report, calendar_arbitrage_ok,
                  svi_total_variance_to_vol)
from .arbitrage import put_call_parity_scan, box_spread_implied_rates


@dataclass
class SliceFit:
    T: float
    k: np.ndarray            # market log-moneyness (clean quotes only)
    iv: np.ndarray           # market implied vol
    params: SVIParams
    info: dict
    arb: dict                # butterfly report


@dataclass
class SurfaceResult:
    underlying: str
    spot: float
    slices: list = field(default_factory=list)
    calendar_free: bool = True
    calendar_worst: float = 0.0
    parity_rows: list = field(default_factory=list)
    box_summary: dict = field(default_factory=dict)

    def slice_at(self, T):
        return min(self.slices, key=lambda s: abs(s.T - T))


def fit_surface(chain, underlying="SYNTHETIC", parity_flag_bps=5.0):
    """Fit every maturity, run butterfly + calendar + parity + box checks."""
    res = SurfaceResult(underlying=underlying, spot=float(chain["spot"]))

    fitted = []
    all_parity = []
    box_summ = {}
    for T, slc in sorted(chain["expiries"].items()):
        F, df, K, call, mid = (slc["F"], slc["df"], slc["K"],
                               slc["call"], slc["mid"])
        iv = implied_vol(mid, F, K, T, df=df, call=call)
        ok = ~np.isnan(iv)
        if ok.sum() < 5:
            continue
        k = np.log(K[ok] / F)
        iv_ok = iv[ok]
        w = iv_ok**2 * T
        vega = bs_vega(F, K[ok], iv_ok, T, df=df)
        p, info = fit_svi_slice(k, w, weights=vega)
        rep = arb_report(p)
        res.slices.append(SliceFit(T=T, k=k, iv=iv_ok, params=p, info=info, arb=rep))
        fitted.append((T, p))

        # parity + box need both-sided quotes; present on two-sided synthetic
        # chains and on live yfinance chains
        if "call_mid" in slc and "put_mid" in slc:
            rows = put_call_parity_scan(slc["K"], slc["call_mid"], slc["put_mid"],
                                        F, df, flag_bps=parity_flag_bps)
            for row in rows:
                row["T"] = T
            all_parity.extend(rows)
            _, summ = box_spread_implied_rates(slc["K"], slc["call_mid"],
                                               slc["put_mid"], T)
            box_summ[T] = summ

    if fitted:
        ok, worst = calendar_arbitrage_ok(fitted)
        res.calendar_free, res.calendar_worst = ok, worst
    all_parity.sort(key=lambda r: -abs(r["gap_bps"]))
    res.parity_rows = all_parity
    res.box_summary = box_summ
    return res


def _atm_vol(s: SliceFit):
    """Vol at k = 0 (at-the-money-forward) from the fitted slice."""
    return float(svi_total_variance_to_vol(np.array([0.0]), s.params, s.T)[0])


def _skew(s: SliceFit, span=0.1):
    """Downside minus upside vol over +/- span log-moneyness (vol points)."""
    down = float(svi_total_variance_to_vol(np.array([-span]), s.params, s.T)[0])
    up = float(svi_total_variance_to_vol(np.array([span]), s.params, s.T)[0])
    return down - up


def surface_metrics(res: SurfaceResult):
    """
    Distil the fitted surface into finished numbers for reporting. Everything
    here is already computed; the LLM layer must only phrase these, never derive.
    """
    Ts = [s.T for s in res.slices]
    atm = {round(s.T, 3): round(_atm_vol(s), 4) for s in res.slices}
    skew = {round(s.T, 3): round(_skew(s), 4) for s in res.slices}

    term_slope = None
    if len(res.slices) >= 2:
        s_lo = min(res.slices, key=lambda s: s.T)
        s_hi = max(res.slices, key=lambda s: s.T)
        term_slope = round(_atm_vol(s_hi) - _atm_vol(s_lo), 4)

    butterfly = {round(s.T, 3): {"free": bool(s.arb["butterfly_free"]),
                                 "min_g": round(s.arb["min_g"], 4)}
                 for s in res.slices}

    flagged = [r for r in res.parity_rows if r.get("flagged")]
    parity_top = [{"T": round(r["T"], 3), "K": round(r["K"], 2),
                   "gap_bps": round(r["gap_bps"], 2)} for r in flagged[:5]]

    box_rates = [v["median_rate"] for v in res.box_summary.values()
                 if v.get("n_boxes")]
    box = None
    if box_rates:
        box = {"median_implied_rate": round(float(np.median(box_rates)), 5),
               "n_maturities": len(box_rates)}

    fit_rmse = {round(s.T, 3): round(s.info["rmse_vol_proxy"] * 1e4, 1)
                for s in res.slices}  # in vol bps

    return {
        "underlying": res.underlying,
        "spot": round(res.spot, 2),
        "n_maturities": len(res.slices),
        "maturities_years": [round(t, 3) for t in Ts],
        "atm_vol_by_T": atm,
        "skew_by_T": skew,
        "term_structure_slope_vol": term_slope,
        "fit_rmse_vol_bps": fit_rmse,
        "butterfly_by_T": butterfly,
        "calendar_arbitrage_free": bool(res.calendar_free),
        "parity_violations_flagged": len(flagged),
        "parity_top": parity_top,
        "box_implied_financing": box,
    }
