"""
End-to-end starter demo.

Runs the full step-1 pipeline on a synthetic (offline, deterministic) chain:
    ingest  ->  invert to implied vol  ->  fit raw SVI per maturity
            ->  run the butterfly no-arb check  ->  check calendar no-arb
            ->  plot smile / g(k) / density for each slice.

Then it demonstrates the arb *detector* firing by hand-building a deliberately
arbitrageable slice and showing g(k) and the implied density go negative.

To run against live data instead, swap `synthetic_chain()` for
`fetch_chain_yfinance("SPY")` (needs `pip install yfinance`, run locally).

    python -m scripts.run_demo
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import synthetic_chain  # noqa: E402
from src.iv import implied_vol  # noqa: E402
from src.svi import (SVIParams, fit_svi_slice, arb_report,  # noqa: E402
                     calendar_arbitrage_ok)
from src.black_scholes import bs_vega  # noqa: E402
from src.plotting import slice_figure  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    chain = synthetic_chain(seed=7)
    print(f"spot={chain['spot']}  r={chain['r']}  q={chain['q']}")
    print(f"{len(chain['expiries'])} expiries: "
          f"{[round(T, 3) for T in chain['expiries']]}\n")

    fitted_slices = []
    for T, slc in sorted(chain["expiries"].items()):
        F, df, K, call, mid = slc["F"], slc["df"], slc["K"], slc["call"], slc["mid"]

        # 1. invert prices -> implied vol (with arb-bound + vega filtering)
        iv = implied_vol(mid, F, K, T, df=df, call=call)
        ok = ~np.isnan(iv)
        k = np.log(K[ok] / F)
        iv = iv[ok]
        w = iv**2 * T  # total variance

        # 2. fit raw SVI, vega-weighted (weights a maker's traded strikes)
        vega = bs_vega(F, K[ok], iv, T, df=df)
        p, info = fit_svi_slice(k, w, weights=vega)

        # 3. butterfly no-arb check
        rep = arb_report(p)
        fitted_slices.append((T, p))

        truth = chain["_truth"][T]
        print(f"T={T:5.2f} | n_quotes={ok.sum():2d} | "
              f"RMSE(vol)={info['rmse_vol_proxy']*1e4:5.1f}bps | "
              f"butterfly_free={rep['butterfly_free']} "
              f"(min g={rep['min_g']:+.4f}) | "
              f"rho_fit={p.rho:+.3f} (truth {truth.rho:+.3f})")

        slice_figure(k, iv, p, T,
                     suptitle=f"Synthetic slice  T={T:.2f}y   "
                              f"(butterfly-free: {rep['butterfly_free']})",
                     path=os.path.join(OUT, f"slice_T{T:.2f}.png"))

    # 4. calendar no-arb across the fitted surface
    cal_ok, worst = calendar_arbitrage_ok(fitted_slices)
    print(f"\ncalendar-arbitrage-free across surface: {cal_ok} "
          f"(worst Δ total-variance = {worst:+.5f}; want ≥ 0)")

    # 5. demonstrate the detector firing on a deliberately bad slice
    print("\n--- arb detector sanity check ---")
    # large b + extreme rho near a short maturity -> butterfly violation
    bad = SVIParams(a=0.01, b=0.9, rho=-0.95, m=0.0, sigma=0.05)
    bad_rep = arb_report(bad)
    print(f"hand-built bad slice: butterfly_free={bad_rep['butterfly_free']} "
          f"(min g={bad_rep['min_g']:+.4f} at k={bad_rep['min_g_at_k']:+.3f}, "
          f"{bad_rep['n_violation_points']} violation pts)")

    k_dummy = np.linspace(-0.4, 0.4, 21)
    iv_dummy = np.sqrt(np.maximum(
        np.array([bad.a + bad.b * (bad.rho * (kk - bad.m)
                  + np.sqrt((kk - bad.m) ** 2 + bad.sigma ** 2))
                  for kk in k_dummy]), 1e-9) / 0.1)
    slice_figure(k_dummy, iv_dummy, bad, 0.1,
                 suptitle="Deliberately arbitrageable slice (detector should fire)",
                 path=os.path.join(OUT, "slice_arbitrageable.png"))
    print(f"\nfigures written to: {OUT}")


if __name__ == "__main__":
    main()
