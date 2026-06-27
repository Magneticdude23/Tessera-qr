"""End-to-end test suite. Run with: pytest -q"""
import numpy as np
import pytest

from src.black_scholes import bs_price, bs_vega, bs_delta, bs_gamma, d1_d2
from src.iv import implied_vol_one, implied_vol
from src.svi import (SVIParams, svi_w, svi_w_prime, svi_w_double_prime,
                     butterfly_g, fit_svi_slice, arb_report)
from src.ingest import synthetic_chain
from src.surface import fit_surface, surface_metrics
from src.arbitrage import put_call_parity_scan, box_spread_implied_rates


# --------------------------- Black-Scholes ---------------------------------- #
def test_put_call_parity_in_pricer():
    F, K, sig, T, df = 100.0, 95.0, 0.2, 0.5, np.exp(-0.04 * 0.5)
    c = bs_price(F, K, sig, T, df=df, call=True)
    p = bs_price(F, K, sig, T, df=df, call=False)
    assert c - p == pytest.approx(df * (F - K), abs=1e-9)


def test_vega_matches_finite_difference():
    F, K, sig, T = 100.0, 100.0, 0.25, 0.5
    h = 1e-5
    fd = (bs_price(F, K, sig + h, T) - bs_price(F, K, sig - h, T)) / (2 * h)
    assert bs_vega(F, K, sig, T) == pytest.approx(float(fd), rel=1e-4)


def test_gamma_matches_finite_difference():
    F, K, sig, T = 100.0, 105.0, 0.3, 0.4
    h = 1e-3
    fd = (bs_price(F + h, K, sig, T) - 2 * bs_price(F, K, sig, T)
          + bs_price(F - h, K, sig, T)) / h**2
    assert bs_gamma(F, K, sig, T) == pytest.approx(float(fd), rel=1e-3)


# ------------------------------ Inversion ----------------------------------- #
@pytest.mark.parametrize("sig", [0.08, 0.15, 0.25, 0.45, 0.9])
def test_iv_round_trip(sig):
    F, K, T = 100.0, 110.0, 0.75
    px = float(bs_price(F, K, sig, T, df=1.0, call=True))
    rec = implied_vol_one(px, F, K, T, df=1.0, call=True)
    assert rec == pytest.approx(sig, abs=1e-6)


def test_iv_rejects_out_of_bounds_price():
    F, K, T = 100.0, 100.0, 0.5
    # price above the upper no-arb bound (forward) must return nan
    assert np.isnan(implied_vol_one(F + 1.0, F, K, T, df=1.0, call=True))


def test_iv_vectorised_filters_nans():
    F, T = 100.0, 0.5
    K = np.array([90.0, 100.0, 110.0])
    prices = bs_price(F, K, 0.2, T, df=1.0, call=True)
    prices[1] = 999.0  # poison one quote
    iv = implied_vol(prices, F, K, T, df=1.0, call=True)
    assert np.isnan(iv[1]) and not np.isnan(iv[0]) and not np.isnan(iv[2])


# -------------------------------- SVI --------------------------------------- #
def test_svi_analytic_derivatives_match_numeric():
    p = SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.05, sigma=0.15)
    k = np.linspace(-0.5, 0.5, 11)
    h = 1e-5
    wp_num = (svi_w(k + h, p) - svi_w(k - h, p)) / (2 * h)
    wpp_num = (svi_w(k + h, p) - 2 * svi_w(k, p) + svi_w(k - h, p)) / h**2
    assert np.allclose(svi_w_prime(k, p), wp_num, atol=1e-5)
    assert np.allclose(svi_w_double_prime(k, p), wpp_num, atol=1e-3)


def test_butterfly_detector_fires_on_bad_slice():
    good = SVIParams(a=0.04, b=0.12, rho=-0.35, m=0.0, sigma=0.12)
    bad = SVIParams(a=0.01, b=0.9, rho=-0.95, m=0.0, sigma=0.04)
    assert arb_report(good)["butterfly_free"] is True
    assert arb_report(bad)["butterfly_free"] is False
    assert arb_report(bad)["min_g"] < 0


def test_svi_recovers_truth_from_synthetic():
    chain = synthetic_chain(seed=3)
    T = 0.5
    slc = chain["expiries"][T]
    truth = chain["_truth"][T]
    from src.iv import implied_vol as iv_fn
    F, df, K, call, mid = (slc["F"], slc["df"], slc["K"], slc["call"], slc["mid"])
    iv = iv_fn(mid, F, K, T, df=df, call=call)
    ok = ~np.isnan(iv)
    k = np.log(K[ok] / F)
    w = iv[ok] ** 2 * T
    p, info = fit_svi_slice(k, w)
    assert info["success"]
    assert p.rho == pytest.approx(truth.rho, abs=0.08)
    assert info["rmse_vol_proxy"] < 5e-3  # < 50 vol bps


# ----------------------------- Arbitrage ------------------------------------ #
def test_parity_scan_flags_injected_dislocation():
    chain = synthetic_chain(seed=1, two_sided=True, parity_dislocation_bps=15.0)
    slc = chain["expiries"][0.25]
    rows = put_call_parity_scan(slc["K"], slc["call_mid"], slc["put_mid"],
                                slc["F"], slc["df"], flag_bps=5.0)
    assert any(r["flagged"] for r in rows)


def test_box_implied_rate_near_assumed():
    chain = synthetic_chain(seed=1, two_sided=True, r=0.04)
    slc = chain["expiries"][0.5]
    _, summ = box_spread_implied_rates(slc["K"], slc["call_mid"],
                                       slc["put_mid"], 0.5)
    assert summ["n_boxes"] > 0
    assert summ["median_rate"] == pytest.approx(0.04, abs=0.01)


# ------------------------------ Surface ------------------------------------- #
def test_full_surface_pipeline():
    chain = synthetic_chain(seed=5, two_sided=True, parity_dislocation_bps=8.0)
    res = fit_surface(chain, underlying="SYNTHETIC")
    m = surface_metrics(res)
    assert m["n_maturities"] == 4
    assert m["calendar_arbitrage_free"] is True
    assert all(v["free"] for v in m["butterfly_by_T"].values())
    # equity-like surface: downside skew positive, term structure defined
    assert all(s > 0 for s in m["skew_by_T"].values())
    assert m["term_structure_slope_vol"] is not None


# ----------------------------- Cost safety ---------------------------------- #
def test_free_model_detection():
    from src.llm import _is_free_model
    assert _is_free_model("openrouter/free")
    assert _is_free_model("meta-llama/llama-3.3-70b-instruct:free")
    assert not _is_free_model("anthropic/claude-3.5-haiku")
    assert not _is_free_model("openai/gpt-4o-mini")


def test_paid_model_blocked_in_free_only(monkeypatch):
    import src.llm as llm
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(llm, "FREE_ONLY", True)
    # a paid model must be refused BEFORE any network call
    out = llm.generate_desk_note({"x": 1}, model="anthropic/claude-3.5-haiku")
    assert out.ok is False and out.reason == "paid_model_blocked"


def test_daily_cap_blocks_after_limit():
    from src.llm import DailyCap
    cap = DailyCap(limit=2)
    assert cap.allow() and cap.allow()
    assert cap.allow() is False  # third call blocked


def test_no_key_means_no_call(monkeypatch):
    import src.llm as llm
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = llm.generate_desk_note({"x": 1}, model="openrouter/free")
    assert out.ok is False and out.reason == "no_api_key"
