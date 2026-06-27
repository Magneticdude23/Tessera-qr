"""
FastAPI backend for the vol-surface market-making console.

Responsibilities:
  * POST /api/surface   -- fit an arbitrage-free SVI surface (synthetic or live)
                           and return plot-ready JSON.
  * POST /api/desk-note -- SECURE OpenRouter proxy. The key lives here, in the
                           server environment, never in the browser. Per-IP rate
                           limiting + input caps + CORS guard the endpoint so a
                           leaked URL can't drain credits.
  * GET  /api/health    -- liveness.
  * /                   -- serves the static frontend (frontend/).

Run:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingest import synthetic_chain, fetch_chain_yfinance
from src.surface import fit_surface, surface_metrics
from src.llm import generate_desk_note, RateLimiter
from backend.serialize import serialize_surface

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Tessera", version="1.0")

# CORS: same-origin needs nothing, but if you split the frontend onto another
# host, list its origin in ALLOWED_ORIGINS (comma-separated). Default is locked
# to localhost rather than "*", on purpose.
_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------- #
# Request models (validation = first line of defence against oversized input)
# --------------------------------------------------------------------------- #
class SurfaceRequest(BaseModel):
    source: Literal["synthetic", "live"] = "synthetic"
    ticker: str = Field("SPY", max_length=12)
    r: float = Field(0.04, ge=-0.05, le=0.25)
    q: float = Field(0.01, ge=-0.05, le=0.25)
    seed: int = Field(7, ge=0, le=10_000)


class DeskNoteRequest(BaseModel):
    metrics: dict


# --------------------------------------------------------------------------- #
# Per-IP rate limiting for the paid endpoint
# --------------------------------------------------------------------------- #
from src.llm import DailyCap  # noqa: E402

_ip_buckets: dict[str, RateLimiter] = {}
_ip_daily: dict[str, DailyCap] = {}
_global_daily = DailyCap(int(os.environ.get("LLM_GLOBAL_PER_DAY", "40")))
_MAX_TRACKED_IPS = 5000


def _client_allowed(ip: str) -> tuple[bool, str]:
    """Layered caps: per-IP/min, per-IP/day, and a global/day ceiling."""
    if not _global_daily.allow():
        return False, "daily_limit"          # whole-app ceiling reached
    if ip not in _ip_buckets:
        if len(_ip_buckets) > _MAX_TRACKED_IPS:
            _ip_buckets.clear()
            _ip_daily.clear()                # crude reset; use Redis TTLs in prod
        _ip_buckets[ip] = RateLimiter(rate=4, per=60.0)   # 4/min per IP
        _ip_daily[ip] = DailyCap(8)                       # 8/day per IP
    if not _ip_daily[ip].allow():
        return False, "daily_limit"
    if not _ip_buckets[ip].allow():
        return False, "rate_limited"
    return True, ""


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": bool(os.environ.get("OPENROUTER_API_KEY"))
            or _secrets_has_key()}


def _secrets_has_key():
    try:
        import streamlit  # noqa
        return False
    except Exception:
        return False


@app.post("/api/surface")
def surface(req: SurfaceRequest):
    try:
        if req.source == "synthetic":
            chain = synthetic_chain(r=req.r, q=req.q, seed=req.seed,
                                    two_sided=True, parity_dislocation_bps=8.0)
            underlying = "SYNTHETIC"
        else:
            chain = fetch_chain_yfinance(req.ticker, r=req.r, q=req.q)
            underlying = req.ticker.upper()
        res = fit_surface(chain, underlying=underlying)
        metrics = surface_metrics(res)
        return serialize_surface(res, metrics)
    except Exception as e:
        return JSONResponse(status_code=502,
                            content={"error": f"surface_failed: {type(e).__name__}: {e}"})


@app.post("/api/desk-note")
def desk_note(req: DeskNoteRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    allowed, why = _client_allowed(ip)
    if not allowed:
        return JSONResponse(status_code=429,
                            content={"ok": False, "reason": why})
    # bound the payload before it ever reaches the model
    if len(json.dumps(req.metrics, default=str)) > 8000:
        return JSONResponse(status_code=413,
                            content={"ok": False, "reason": "payload_too_large"})
    out = generate_desk_note(req.metrics)
    status = 200 if out.ok else (503 if out.reason != "no_api_key" else 200)
    return JSONResponse(status_code=status,
                        content={"ok": out.ok, "text": out.text, "reason": out.reason})


# Serve the SPA last so /api/* keeps priority over the catch-all static mount.
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="static")
