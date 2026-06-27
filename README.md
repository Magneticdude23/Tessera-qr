# Tessera — an arbitrage-free volatility-surface & options market-making console

> *A tessera is a single tile in a mosaic.* The implied-volatility surface is
> assembled the same way — one calibrated maturity slice at a time — into a
> coherent, arbitrage-free whole.

A from-scratch implied-volatility surface engine and full-stack market-making
research console, built around one idea: **a derivatives market maker maintains an
arbitrage-free fair-value model for a derivative, quotes two-sided around it, and
manages the Greek risk the quoting leaves behind.** Everything radiates from the
object an options MM actually lives inside — the **volatility surface**.

The console fits the surface, checks it for static arbitrage, renders the 3D
surface and per-maturity smile / risk-neutral-density, scans for put-call-parity
and box-spread dislocations, and (optionally) drafts a desk-style commentary with
an LLM that only *phrases* the computed numbers.

## Stack

```
frontend/   vanilla HTML + CSS + JS, Plotly.js charts        (thin render layer)
backend/    FastAPI: surface API + SECURE OpenRouter proxy   (numerics + secrets)
src/        the engine: pricing, IV inversion, SVI, arbitrage (pure Python)
tests/      pytest suite                                       (CI on 3.10–3.12)
```

```
browser ──▶ FastAPI  ──▶ scipy engine (src/)         every number computed here
   │         │       ──▶ OpenRouter (key server-side, rate-limited)
   │         ▼
   └── only ever calls our own /api/* — never OpenRouter, never sees the key
```

## Why it's built this way

No-arbitrage is the part that separates *reading* about vol surfaces from
*building* one. Two static conditions are checked, each with an exact meaning:

- **No butterfly arbitrage** means the risk-neutral density is non-negative
  (Gatheral's `g(k) >= 0`). Where `g(k) < 0`, a butterfly spread has negative
  cost — the model is quoting a negative probability. The density panel is the
  console's signature: it turns red and fills the negative region the instant a
  slice goes arbitrageable.
- **No calendar arbitrage** means total implied variance `w(k) = sigma(k)^2 * T`
  is non-decreasing in maturity at every log-moneyness.

The static scanners (parity, box) report *gross* dislocations and say plainly that
most close once spread, borrow and dividends net out — that honesty is the point.

## Project layout

```
tessera/
|-- frontend/
|   |-- index.html
|   |-- css/styles.css            # quant-terminal theme (IBM Plex, ink-navy)
|   |-- js/{api,charts,app}.js     # fetch client, Plotly rendering, controller
|-- backend/
|   |-- main.py                   # FastAPI: /api/surface, /api/desk-note, static
|   |-- serialize.py              # fitted surface -> plot-ready JSON
|-- src/
|   |-- black_scholes.py          # forward-space pricing + Greeks
|   |-- iv.py                     # Brent IV inversion + arb-bound/vega filtering
|   |-- svi.py                    # raw SVI, analytic w'/w'', g(k), RN density, fit
|   |-- arbitrage.py              # put-call parity + box-spread implied rates
|   |-- ingest.py                 # yfinance live chains + synthetic generator
|   |-- surface.py                # orchestrator: fit all slices, checks, metrics
|   |-- llm.py                    # SECURE OpenRouter client (key + rate limit)
|   |-- plotting.py               # matplotlib figures for the offline script
|-- scripts/run_demo.py           # offline end-to-end demo -> output/*.png
|-- tests/test_suite.py           # pricing, inversion, SVI, arbitrage, surface
|-- app.py                        # OPTIONAL single-file Streamlit runner
|-- Dockerfile  .env.example  requirements.txt  .github/workflows/ci.yml
```

## Run it

```bash
pip install -r requirements.txt
pip install pytest httpx          # dev extras

# 1. the full-stack console (numerics + proxy + frontend, one server)
uvicorn backend.main:app --reload --port 8000
#   -> open http://localhost:8000

# 2. offline correctness demo (deterministic, no network) -> figures in output/
python -m scripts.run_demo

# 3. tests
pytest -q
```

The console boots in **synthetic** mode and fits a surface on load — no keys or
network required. Switch to **live** and enter a ticker (SPY, QQQ, ^SPX) to fit a
real chain; the per-expiry forward is recovered from put-call parity, not a quoted
spot. (Live mode needs `pip install yfinance` and outbound internet.)

## The OpenRouter layer — secure and free by default

- **Cannot be charged.** The default model is OpenRouter's free router
  (`openrouter/free`), and **free-only mode is ON** unless you explicitly set
  `OPENROUTER_ALLOW_PAID=1`. The backend refuses to call any non-free model
  *before* making a request. Pair a `$0`-balance OpenRouter key with a `:free`
  model and there is nothing to bill against — the feature is genuinely free.
- **Hard caps**, set deliberately below OpenRouter's own free limits (20/min,
  50/day): per-IP 4/min and 8/day, plus a global 40/day ceiling. When a cap is
  hit the button shows "daily free-tier limit reached," never an error or a bill.
- The key is read **server-side only** (backend env / `.env`); the browser calls
  `POST /api/desk-note` and never sees the key or talks to OpenRouter directly.
- The model receives a **finished metrics dict** and is instructed to phrase it —
  never to compute, estimate, or invent a number. Plus request timeout, payload
  cap (413), CORS lock, and Pydantic validation (422).

Enable locally: `cp .env.example .env`, add a `$0`-balance key, restart uvicorn.
Leave `OPENROUTER_MODEL` on the free router and you cannot spend a cent.

## Deploy for free

- **Render / Railway / Fly.io**: deploy the `Dockerfile` directly; set
  `OPENROUTER_API_KEY` in the host's environment. One container serves numerics,
  proxy, and frontend.
- **Hugging Face Spaces (Docker SDK)**: push the repo with the Dockerfile; add the
  key under Settings -> Secrets.

All run the numerics server-side, so the secure-key story holds with no separate
backend to build.

## Honest scope

The options layer is **snapshot frequency** (free data is enough for surface
fitting and static arb). SVI is fit per-slice; **SSVI** for a globally
arbitrage-free surface is the next depth upgrade. A **quoting + delta-hedge + P&L
decomposition** leg is what turns this from a *pricing* engine into a full
*market-making* one, and genuine high-frequency mechanics belong in the
underlying/hedge instrument.

## References

- Gatheral (2004), *A parsimonious arbitrage-free implied volatility parameterization* (SVI).
- Gatheral & Jacquier (2014), *Arbitrage-free SVI volatility surfaces*, Quantitative Finance.
- Gatheral, *The Volatility Surface*.  /  Sinclair, *Volatility Trading*.
- Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading*.
