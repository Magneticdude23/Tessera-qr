/* Controller: wires the controls to the API, renders tiles/charts/tables,
   tracks the selected maturity, and drives the desk-note request. */
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { source: "synthetic", data: null, activeT: null };

  // ---- live UTC session clock ----
  function tick() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    $("clock").textContent =
      `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`;
  }
  tick(); setInterval(tick, 1000);

  // ---- backend health dot ----
  API.health().then((h) => {
    const dot = $("health-dot");
    if (h && h.status === "ok") dot.classList.add("up");
    else dot.classList.add("down");
  });

  // ---- source toggle ----
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.source = btn.dataset.source;
      const live = state.source === "live";
      $("ticker-field").hidden = !live;
      $("seed-field").hidden = live;
      $("live-note").hidden = !live;
    });
  });

  // ---- fit ----
  $("fit-btn").addEventListener("click", fit);
  async function fit() {
    setLoading(true);
    const body = {
      source: state.source,
      ticker: $("ticker").value || "SPY",
      r: parseFloat($("r").value),
      q: parseFloat($("q").value),
      seed: parseInt($("seed").value || "7", 10),
    };
    const { ok, data } = await API.fitSurface(body);
    setLoading(false);
    if (!ok || !data || data.error) {
      flashError(data && data.error ? data.error : "surface request failed");
      return;
    }
    state.data = data;
    state.activeT = data.slices.length ? data.slices[0].T : null;
    render();
  }

  function setLoading(on) {
    $("fit-btn").disabled = on;
    $("fit-btn").querySelector(".btn-label").textContent = on ? "fitting…" : "fit surface";
    $("fit-btn").querySelector(".btn-spin").hidden = !on;
    $("main").classList.toggle("busy", on);
  }

  // ---- render everything ----
  function render() {
    const d = state.data, m = d.metrics;
    $("t-underlying").textContent = m.underlying;
    $("t-mats").textContent = m.n_maturities;
    $("t-spot").textContent = m.spot;

    const calOk = d.calendar_free;
    setTile($("t-calendar"), calOk ? "free" : "VIOLATION", calOk);
    setPill($("status-calendar"), "calendar " + (calOk ? "ok" : "arb"), calOk);

    const bfOk = d.slices.every((s) => s.butterfly_free);
    setTile($("t-butterfly"), bfOk ? "free" : "VIOLATION", bfOk);
    setPill($("status-butterfly"), "butterfly " + (bfOk ? "ok" : "arb"), bfOk);

    Charts.surface3d($("surface3d"), d.surface3d);
    renderTabs();
    renderSlice();
    renderParity();
    renderBox();

    // brief flash on the readout cells to signal a fresh fit
    document.querySelectorAll(".ro").forEach((t) => {
      t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash");
    });

    $("raw-metrics").textContent = JSON.stringify(m, null, 2);
    $("note-btn").disabled = false;
    $("note-out").textContent = "";
    $("note-out").className = "note-out";
  }

  function setTile(el, text, ok) {
    el.textContent = text;
    el.className = "ro-v " + (ok ? "ok" : "bad");
  }
  function setPill(el, text, ok) {
    el.textContent = text;
    el.className = "pill " + (ok ? "pill-ok" : "pill-bad");
  }

  // ---- maturity tabs ----
  function renderTabs() {
    const wrap = $("mat-tabs");
    wrap.innerHTML = "";
    state.data.slices.forEach((s) => {
      const b = document.createElement("button");
      b.className = "mat-tab" + (s.T === state.activeT ? " active" : "");
      b.textContent = s.label;
      b.addEventListener("click", () => { state.activeT = s.T; renderTabs(); renderSlice(); });
      wrap.appendChild(b);
    });
  }

  function renderSlice() {
    const s = state.data.slices.find((x) => x.T === state.activeT);
    if (!s) return;
    Charts.smile($("smile"), s);
    Charts.density($("density"), s);
    const tw = $("tripwire-status");
    $("slice-panel").classList.toggle("arb", !s.butterfly_free);
    if (s.butterfly_free) {
      tw.textContent = "no butterfly arb · min g(k) = " + s.min_g.toFixed(3);
      tw.className = "pill pill-ok";
    } else {
      tw.textContent = "ARBITRAGE · min g(k) = " + s.min_g.toFixed(3);
      tw.className = "pill pill-bad";
    }
  }

  // ---- scan tables ----
  function renderParity() {
    const rows = state.data.parity;
    const el = $("parity");
    if (!rows || !rows.length) {
      el.innerHTML = '<p class="muted">No dislocations beyond threshold.</p>';
      return;
    }
    let html = "<table><thead><tr><th>T</th><th>strike</th><th>gap (bps)</th></tr></thead><tbody>";
    rows.forEach((r) => {
      const flag = Math.abs(r.gap_bps) >= 5 ? " class='flag'" : "";
      html += `<tr><td>${r.T.toFixed(2)}y</td><td>${r.K.toFixed(2)}</td><td${flag}>${r.gap_bps.toFixed(2)}</td></tr>`;
    });
    html += "</tbody></table><p class='hint'>Gross, pre-cost. Most close once spread, borrow and dividends net out.</p>";
    el.innerHTML = html;
  }

  function renderBox() {
    const box = state.data.box;
    const el = $("box");
    if (!box) { el.innerHTML = '<p class="muted">Needs two-sided quotes (synthetic mode supplies them).</p>'; return; }
    const r = parseFloat($("r").value) * 100;
    el.innerHTML =
      `<div class="bigrate">${(box.median_implied_rate * 100).toFixed(2)}%</div>` +
      `<p class="hint">median box-implied financing across ${box.n_maturities} maturities. ` +
      `Assumed r = ${r.toFixed(2)}%. A wide spread of box rates flags a mispriced strike pair.</p>`;
  }

  // ---- desk note ----
  $("note-btn").addEventListener("click", async () => {
    if (!state.data) return;
    const out = $("note-out");
    out.className = "note-out info";
    out.textContent = "drafting…";
    $("note-btn").disabled = true;
    const { data } = await API.deskNote(state.data.metrics);
    $("note-btn").disabled = false;
    if (data && data.ok) { out.className = "note-out"; out.textContent = data.text; return; }
    const reason = data ? data.reason : "error";
    if (reason === "no_api_key") {
      out.className = "note-out info";
      out.textContent = "Commentary disabled: no OPENROUTER_API_KEY configured on the server. Add it to the backend environment to enable.";
    } else if (reason === "rate_limited") {
      out.className = "note-out info"; out.textContent = "Rate limit reached — try again in a minute.";
    } else if (reason === "daily_limit") {
      out.className = "note-out info"; out.textContent = "Daily free-tier limit reached — resets tomorrow. (This cap keeps the app free.)";
    } else if (reason === "paid_model_blocked") {
      out.className = "note-out info"; out.textContent = "Blocked: a non-free model was configured while free-only mode is on. The app will not call a paid model.";
    } else {
      out.className = "note-out err"; out.textContent = "Commentary unavailable (" + reason + ").";
    }
  });

  function flashError(msg) {
    const out = $("note-out");
    out.className = "note-out err";
    out.textContent = msg;
  }

  // fit once on load so the console isn't empty
  fit();
})();
