/* Thin client over the FastAPI backend. No secrets here — the browser only ever
   talks to our own /api/* endpoints, never to OpenRouter directly. */
const API = (() => {
  async function postJSON(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (_) {}
    return { ok: res.ok, status: res.status, data };
  }

  return {
    health: () => fetch("/api/health").then(r => r.json()).catch(() => null),
    fitSurface: (params) => postJSON("/api/surface", params),
    deskNote: (metrics) => postJSON("/api/desk-note", { metrics }),
  };
})();
