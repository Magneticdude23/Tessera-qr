/* Plotly rendering, themed to the terminal palette. All curves arrive
   pre-computed from the backend; this module only draws them. */
const Charts = (() => {
  const C = {
    bg: "rgba(0,0,0,0)", ink: "#EAE7E0", dim: "#9C988E", faint: "#615E57",
    grid: "#1E1E25", amber: "#E3A838", teal: "#57B6A6", red: "#DC5B4D",
    mono: "Plex Mono, monospace",
  };
  const BASE = {
    paper_bgcolor: C.bg, plot_bgcolor: C.bg,
    font: { family: C.mono, size: 11, color: C.dim },
    margin: { l: 46, r: 14, t: 8, b: 36 },
    showlegend: false,
    xaxis: { gridcolor: C.grid, zerolinecolor: C.faint, linecolor: C.grid,
             tickfont: { size: 10 } },
    yaxis: { gridcolor: C.grid, zerolinecolor: C.faint, linecolor: C.grid,
             tickfont: { size: 10 } },
  };
  const CFG = { displayModeBar: false, responsive: true };

  function surface3d(el, s3) {
    const data = [{
      type: "surface", x: s3.k, y: s3.T, z: s3.z,
      colorscale: [[0, "#1A1A14"], [0.5, C.amber], [1, "#F4D98A"]],
      showscale: false, opacity: 0.96,
      contours: { z: { show: true, usecolormap: true, width: 1,
                       project: { z: false } } },
    }];
    const layout = Object.assign({}, BASE, {
      margin: { l: 0, r: 0, t: 6, b: 0 },
      scene: {
        bgcolor: C.bg,
        xaxis: { title: "k", color: C.dim, gridcolor: C.grid,
                 backgroundcolor: C.bg, showbackground: true,
                 zerolinecolor: C.faint },
        yaxis: { title: "T", color: C.dim, gridcolor: C.grid,
                 backgroundcolor: C.bg, showbackground: true },
        zaxis: { title: "IV %", color: C.dim, gridcolor: C.grid,
                 backgroundcolor: C.bg, showbackground: true },
        camera: { eye: { x: 1.5, y: -1.5, z: 0.85 } },
      },
    });
    Plotly.react(el, data, layout, CFG);
  }

  function smile(el, slice) {
    const data = [
      { x: slice.fit_k, y: slice.fit_vol, mode: "lines", name: "SVI fit",
        line: { color: C.amber, width: 2.5 } },
      { x: slice.market_k, y: slice.market_iv, mode: "markers", name: "market",
        marker: { color: C.ink, size: 6, line: { color: C.bg, width: 1 } } },
    ];
    const layout = Object.assign({}, BASE, {
      xaxis: Object.assign({}, BASE.xaxis, { title: { text: "log-moneyness k", font: { size: 10 } } }),
      yaxis: Object.assign({}, BASE.yaxis, { title: { text: "IV %", font: { size: 10 } } }),
    });
    Plotly.react(el, data, layout, CFG);
  }

  function density(el, slice) {
    const free = slice.butterfly_free;
    const line = free ? C.teal : C.red;
    // shade the negative-density region (the arbitrage) in red
    const negX = [], negY = [];
    slice.fit_k.forEach((k, i) => {
      const d = slice.density[i];
      negX.push(k); negY.push(d < 0 ? d : 0);
    });
    const data = [
      { x: slice.fit_k, y: negY, mode: "lines", fill: "tozeroy",
        line: { width: 0 }, fillcolor: "rgba(220,91,77,.28)",
        hoverinfo: "skip" },
      { x: slice.fit_k, y: slice.density, mode: "lines",
        line: { color: line, width: 2.5 }, name: "density" },
      { x: slice.fit_k, y: slice.g, mode: "lines", yaxis: "y2",
        line: { color: C.faint, width: 1, dash: "dot" }, name: "g(k)" },
    ];
    const layout = Object.assign({}, BASE, {
      shapes: [{ type: "line", x0: Math.min(...slice.fit_k), x1: Math.max(...slice.fit_k),
                 y0: 0, y1: 0, line: { color: C.faint, width: 1, dash: "dash" } }],
      xaxis: Object.assign({}, BASE.xaxis, { title: { text: "log-moneyness k", font: { size: 10 } } }),
      yaxis: Object.assign({}, BASE.yaxis, { title: { text: "RN density", font: { size: 10 } } }),
      yaxis2: { overlaying: "y", side: "right", showgrid: false,
                tickfont: { size: 9, color: C.faint }, title: { text: "g(k)", font: { size: 9, color: C.faint } } },
    });
    Plotly.react(el, data, layout, CFG);
  }

  return { surface3d, smile, density };
})();
