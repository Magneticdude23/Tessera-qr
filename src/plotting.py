"""
Plotting. Three figures that together tell the whole story of a slice fit:

1. smile      -- market IV points vs the fitted SVI curve (the fit quality)
2. butterfly  -- g(k) across log-moneyness, with the zero line (the arb check)
3. density    -- implied risk-neutral density, which goes negative exactly where
                 g(k) does (the economic meaning of a butterfly violation)

Kept dependency-light: matplotlib only, no seaborn, no styling magic.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .svi import (SVIParams, svi_total_variance_to_vol, butterfly_g,
                  risk_neutral_density)


def plot_smile(ax, k_mkt, iv_mkt, p: SVIParams, T, title=""):
    ax.scatter(k_mkt, iv_mkt * 100, s=22, zorder=3, label="market IV", color="#1b3a5b")
    k_dense = np.linspace(k_mkt.min() - 0.1, k_mkt.max() + 0.1, 400)
    iv_fit = svi_total_variance_to_vol(k_dense, p, T) * 100
    ax.plot(k_dense, iv_fit, lw=2, color="#c0392b", label="SVI fit")
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)


def plot_butterfly(ax, p: SVIParams, k_range=(-1.5, 1.5)):
    k = np.linspace(*k_range, 600)
    g = butterfly_g(k, p)
    ax.plot(k, g, lw=2, color="#2c3e50")
    ax.axhline(0.0, color="#c0392b", lw=1.2, ls="--")
    ax.fill_between(k, g, 0, where=(g < 0), color="#c0392b", alpha=0.25,
                    label="butterfly arbitrage")
    ax.set_xlabel("log-moneyness  k")
    ax.set_ylabel("g(k)")
    ax.set_title("Butterfly check:  g(k) ≥ 0 ⇔ no butterfly arb")
    if (g < 0).any():
        ax.legend(frameon=False, fontsize=9)


def plot_density(ax, p: SVIParams, k_range=(-1.5, 1.5)):
    k = np.linspace(*k_range, 600)
    dens = risk_neutral_density(k, p)
    ax.plot(k, dens, lw=2, color="#16a085")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.fill_between(k, dens, 0, where=(dens < 0), color="#c0392b", alpha=0.3,
                    label="negative density")
    ax.set_xlabel("log-moneyness  k")
    ax.set_ylabel("implied RN density")
    ax.set_title("Risk-neutral density")
    if (dens < 0).any():
        ax.legend(frameon=False, fontsize=9)


def slice_figure(k_mkt, iv_mkt, p: SVIParams, T, suptitle="", path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    plot_smile(axes[0], k_mkt, iv_mkt, p, T, title="Smile & SVI fit")
    plot_butterfly(axes[1], p)
    plot_density(axes[2], p)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
