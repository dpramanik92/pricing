"""
cev_iv_smile.py
---------------
Interactive IV smile plot for the CEV model using the closed-form
Schroder (1989) non-central chi-squared formula — no Monte Carlo.

Model:   dS = r S dt  +  σ_CEV · S^β · dW
         where  σ_CEV = σ_atm · S₀^{1-β}
         so that the ATM implied vol ≈ σ_atm for all β.

Formula (Schroder 1989, β < 1):
    C = S₀ · [1 − Φ(2κ_K ; df+2, 2κ_S)]  −  K e^{-rT} · Φ(2κ_S ; df, 2κ_K)

    df    = 2 / (1 − β)
    κ_S   = e^{2r(1-β)²T}  /  [σ_atm² (1-β)² T_eff]
    κ_K   = (K/S₀)^{2(1-β)}  /  [σ_atm² (1-β)² T_eff]
    T_eff = (e^{2r(1-β)T} − 1) / (2r(1-β))   [= T when r = 0]

Sliders: β, σ_atm (ATM vol), r (risk-free rate), T (days to expiry)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2, norm
from scipy.optimize import brentq
from matplotlib.widgets import Slider

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────────────────────────────

def bs_call(S0, K, r, sigma, T):
    """Standard Black-Scholes European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(S0, K, r, T, price):
    """BS implied vol via Brent's method."""
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S0, K, r, s, T) - price,
                      1e-6, 20.0, xtol=1e-7)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# CEV analytical call price  (Schroder 1989)
# ─────────────────────────────────────────────────────────────────────────────

def cev_call(S0, K, r, sigma_atm, beta, T):
    """
    European call price under the CEV model (β ∈ (0, 1]).

    sigma_atm : approximate ATM Black-Scholes vol.
                Internally  σ_CEV = sigma_atm · S₀^{1-β},
                so the ATM IV is ≈ sigma_atm regardless of β.
    """
    if K <= 0 or T <= 0 or sigma_atm <= 0:
        return float(max(S0 - K * np.exp(-r * T), 0.0))

    # β = 1  →  plain Black-Scholes (GBM)
    if abs(beta - 1.0) < 1e-4:
        return bs_call(S0, K, r, sigma_atm, T)

    nu = 1.0 - beta          # ν = 1 − β  > 0

    # ── Effective time (accounts for risk-free drift) ─────────────────────────
    if abs(r) < 1e-10:
        T_eff      = T
        exp_factor = 1.0
    else:
        rnu       = 2.0 * r * nu
        T_eff      = (np.exp(rnu * T) - 1.0) / rnu
        exp_factor = np.exp(2.0 * r * nu**2 * T)   # e^{2r(1-β)²T}

    # ── Non-centrality parameters ─────────────────────────────────────────────
    # Using ATM-vol normalisation: σ_CEV = sigma_atm · S₀^ν → factors of S₀ cancel.
    denom   = sigma_atm**2 * nu**2 * T_eff

    kappa_S = exp_factor / denom                   # κ_S
    kappa_K = (K / S0) ** (2.0 * nu) / denom       # κ_K  (moneyness-based)

    df = 2.0 / nu     # degrees of freedom  = 2/(1-β)

    try:
        A = 1.0 - ncx2.cdf(2.0 * kappa_K, df + 2.0, 2.0 * kappa_S)
        B =       ncx2.cdf(2.0 * kappa_S, df,        2.0 * kappa_K)
    except Exception:
        return float(max(S0 - K * np.exp(-r * T), 0.0))

    return float(max(S0 * A - K * np.exp(-r * T) * B, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# IV smile computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_iv_smile(S0, r, sigma_atm, beta, T, moneyness_grid):
    """
    Returns Black-Scholes IV (as fraction) for each m = K/S₀ in moneyness_grid.
    """
    ivs = np.full(len(moneyness_grid), np.nan)
    for i, m in enumerate(moneyness_grid):
        K     = m * S0
        price = cev_call(S0, K, r, sigma_atm, beta, T)
        ivs[i] = bs_implied_vol(S0, K, r, T, price)
    return ivs


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters  (NIFTY-like)
# ─────────────────────────────────────────────────────────────────────────────
S0      = 22700.0
r0      = 0.07
sig0    = 0.18      # ATM implied vol (18%)
beta0   = 0.5
T_days0 = 30.0

moneyness = np.linspace(0.87, 1.13, 80)    # K / S₀  range

# Reference curves (fixed β values drawn as dashed lines)
BETAS_REF  = [0.2,       0.4,       0.6,       0.8,       1.0]
COLORS_REF = ["#9467bd", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

# ─────────────────────────────────────────────────────────────────────────────
# Figure & axes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.33)

# ── Reference curves (dashed, thin) ──────────────────────────────────────────
for b, c in zip(BETAS_REF, COLORS_REF):
    ivs_ref = compute_iv_smile(S0, r0, sig0, b, T_days0 / 365.0, moneyness)
    ax.plot(moneyness * 100, ivs_ref * 100,
            "--", color=c, lw=1.3, alpha=0.55, label=f"β = {b:.1f}")

# ── Active curve (moves with sliders) ────────────────────────────────────────
ivs0  = compute_iv_smile(S0, r0, sig0, beta0, T_days0 / 365.0, moneyness)
(line,) = ax.plot(moneyness * 100, ivs0 * 100,
                  "k-", lw=2.8, zorder=5, label=f"β = {beta0:.2f}  (active)")

ax.axvline(100, color="gray", ls=":", lw=1.2, alpha=0.5, label="ATM")

ax.set_xlabel("Moneyness   K / S₀  (%)", fontsize=12)
ax.set_ylabel("Implied Volatility (%)", fontsize=12)
ax.set_title(
    f"CEV Model — Implied Volatility Smile\n"
    f"S₀ = {S0:,.0f}    [Schroder 1989 closed-form]",
    fontsize=13,
)
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_xlim(moneyness[0] * 100, moneyness[-1] * 100)
ax.set_ylim(0, 50)

# Annotation box showing active parameters
param_text = ax.text(
    0.02, 0.97, "", transform=ax.transAxes,
    fontsize=9, va="top", family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

def _fmt_params(b, sig, rr, T_d):
    return (f"β = {b:.2f}   σ_atm = {sig:.1%}   "
            f"r = {rr:.1%}   T = {T_d:.0f}d")

param_text.set_text(_fmt_params(beta0, sig0, r0, T_days0))

# ─────────────────────────────────────────────────────────────────────────────
# Sliders
# ─────────────────────────────────────────────────────────────────────────────
ax_beta  = plt.axes([0.12, 0.22, 0.78, 0.03])
ax_sigma = plt.axes([0.12, 0.16, 0.78, 0.03])
ax_r     = plt.axes([0.12, 0.10, 0.36, 0.03])
ax_T     = plt.axes([0.54, 0.10, 0.36, 0.03])

s_beta  = Slider(ax_beta,  "β  (Beta)",    0.05, 0.99, valinit=beta0,   valstep=0.01,
                 color="#aec7e8")
s_sigma = Slider(ax_sigma, "σ  (ATM Vol)", 0.05, 0.60, valinit=sig0,    valstep=0.005,
                 color="#ffbb78")
s_r     = Slider(ax_r,     "r  (Rate)",    0.00, 0.20, valinit=r0,      valstep=0.005,
                 color="#98df8a")
s_T     = Slider(ax_T,     "T  (Days)",    5,    365,  valinit=T_days0, valstep=5,
                 color="#c5b0d5")


def update(_):
    b   = s_beta.val
    sig = s_sigma.val
    rr  = s_r.val
    T   = s_T.val / 365.0

    ivs_new = compute_iv_smile(S0, rr, sig, b, T, moneyness)
    line.set_ydata(ivs_new * 100)
    line.set_label(f"β = {b:.2f}  (active)")

    param_text.set_text(_fmt_params(b, sig, rr, s_T.val))

    ax.set_ylim(0, 50)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    fig.canvas.draw_idle()


s_beta.on_changed(update)
s_sigma.on_changed(update)
s_r.on_changed(update)
s_T.on_changed(update)

plt.show()
