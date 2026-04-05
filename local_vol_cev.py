"""
CEV (Constant Elasticity of Variance) local volatility model.
dS = r * S * dt + sigma * S^beta * dW

Left panel  : sample asset paths
Right panel : 3-D implied volatility surface (Strike x Maturity x IV)

Interactive sliders for sigma and beta (CEV exponent in [0, 2]).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm
from scipy.optimize import brentq

# ── Parameters ───────────────────────────────────────────────────────────────
S0     = 100.0
r      = 0.05
sigma0 = 0.2
beta0  = 0.5
T_MAX  = 3.0
NSTEPS = 150
NPATHS = 3000
N_DISPLAY = 40

K_values = np.linspace(75, 130, 18)
T_values = np.linspace(0.1, T_MAX, 12)
KK, TT   = np.meshgrid(K_values, T_values)


# ── Core functions ────────────────────────────────────────────────────────────
def simulate_cev(sigma, beta):
    dt = T_MAX / NSTEPS
    S  = np.zeros((NPATHS, NSTEPS + 1))
    S[:, 0] = S0
    for i in range(1, NSTEPS + 1):
        Z      = np.random.standard_normal(NPATHS)
        S_prev = np.maximum(S[:, i - 1], 1e-8)
        dW     = sigma * S_prev ** (beta - 1) * np.sqrt(dt) * Z
        drift  = (r - 0.5 * sigma**2 * S_prev ** (2 * beta - 2)) * dt
        S[:, i] = S_prev * np.exp(drift + dW)
    return S


def bs_call(F, K, T, sigma):
    """Black-76 call price (F already discounted to forward)."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol(S0, K, T, r, price):
    """BSM IV via Brent; returns NaN on failure."""
    discount   = np.exp(-r * T)
    F          = S0 * np.exp(r * T)
    intrinsic  = max(S0 - K * discount, 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        iv = brentq(lambda s: discount * bs_call(F, K, T, s) - price,
                    1e-4, 5.0, xtol=1e-6)
        return iv
    except Exception:
        return np.nan


def compute_iv_surface(S_paths):
    """Return IV surface (%) on the T_values × K_values grid."""
    dt       = T_MAX / NSTEPS
    iv_surf  = np.full((len(T_values), len(K_values)), np.nan)
    discount = np.exp(-r * T_values)

    for i, T in enumerate(T_values):
        step = max(1, round(T / dt))
        S_T  = S_paths[:, step]
        for j, K in enumerate(K_values):
            price         = discount[i] * np.mean(np.maximum(S_T - K, 0.0))
            iv_surf[i, j] = implied_vol(S0, K, T, r, price)

    return iv_surf * 100.0  # convert to percent


# ── Initial computation ───────────────────────────────────────────────────────
t       = np.linspace(0, T_MAX, NSTEPS + 1)
S_paths = simulate_cev(sigma0, beta0)
iv_surf = compute_iv_surface(S_paths)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
plt.subplots_adjust(bottom=0.22, wspace=0.35)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

# Left: sample paths
path_lines = ax1.plot(t, S_paths[:N_DISPLAY].T, lw=0.6, alpha=0.45)
ax1.axhline(S0, color="k", lw=0.9, ls="--", label=f"S₀ = {S0}")
ax1.set_title("CEV Asset Paths")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Asset Price")
ax1.legend(fontsize=8)

# Right: 3-D IV surface
surf_container = [None]
cbar_container = [None]


def draw_surface(iv_data):
    if cbar_container[0] is not None:
        cbar_container[0].remove()
        cbar_container[0] = None
    ax2.cla()

    Z = np.ma.masked_invalid(iv_data)
    s = ax2.plot_surface(KK, TT, Z, cmap="viridis", edgecolor="none", alpha=0.88)
    surf_container[0] = s
    cbar_container[0] = fig.colorbar(s, ax=ax2, shrink=0.5, pad=0.12, label="IV (%)")

    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Maturity T (yrs)")
    ax2.set_zlabel("Implied Vol (%)")
    ax2.set_title("CEV Implied Volatility Surface")


draw_surface(iv_surf)

# ── Sliders ───────────────────────────────────────────────────────────────────
ax_sigma = plt.axes([0.15, 0.10, 0.70, 0.03])
ax_beta  = plt.axes([0.15, 0.05, 0.70, 0.03])

slider_sigma = Slider(ax_sigma, "sigma",    0.05, 1.0, valinit=sigma0, valstep=0.05)
slider_beta  = Slider(ax_beta,  "beta (CEV)", 0.0, 2.0, valinit=beta0,  valstep=0.1)


def update(_):
    sig  = slider_sigma.val
    beta = slider_beta.val

    S_new  = simulate_cev(sig, beta)
    iv_new = compute_iv_surface(S_new)

    # update paths
    for line, path in zip(path_lines, S_new[:N_DISPLAY]):
        line.set_ydata(path)
    lo = max(0.0, S_new[:N_DISPLAY].min() * 0.95)
    hi = S_new[:N_DISPLAY].max() * 1.05
    ax1.set_ylim(lo, hi)

    # update 3-D surface
    draw_surface(iv_new)
    fig.canvas.draw_idle()


slider_sigma.on_changed(update)
slider_beta.on_changed(update)

plt.suptitle("CEV Local Volatility Model  —  drag sliders to explore", y=1.01)
plt.show()
