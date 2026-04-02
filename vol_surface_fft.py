"""
Constructs the implied volatility surface for Nifty options using the
Heston model priced via FFT. Reads option chain CSVs for multiple maturities,
calibrates Heston parameters to observed market prices, then plots the 3D
implied volatility surface derived from Heston model prices.
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from extract_option_chain import get_option_data
from heston_fft_eu import heston_fft

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STRIKES = [22400, 22600, 22800, 23000, 23200, 23400]
CURRENT_PRICE = 22679.40
CURRENT_DATE = datetime(2026, 4, 1)
RISK_FREE_RATE = 0.10
DIVIDEND_YIELD = 0.0132
OUTPUT_DIR = "./outputs"

OPTION_CHAIN_FILES = {
    "07-Apr-2026": "inputs/option-chain-ED-NIFTY-07-Apr-2026.csv",
    "13-Apr-2026": "inputs/option-chain-ED-NIFTY-13-Apr-2026.csv",
    "21-Apr-2026": "inputs/option-chain-ED-NIFTY-21-Apr-2026.csv",
    # "28-Apr-2026": "inputs/option-chain-ED-NIFTY-28-Apr-2026.csv",
}

# ---------------------------------------------------------------------------
# BSM IV inversion (to convert Heston model prices -> implied vols)
# ---------------------------------------------------------------------------

def bsm_price(F, K, T, r, sigma, option_type="call"):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bsm_vega(F, K, T, r, sigma):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T) * np.exp(-r * T)


def implied_volatility(F, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=100):
    """Newton-Raphson IV solver with bisection fallback."""
    sigma = 0.2
    for _ in range(max_iter):
        price = bsm_price(F, K, T, r, sigma, option_type)
        vega = bsm_vega(F, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega == 0:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001

    lo, hi = 1e-6, 5.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if bsm_price(F, K, T, r, mid, option_type) > market_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_option_data(files, strikes):
    return {
        maturity: get_option_data(path, strikes=strikes)
        for maturity, path in files.items()
    }


def compute_maturities(all_option_data, current_date):
    maturities, days = [], []
    for maturity_str in all_option_data:
        maturity_date = datetime.strptime(maturity_str, "%d-%b-%Y")
        d = (maturity_date - current_date).days
        days.append(d)
        maturities.append(d / 365.0)
    return np.array(maturities), np.array(days)


# ---------------------------------------------------------------------------
# Heston calibration
# ---------------------------------------------------------------------------

def calibration_objective(params, all_option_data, maturities, S0, r, q):
    """
    Mean squared relative error between Heston FFT call prices and market prices.
    params = [v0, kappa, theta, sigma_v, rho]
    """
    v0, kappa, theta, sigma_v, rho = params

    # Penalise parameter regions that are numerically explosive before evaluating
    if 2 * kappa * theta <= sigma_v ** 2:
        return 1e6   # Feller condition violated — CF will overflow

    total_error = 0.0
    count = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for T, df in zip(maturities, all_option_data.values()):
            for _, row in df.iterrows():
                K = row["Strike"]
                market_price = row["Call Price"]
                if market_price is None or np.isnan(market_price) or market_price <= 0:
                    continue
                try:
                    model_price = heston_fft(
                        S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                        option_type="call"
                    )
                    if not np.isfinite(model_price) or model_price <= 0:
                        total_error += 1e4
                        continue
                    total_error += ((model_price - market_price) / market_price) ** 2
                    count += 1
                except Exception:
                    total_error += 1e4

    return total_error / max(count, 1)


def calibrate_heston(all_option_data, maturities, S0, r, q):
    """Calibrate Heston parameters via differential evolution."""
    print("Calibrating Heston model parameters...")

    # Bounds: [v0, kappa, theta, sigma_v, rho]
    bounds = [
        (0.001, 0.5),    # v0:      initial variance
        (0.1,  10.0),    # kappa:   mean reversion speed
        (0.001, 0.5),    # theta:   long-run variance
        (0.01,  1.0),    # sigma_v: vol of vol
        (-0.99, 0.0),    # rho:     correlation (negative for equities)
    ]

    result = differential_evolution(
        calibration_objective,
        bounds,
        args=(all_option_data, maturities, S0, r, q),
        maxiter=30,
        tol=1e-3,
        seed=42,
        disp=True,
        popsize=12,
        mutation=(0.5, 1),
        recombination=0.7,
    )

    v0, kappa, theta, sigma_v, rho = result.x
    print(f"\nCalibrated Heston parameters:")
    print(f"  v0      = {v0:.6f}  (initial variance,  sqrt = {np.sqrt(v0):.4f})")
    print(f"  kappa   = {kappa:.4f}   (mean reversion speed)")
    print(f"  theta   = {theta:.6f}  (long-run variance, sqrt = {np.sqrt(theta):.4f})")
    print(f"  sigma_v = {sigma_v:.4f}   (vol of vol)")
    print(f"  rho     = {rho:.4f}   (spot-vol correlation)")
    print(f"  Feller condition satisfied: {2 * kappa * theta > sigma_v**2}")
    print(f"  Calibration MSRE: {result.fun:.6f}")
    return result.x


# ---------------------------------------------------------------------------
# Vol surface construction
# ---------------------------------------------------------------------------

def build_heston_iv_surface(params, maturities, strikes, S0, r, q):
    """
    Price every (T, K) combination with Heston FFT and convert to BSM IVs.
    Returns ndarray of shape (n_maturities, n_strikes) in percent.
    """
    v0, kappa, theta, sigma_v, rho = params
    iv_surface = np.full((len(maturities), len(strikes)), np.nan)

    for i, T in enumerate(maturities):
        F = S0 * np.exp((r - q) * T)
        for j, K in enumerate(strikes):
            try:
                heston_price = heston_fft(
                    S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                    option_type="call"
                )
                if heston_price > 0:
                    iv = implied_volatility(F, K, T, r, heston_price) * 100
                    iv_surface[i, j] = iv
            except Exception:
                pass

    return iv_surface


def fill_nan_ivs(iv_surface):
    """Interpolate NaN entries from immediate strike-axis neighbours."""
    filled = iv_surface.copy()
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            if np.isnan(filled[i, j]):
                neighbors = []
                if j > 0 and not np.isnan(filled[i, j - 1]):
                    neighbors.append(filled[i, j - 1])
                if j < filled.shape[1] - 1 and not np.isnan(filled[i, j + 1]):
                    neighbors.append(filled[i, j + 1])
                filled[i, j] = np.mean(neighbors) if neighbors else np.nan
    return filled


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_iv_surface(strikes, days_to_maturities, iv_surface, actual_ivs, S0, output_dir):
    moneyness = np.array(strikes) / S0
    X, Y = np.meshgrid(moneyness, days_to_maturities)
    Z = np.ma.masked_invalid(iv_surface)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    for i, days in enumerate(days_to_maturities):
        valid = ~np.isnan(actual_ivs[i, :])
        ax.scatter(
            moneyness[valid], [days] * int(valid.sum()), actual_ivs[i, valid],
            color="r", marker="o", zorder=5,
            label="Market IVs" if i == 0 else "",
        )

    ax.set_title("Nifty Call Option IV Surface (Heston FFT)")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Maturity")
    ax.set_zlabel("Implied Volatility (%)")
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=10)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "nifty_heston_iv_surface.png")
    plt.savefig(out_path)
    print(f"\nSurface plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    all_option_data = load_option_data(OPTION_CHAIN_FILES, STRIKES)
    maturities, days_to_maturities = compute_maturities(all_option_data, CURRENT_DATE)

    # Market IVs for scatter overlay on the plot
    actual_ivs = np.array([df["Actual IV Call"].values for df in all_option_data.values()])

    # Step 1: calibrate Heston params to observed market call prices
    params = calibrate_heston(
        all_option_data, maturities, CURRENT_PRICE, RISK_FREE_RATE, DIVIDEND_YIELD
    )

    # Step 2: build IV surface from calibrated Heston model
    iv_surface = build_heston_iv_surface(
        params, maturities, STRIKES, CURRENT_PRICE, RISK_FREE_RATE, DIVIDEND_YIELD
    )
    iv_surface = fill_nan_ivs(iv_surface)

    # Step 3: plot
    plot_iv_surface(
        STRIKES, days_to_maturities, iv_surface, actual_ivs, CURRENT_PRICE, OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
