"""
Constructs the implied volatility surface for Nifty options using the
Black-Scholes-Merton model. Reads option chain CSVs for multiple maturities,
computes BSM implied volatility for each strike, and plots the 3-D surface.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for projection='3d'
from scipy.stats import norm

from extract_option_chain import get_option_data

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
    "28-Apr-2026": "inputs/option-chain-ED-NIFTY-28-Apr-2026.csv",
}

# ---------------------------------------------------------------------------
# BSM pricing functions
# ---------------------------------------------------------------------------

def black_scholes_option_with_dividend(F, K, T, r, sigma, option_type="call"):
    """Black-Scholes price using forward price F = S * exp((r - q) * T)."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def option_vega(F, K, T, r, sigma):
    """BSM vega (sensitivity to sigma)."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T) * np.exp(-r * T)


def implied_volatility(F, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=100):
    """
    Compute implied volatility via Newton-Raphson, falling back to bisection
    if Newton-Raphson does not converge.
    """
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_option_with_dividend(F, K, T, r, sigma, option_type)
        vega = option_vega(F, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega == 0:
            break
        sigma -= diff / vega

    # Bisection fallback
    lo, hi = 1e-6, 5.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if black_scholes_option_with_dividend(F, K, T, r, mid, option_type) > market_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_option_data(files: dict, strikes: list) -> dict:
    """Return {maturity_str: DataFrame} for each file in *files*."""
    return {
        maturity: get_option_data(path, strikes=strikes)
        for maturity, path in files.items()
    }


def compute_maturities(all_option_data: dict, current_date: datetime):
    """Return (maturities_in_years, days_to_maturities) aligned with all_option_data keys."""
    maturities, days = [], []
    for maturity_str in all_option_data:
        maturity_date = datetime.strptime(maturity_str, "%d-%b-%Y")
        d = (maturity_date - current_date).days
        days.append(d)
        maturities.append(d / 365.0)
    return np.array(maturities), np.array(days)


# ---------------------------------------------------------------------------
# IV computation
# ---------------------------------------------------------------------------

def compute_iv_grid(all_option_data, maturities, current_price, risk_free_rate, dividend_yield):
    """
    Returns (strike_prices, call_prices, actual_ivs, implied_vols) as 2-D arrays
    with shape (n_maturities, n_strikes).
    """
    strike_prices = np.array([df["Strike"].values for df in all_option_data.values()])
    call_prices = np.array([df["Call Price"].values for df in all_option_data.values()])
    actual_ivs = np.array([df["Actual IV Call"].values for df in all_option_data.values()])

    forward_prices = current_price * np.exp((risk_free_rate - dividend_yield) * maturities)

    implied_vols = np.zeros_like(call_prices)
    for i, (T, F) in enumerate(zip(maturities, forward_prices)):
        implied_vols[i, :] = [
            implied_volatility(F, K, T, risk_free_rate, mp, option_type="call") * 100
            if mp is not None and not np.isnan(mp) and mp > 0
            else np.nan
            for K, mp in zip(strike_prices[i], call_prices[i])
        ]

    return strike_prices, call_prices, actual_ivs, implied_vols


def fill_nan_ivs(implied_vols: np.ndarray) -> np.ndarray:
    """Replace NaN entries with the mean of their valid strike-axis neighbours (in-place)."""
    filled = implied_vols.copy()
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

def plot_iv_surface(
    strike_prices, days_to_maturities, implied_vols, actual_ivs,
    current_price, output_dir
):
    """Plot and save the 3-D implied volatility surface."""
    moneyness = strike_prices[0] / current_price
    X, Y = np.meshgrid(moneyness, days_to_maturities)
    Z = np.ma.masked_invalid(implied_vols)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    for i, days in enumerate(days_to_maturities):
        valid = ~np.isnan(actual_ivs[i, :])
        moneyness_i = strike_prices[i] / current_price
        ax.scatter(
            moneyness_i[valid], [days] * valid.sum(), actual_ivs[i, valid],
            color="r", marker="o", zorder=5,
            label="Actual IVs" if i == 0 else "",
        )

    ax.set_title("Nifty Call Option Implied Volatility Surface")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Maturity")
    ax.set_zlabel("Implied Volatility (%)")
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=10)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "nifty_call_iv_surface.png"))
    # plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    all_option_data = load_option_data(OPTION_CHAIN_FILES, STRIKES)
    maturities, days_to_maturities = compute_maturities(all_option_data, CURRENT_DATE)

    strike_prices, _, actual_ivs, implied_vols = compute_iv_grid(
        all_option_data, maturities, CURRENT_PRICE, RISK_FREE_RATE, DIVIDEND_YIELD
    )

    smoothed_ivs = fill_nan_ivs(implied_vols)

    plot_iv_surface(
        strike_prices, days_to_maturities, smoothed_ivs, actual_ivs,
        CURRENT_PRICE, OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
