import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from matplotlib.widgets import Slider

# -----------------------------
# 1. Brownian motion generator
# -----------------------------
def generate_brownian_paths(T, N, M, Z=None):
    dt = T / N
    t = np.linspace(0, T, N+1)
    if Z is None:
        Z = np.random.standard_normal((M, N))
    W = np.zeros((M, N+1))
    W[:, 1:] = np.cumsum(np.sqrt(dt) * Z, axis=1)
    return t, W

# -----------------------------
# 2. GBM paths
# -----------------------------
def generate_gbm_paths(S0, r, sigma, T, N, M, Z=None):
    dt = T / N
    t, W = generate_brownian_paths(T, N, M, Z)
    S = np.zeros((M, N+1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*(W[:,1:] - W[:,:-1]), axis=1))
    return t, S

# -----------------------------
# 3. CEV paths with Milstein
# -----------------------------
def generate_cev_paths_milstein(S0, r, sigma, beta, T, N, M, Z=None):
    dt = T / N
    t, W = generate_brownian_paths(T, N, M, Z)
    S = np.zeros((M, N+1))
    S[:, 0] = S0

    for i in range(N):
        dW = W[:, i+1] - W[:, i]
        S_next = (S[:, i] 
                  + r*S[:, i]*dt 
                  + sigma*S[:, i]**beta*dW 
                  + 0.5*sigma**2*beta*S[:, i]**(2*beta - 1)*(dW**2 - dt))
        S[:, i+1] = np.maximum(S_next, 0)  # ensure non-negative

    return t, S

# -----------------------------
# 4. Plot CEV vs GBM paths
# -----------------------------
def plot_cev_and_gbm(S0, r, sigma, beta, T, N, M):
    t_cev, S_cev = generate_cev_paths_milstein(S0, r, sigma, beta, T, N, M)
    t_gbm, S_gbm = generate_gbm_paths(S0, r, sigma, T, N, M)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    for i in range(M):
        plt.plot(t_cev, S_cev[i], lw=0.5)
    plt.title(f'CEV Paths (Milstein, beta={beta})')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid()

    plt.subplot(1,2,2)
    for i in range(M):
        plt.plot(t_gbm, S_gbm[i], lw=0.5)
    plt.title('GBM Paths')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid()

    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. CEV European Call pricing
# -----------------------------
def price_cev_european_call_milstein(S0, K, r, sigma, beta, T, N, M, Z=None):
    _, S = generate_cev_paths_milstein(S0, r, sigma, beta, T, N, M, Z)
    payoff = np.maximum(S[:, -1] - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

# -----------------------------
# 6. Black-Scholes IV function
# -----------------------------
def implied_volatility_cev_call(S0, K, r, T, cev_price):
    def black_scholes_call_price(S0, K, r, T, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    objective = lambda sigma: black_scholes_call_price(S0, K, r, T, sigma) - cev_price

    try:
        iv = brentq(objective, 1e-3, 5.0)
        return iv
    except Exception:
        return np.nan

# -----------------------------
# 7. Strike vs Price plot with sliders
# -----------------------------
def plot_strike_vs_price(S0, r, sigma, beta, T, N, M):
    strikes = np.linspace(0.95*S0, 1.05*S0, 20)
    Z = np.random.standard_normal((M, N))

    def compute_prices(r_, sigma_, beta_):
        _, S_cev = generate_cev_paths_milstein(S0, r_, sigma_, beta_, T, N, M, Z)
        _, S_gbm = generate_gbm_paths(S0, r_, sigma_, T, N, M, Z)
        S_T_cev = S_cev[:, -1]
        S_T_gbm = S_gbm[:, -1]
        disc = np.exp(-r_*T)
        cev_prices = [disc*np.mean(np.maximum(S_T_cev - K, 0)) for K in strikes]
        gbm_prices = [disc*np.mean(np.maximum(S_T_gbm - K, 0)) for K in strikes]
        return cev_prices, gbm_prices

    cev_prices, gbm_prices = compute_prices(r, sigma, beta)

    fig, ax = plt.subplots(figsize=(10,7))
    plt.subplots_adjust(bottom=0.3)
    line_cev, = ax.plot(strikes, cev_prices, label=f'CEV (beta={beta:.2f})', marker='o')
    line_gbm, = ax.plot(strikes, gbm_prices, label='GBM (beta=1)', marker='x')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Option Price')
    ax.set_title('European Call Price vs Strike')
    ax.grid()
    ax.legend()

    ax_beta = plt.axes([0.15, 0.18, 0.70, 0.03])
    ax_sigma = plt.axes([0.15, 0.12, 0.70, 0.03])
    ax_r = plt.axes([0.15, 0.06, 0.70, 0.03])

    slider_beta = Slider(ax_beta, 'Beta', 0.1, 2.0, valinit=beta, valstep=0.05)
    slider_sigma = Slider(ax_sigma, 'Sigma', 0.05, 0.8, valinit=sigma, valstep=0.01)
    slider_r = Slider(ax_r, 'Rate r', 0.0, 0.2, valinit=r, valstep=0.005)

    def update(val):
        b = slider_beta.val
        s = slider_sigma.val
        rr = slider_r.val
        cev_prices, gbm_prices = compute_prices(rr, s, b)
        line_cev.set_ydata(cev_prices)
        line_gbm.set_ydata(gbm_prices)
        line_cev.set_label(f'CEV (beta={b:.2f})')
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()

    slider_beta.on_changed(update)
    slider_sigma.on_changed(update)
    slider_r.on_changed(update)
    plt.show()

# -----------------------------
# 8. CEV implied volatility skew plot
# -----------------------------
def plot_cev_iv_skew(S0, r, sigma, beta, T, N, M):
    strikes = np.linspace(0.95*S0, 1.05*S0, 50)
    Z = np.random.standard_normal((M, N))

    def compute_iv_skew(beta_):
        _, S_cev = generate_cev_paths_milstein(S0, r, sigma, beta_, T, N, M, Z)
        S_T_cev = S_cev[:, -1]
        disc = np.exp(-r*T)
        ivs = []
        for K in strikes:
            cev_price = disc*np.mean(np.maximum(S_T_cev - K, 0))
            iv = implied_volatility_cev_call(S0, K, r, T, cev_price)
            ivs.append(iv)
        return ivs

    iv_skew = compute_iv_skew(beta)

    fig, ax = plt.subplots(figsize=(10,7))
    plt.subplots_adjust(bottom=0.25)
    line_iv, = ax.plot(strikes, iv_skew, label=f'CEV IV Skew (beta={beta:.2f})',lw='2',c='k')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('CEV Implied Volatility Skew')
    ax.grid()
    ax.legend()
    ax.set_ylim(0,0.5)
    ax_beta = plt.axes([0.15, 0.10, 0.70, 0.03])
    slider_beta = Slider(ax_beta, 'Beta', 0.1, 2.0, valinit=beta, valstep=0.05)

    def update(val):
        b = slider_beta.val
        ivs = compute_iv_skew(b)
        line_iv.set_ydata(ivs)
        line_iv.set_label(f'CEV IV Skew (beta={b:.2f})')
        # ax.ylim(0,0.5)
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()

    slider_beta.on_changed(update)
    plt.show()

# -----------------------------
# 9. Example usage
# -----------------------------
S0 = 22700
r = 0.1
sigma = 0.30
beta = 1.0
T = 10/365
N = 252
M = 50000  # Increase to 50000+ for smoother IV skew

# 1. Plot paths
# plot_cev_and_gbm(S0, r, sigma, beta, T, N, M)

# 2. Interactive strike vs price
# plot_strike_vs_price(S0, r, sigma, beta, T, N, M)

# 3. Interactive IV skew
plot_cev_iv_skew(S0, r, sigma, beta, T, N, M)