import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# Simulation of GBM paths
def simulate_gbm_paths(S0, mu, sigma,dt,Nsteps, Npaths):
    dt = T / Nsteps
    t = np.linspace(0, T, Nsteps + 1)
    S = np.zeros((Npaths, Nsteps + 1))
    S[:, 0] = S0
    for i in range(1, Nsteps + 1):
        Z = np.random.standard_normal(Npaths)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return t, S

def plot_gbm_paths(t, S):
    plt.figure(figsize=(10, 6))
    for i in range(S.shape[0]):
        plt.plot(t, S[i], lw=0.5)
    plt.title('Simulated GBM Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()

def plot_single_gbm_path(t, S):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, lw=2)
    plt.title('Single Simulated GBM Path')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()

# analytical black scholes formula for european option
def black_scholes_option(S0, K,T, r, sigma,option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return option_price



# # Plot the simulated GBM paths
# plot_gbm_paths(t, S)
# # Plot a single GBM path
# plot_single_gbm_path(t, S[0])
# # Show randomly selected paths
# plt.figure(figsize=(10, 6))
# for i in range(10):
#     plt.plot(t, S[i], lw=0.5)
# plt.title('Randomly Selected GBM Paths')
# plt.xlabel('Time (years)')
# plt.ylabel('Stock Price')
# plt.grid()
# plt.show()  

def compute_gbm_statistics(S):
    mean_path = np.mean(S, axis=0)
    std_path = np.std(S, axis=0)
    return mean_path, std_path

def price_european_option(S, K, r, T,option_type='call'):
    if option_type == 'call':
        payoff = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S[:, -1], 0)

    discounted_payoff = np.exp(-r * T) * payoff
    option_price = np.mean(discounted_payoff)
    return option_price

# plot strike vs option price
def plot_strike_vs_option_price(S, K_values, r, T):
    option_prices = []
    for K in K_values:
        option_price = price_european_call_option(S, K, r, T)
        option_prices.append(option_price)
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, option_prices,c='k')
    plt.title('Strike Price vs European Call Option Price')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Option Price')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Parameters for option pricing
    K = 1000  # Strike price
    r = 0.05  # Risk-free rate
    # Parameters
    S0 = 900  # Initial stock price
    mu = 0.05  # Drift (expected return)
    sigma = 0.2  # Volatility
    T = 1 # Time to maturity (1 years)
    Nsteps = 200  # Number of time steps
    Npaths = 20000  # Number of simulated paths
    K_values = np.arange(400, 2000, 50)  # Range of strike prices for plotting


    # Simulate GBM paths
    t, S = simulate_gbm_paths(S0, r, sigma, T, Nsteps, Npaths)

    date1 = datetime.datetime(2024, 1, 1)


    # Price the European call option using Monte Carlo simulation
    # plot_strike_vs_option_price(S, K_values, r, T)

    option_price = price_european_option(S, K, r, T, option_type='call')

    print(f"Estimated European Call Option Price: {option_price:.2f}")
    print(f"Analytical European Call Option Price: {black_scholes_option(S0, K, T, r, sigma, option_type='call'):.2f}")