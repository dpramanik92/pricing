import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation of GBM paths
def simulate_gbm_paths(S0, mu, sigma, dt, Nsteps, Npaths):
    t = np.linspace(0, Nsteps * dt, Nsteps + 1)
    S = np.zeros((Npaths, Nsteps + 1))
    S[:, 0] = S0
    for i in range(1, Nsteps + 1):
        Z = np.random.standard_normal(Npaths)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return t, S

# price american option using longstaff schwartz method
def price_american_option(S, K, r, T, option_type='call'):
    dt = T / (S.shape[1] - 1)
    discount_factor = np.exp(-r * dt)
    if option_type == 'call':
        payoff = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S[:, -1], 0)

    for t in range(S.shape[1] - 2, 0, -1):
        in_the_money = payoff > 0
        if np.sum(in_the_money) == 0:
            continue
        X = S[in_the_money, t]
        Y = payoff[in_the_money] * discount_factor
        coeffs = np.polyfit(X, Y, deg=2)
        continuation_value = np.polyval(coeffs, S[in_the_money, t])
        payoff[in_the_money] = np.where(payoff[in_the_money] > continuation_value, payoff[in_the_money], continuation_value)

    return np.mean(payoff) * discount_factor

# price american option using brute force method
def price_american_option_brute_force(S, K, r, T, option_type='call'):
    dt = T / (S.shape[1] - 1)
    discount_factor = np.exp(-r * dt)
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S, 0)
    option_values = np.zeros_like(payoff)
    option_values[:, -1] = payoff[:, -1]
    for t in range(S.shape[1] - 2, -1, -1):
        option_values[:, t] = np.maximum(payoff[:, t], option_values[:, t + 1] * discount_factor)
    return np.mean(option_values[:, 0])

# price american option using PDE method
def price_american_option_pde(S0, K, r, sigma, T, Smax, Nsteps, Nprice):
    dt = T / Nsteps
    dS = Smax / Nprice
    S_grid = np.linspace(0, Smax, Nprice + 1)
    V = np.zeros((Nprice + 1, Nsteps + 1))
    if option_type == 'call':
        V[:, -1] = np.maximum(S_grid - K, 0)
    elif option_type == 'put':
        V[:, -1] = np.maximum(K - S_grid, 0)

    for t in range(Nsteps - 1, -1, -1):
        for i in range(1, Nprice):
            delta = (V[i + 1, t + 1] - V[i - 1, t + 1]) / (2 * dS)
            gamma = (V[i + 1, t + 1] - 2 * V[i, t + 1] + V[i - 1, t + 1]) / (dS ** 2)
            theta = -0.5 * sigma ** 2 * S_grid[i] ** 2 * gamma - r * S_grid[i] * delta + r * V[i, t + 1]
            V[i, t] = max(V[i, t + 1] + theta * dt, payoff(S_grid[i], K))

    return np.interp(S0, S_grid, V[:, 0])
    
    

if __name__ == "__main__":
    # Parameters
    S0 = 1000  # Initial stock price
    mu = 0.05  # Drift (expected return)
    sigma = 0.2  # Volatility
    T = 1  # Time to maturity (1 year)
    Nsteps = 1000  # Number of time steps
    Npaths = 20000  # Number of simulated paths

    dt = T / Nsteps
    t, S = simulate_gbm_paths(S0, mu, sigma, dt, Nsteps, Npaths)

    # Plot the simulated GBM paths
    plt.figure(figsize=(10, 6))
    for i in range(S.shape[0]):
        plt.plot(t, S[i], lw=0.5)
    plt.title('Simulated GBM Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.grid()
    output_dir = "./outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'simulated_gbm_paths.png'))
    mean, std = np.mean(S[:, -1]), np.std(S[:, -1])
    print(f"Mean of final stock prices: {mean:.2f}")
    print(f"Standard deviation of final stock prices: {std:.2f}")

