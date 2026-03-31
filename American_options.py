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

def price_american_options(S,K,r,T,option_type='call'):
    dt = T / S.shape[1]
    discount_factor = np.exp(-r * dt)
    option_values = np.zeros_like(S)

    
    

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

