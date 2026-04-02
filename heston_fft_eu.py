import numpy as np

def heston_fft(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type='call', N=4096, alpha=1.5):
    """
    Price a European call or put option using the Heston model and FFT.

    Parameters:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        v0: Initial variance
        kappa: Mean reversion rate
        theta: Long-term variance
        sigma: Volatility of volatility
        rho: Correlation between stock and variance
        option_type: 'call' or 'put'
        N: Number of FFT points
        alpha: Damping factor for call/put transform

    Returns:
        Option price
    """
    # Characteristic function of log(S_T) under risk-neutral measure
    def heston_cf(u):
        a = kappa * theta
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b**2 + sigma**2 * (u**2 + 1j * u))
        g = (b - d) / (b + d)
        C = (r - q) * 1j * u * T + (a / sigma**2) * ((b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((b - d) / sigma**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        return np.exp(C + D * v0 + 1j * u * np.log(S0))

    # Carr-Madan FFT parameters
    eta = 0.25
    lambda_ = 2 * np.pi / (N * eta)
    b_bound = N * lambda_ / 2  # log-strike grid spans [-b_bound, b_bound)

    j = np.arange(N)
    v = j * eta  # integration variable grid

    # Analytical Fourier transform of the dampened call payoff
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    psi = np.exp(-r * T) * heston_cf(v - (alpha + 1) * 1j) / denom

    # FFT with trapezoidal weights (halve the first term)
    weights = np.ones(N)
    weights[0] = 0.5
    x = weights * np.exp(1j * b_bound * v) * psi * eta

    fft_result = np.fft.fft(x)

    # Log-strike grid
    k_grid = -b_bound + lambda_ * j  # log-strikes
    call_prices = (np.exp(-alpha * k_grid) / np.pi) * fft_result.real

    # Find the index closest to log(K)
    log_K = np.log(K)
    idx = int(np.round((log_K + b_bound) / lambda_))
    idx = np.clip(idx, 0, N - 1)
    call_price = call_prices[idx]

    if option_type == 'call':
        return call_price
    elif option_type == 'put':
        # Use put-call parity
        return call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

# Example usage
S0 = 22679
K = 23000
T = 9/365
r = 0.1
q = 0.0132
v0 = 0.04
kappa = 2
theta = 0.04
sigma = 0.25
rho = -0.7

call_price = heston_fft(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type='call')
put_price = heston_fft(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type='put')

print(f"Call Price: {call_price}")
print(f"Put Price: {put_price}")

