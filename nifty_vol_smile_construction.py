# The code reads the option data from a CSV file, extracts the relevant columns, and then plots the implied volatility smile for a specific maturity. The plot shows how implied volatility varies with strike price, which is a common way to visualize the volatility smile in options markets.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import norm
import os
from extract_option_chain import get_option_data
# Load the option data from the NSE option chain CSV
option_data = get_option_data(
    'inputs/option-chain-ED-NIFTY-13-Apr-2026.csv',
    strikes=[22000, 22400, 22800, 23200, 23400, 23800, 24000, 24200]  
)
print(option_data)
# df['Strike Price'] = df['Strike Price'].astype(float)
# df['Call Price'] = df['Call Price'].astype(float)
# df['Put Price'] = df['Put Price'].astype(float)

# Define Black-Scholes analytical formula with dividend yield and forward price
def black_scholes_option_with_dividend(F, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes formula using forward price and considering dividend yield.
    
    Parameters:
    F : float : Forward price of the underlying asset
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset
    option_type : str : 'call' or 'put'
    
    Returns:
    float : Option price
    """
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        option_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return option_price

# define a function to compute the vega of the option
def option_vega(F, K, T, r, sigma):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    vega = F * norm.pdf(d1) * np.sqrt(T) * np.exp(-r * T)
    return vega

# deffine a function to compute the implied volatility using Newton-Raphson method
# or fall back to bisection method if the newton-raphson method fails to converge
def implied_volatility(F, K, T, r, market_price, option_type='call', tol=1e-6, max_iter=100):
    sigma = 0.2  # Initial guess for volatility
    for i in range(max_iter):
        price = black_scholes_option_with_dividend(F, K, T, r, sigma, option_type)
        vega = option_vega(F, K, T, r, sigma)
        price_diff = price - market_price
        
        if abs(price_diff) < tol:
            return sigma
        
        if vega == 0:  # Avoid division by zero
            break
        
        sigma -= price_diff / vega
    
    # If Newton-Raphson fails to converge, use bisection method
    lower_bound = 1e-6
    upper_bound = 5.0
    while upper_bound - lower_bound > tol:
        sigma = (lower_bound + upper_bound) / 2
        price = black_scholes_option_with_dividend(F, K, T, r, sigma, option_type)
        
        if price > market_price:
            upper_bound = sigma
        else:
            lower_bound = sigma
            
    return (lower_bound + upper_bound) / 2

test_strike = 22750
bid = 425.25
ask = 427.80

test_price = 446.10 #(bid + ask) / 2
spot_index = 22689
test_rate = 0.1
test_yield = 0.0132
current_date = datetime(2026, 4, 1)
maturity_date = datetime(2026, 4, 13)
days_to_maturity = (maturity_date - current_date).days
T = days_to_maturity / 365.0
F = spot_index * np.exp((test_rate - test_yield) * T)
test_iv = implied_volatility(F, test_strike, T, test_rate, test_price, option_type='call')
print(f"Implied Volatility for strike {test_strike}: {test_iv:.4f}")

# now compute the implied volatility for all the strikes and plot the smile
strikes = option_data['Strike'].values
call_prices = option_data['Call Price'].values
actual_ivs = option_data['Actual IV Call'].values
implied_vols = []
for K, market_price in zip(strikes, call_prices):
    iv = implied_volatility(F, K, T, test_rate, market_price, option_type='call')*100
    implied_vols.append(iv) 
plt.figure(figsize=(10, 6))
plt.plot(strikes, implied_vols, marker='o', linestyle='-', color='k',lw=2,label='Volatility Smile')
plt.plot(strikes, actual_ivs, marker='x', linestyle='--', color='r', lw=2, label='Actual IV')
plt.title('Nifty Option Implied Volatility Smile')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.ylim(0,np.max(implied_vols)*1.5)
plt.grid()
plt.legend()
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'nifty_option_iv_smile.png'))
# plt.show()
plt.close()

print("Implied Volatility Smile plotted and saved successfully.")
# Print the implied volatilities for each strike vs actual implied volatilities
print("Strike Price | Market Price | Actual IV | Implied IV")
for K, market_price, actual_iv, implied_iv in zip(strikes, call_prices, actual_ivs, implied_vols):
    print(f"{K:12} | {market_price:12.2f} | {actual_iv:9.4f} | {implied_iv:10.4f}")

#==========================================================================
# PUT Calculation
#===========================================================================



# now compute the implied volatility for puts for all the strikes and plot the smile
strikes = option_data['Strike'].values
put_prices = option_data['Put Price'].values
actual_ivs_put = option_data['Actual IV Put'].values
implied_vols_put = []
for K, market_price in zip(strikes, put_prices):
    iv = implied_volatility(F, K, T, test_rate, market_price, option_type='put')*100
    implied_vols_put.append(iv) 
plt.figure(figsize=(10, 6))
plt.plot(strikes, implied_vols_put, marker='o', linestyle='-', color='k',lw=2,label='Put Volatility Smile')
plt.plot(strikes, actual_ivs_put, marker='x', linestyle='--', color='r', lw=2, label='Actual IV Put')
plt.title('Nifty Put Option Implied Volatility Smile')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.ylim(0,np.max(implied_vols_put)*1.5)
plt.grid()
plt.legend()
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'nifty_put_option_iv_smile.png'))
# plt.show()
plt.close()

print("Put Option Implied Volatility Smile plotted and saved successfully.")
# Print the implied volatilities for each strike vs actual implied volatilities for puts
print("Strike Price | Market Price | Actual IV Put | Implied IV Put")
for K, market_price, actual_iv, implied_iv in zip(strikes, put_prices, actual_ivs_put, implied_vols_put):
    print(f"{K:12} | {market_price:12.2f} | {actual_iv:14.4f} | {implied_iv:15.4f}")

