import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Read the data from the file
yields = pd.read_excel("inputs/sample_yield_curve.xlsx", sheet_name="Sheet1")
print("The current yield curve data is:")
print(yields)
print("=="*50)
yields['P(0,T)'] = np.exp(-yields['Yield'] * yields['Time']/100)
print(yields)
print("=="*50)

yields['log_P(0,T)'] = np.log(yields['P(0,T)'])

# Define a linear interpolation function  and extrapolation function
def interpolate(T, variable='Yield',method='cubic'):
    if method == 'linear':
        return np.interp(T, yields['Time'], yields[variable])
    elif method == 'cubic':
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(yields['Time'], yields[variable], extrapolate=True)
        return cs(T)
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubic'.")
    
T = np.arange(0.1, 30.0, 0.1)  # Time to maturity from 0.1 to 30 years in steps of 0.1
interpolated_yields = [interpolate(t, variable='Yield') for t in T]
interpolated_discount_factors = [interpolate(t, variable='P(0,T)') for t in T]

def compute_instantaneous_forward_rate(T,method='cubic'):
    log_P_T = interpolate(T, variable='log_P(0,T)',method=method)
    log_P_T_plus_del = interpolate(T + 0.1, variable='log_P(0,T)',method=method)
    forward_rate = -1/0.1 * (log_P_T_plus_del - log_P_T)
    return forward_rate

    
# Plot the yield curve and the discount factors along with interpolated values
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(yields['Time'], yields['Yield'], 'o', label='Data Points')
# plt.plot(T, interpolated_yields, label='Interpolated Yield Curve')
# plt.title('Yield Curve')
# plt.xlabel('Time to Maturity (Years)')
# plt.ylabel('Yield (%)')
# plt.grid()
# plt.legend()    

# plt.subplot(1, 2, 2)
# plt.plot(yields['Time'], yields['P(0,T)'], 'o', label='Data Points')
# plt.plot(T, interpolated_discount_factors, label='Interpolated Discount Factors')
# plt.title('Discount Factors')
# plt.xlabel('Time to Maturity (Years)')
# plt.ylabel('P(0,T)')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()

###################################################################################
# The calculation of Forward Rates
###################################################################################
def compute_forward_rate(t,yields):
    P_T = interpolate(t, variable='P(0,T)')
    P_T_plus_del = interpolate(t + 0.1, variable='P(0,T)')
    forward_rate = -1/0.1 * np.log(P_T_plus_del / P_T)
    return forward_rate

forward_rates = [compute_instantaneous_forward_rate(t)*100 for t in T]

# Plot the forward rates
# plt.figure(figsize=(10, 6))
# plt.plot(T, forward_rates, label='Forward Rates', color='green')
# plt.plot(T, interpolated_yields, label='Interpolated Yield Curve')

# plt.title('Forward Rates')
# plt.xlabel('Time to Maturity (Years)')
# plt.ylabel('Forward Rate (%)')
# plt.grid()
# plt.legend()
# plt.show()

# Compute the theta parameter of the Hull-White model
def hull_white_theta(T, a, sigma):
    forward_rate = compute_instantaneous_forward_rate(T)
    forward_rate_1 = compute_instantaneous_forward_rate(T+0.1)

    forward_rate_derivative = (forward_rate_1 - forward_rate) / 0.1

    
    theta = a*forward_rate + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * T)) + forward_rate_derivative
    return theta

# Example parameters for the Hull-White model
a = 0.1  # Mean reversion speed
sigma = 0.01  # Volatility
theta_values = [hull_white_theta(t, a, sigma) for t in T]
# Plot the theta parameter
# plt.figure(figsize=(10, 6))
# plt.plot(T, theta_values, label='Theta Parameter', color='purple')
# plt.title('Theta Parameter of the Hull-White Model')
# plt.xlabel('Time to Maturity (Years)')
# plt.ylabel('Theta')
# plt.grid()
# plt.legend()
# plt.show()
# print ("Theta values vs T for the Hull-White model:")
# for t, theta in zip(T, theta_values):
#     print(f"{t:.1f}, {theta:.6f}")

###### Calibrate the Hull-White model parameters to fit the Swaption Volatility Surface

# Compare interpolation methods for instantaneous forward rates and yields
T = np.arange(0.1, 30.0, 0.1)

# Compute forward rates and yields for both linear and cubic interpolation
linear_forward_rates = [compute_instantaneous_forward_rate(t,'linear')*100 for t in T]
cubic_forward_rates = [compute_instantaneous_forward_rate(t)*100 for t in T]

linear_yields = [interpolate(t, variable='Yield', method='linear') for t in T]
cubic_yields = [interpolate(t, variable='Yield', method='cubic') for t in T]

# Plot the results
plt.figure(figsize=(10, 6))

# Plot yields, linear interpolated forward rates, and cubic spline forward rates
plt.plot(T, linear_yields, label='Yields (Linear Interpolation)', color='blue')
plt.plot(T, linear_forward_rates, label='Forward Rates (Linear Interpolation)', color='green')
plt.plot(T, cubic_forward_rates, label='Forward Rates (Cubic Spline Interpolation)', color='orange')

plt.title('Yields and Forward Rates Interpolation')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Rate (%)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
