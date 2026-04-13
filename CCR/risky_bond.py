import numpy as np
import matplotlib.pyplot as plt

# define matplotlib parameters for better visualization
# plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def compute_zero_coupon_bond_price(face_value, T, r):
    return face_value * np.exp(-r * T)

def compute_risky_bond_price(face_value, T, r, default_prob, recovery_rate):
    zero_coupon_price = compute_zero_coupon_bond_price(face_value, T, r)
    risky_bond_price = zero_coupon_price - np.exp(-r * T) * default_prob * (1 - recovery_rate) * face_value
    return risky_bond_price


# Parameters
face_value = 1000  # Face value of the bond
T = 5  # Time to maturity in years
r = 0.05  # Risk-free interest rate
#default_prob = 0.2  # Probability of default
recovery_rate = 0.4  # Recovery rate in case of default 

# Compute sensitivity of risky bond price to changes in the default probability
default_prob_values = np.arange(0.001, 1,0.01)  # Varying default probability from 0.001 to 0.99
risky_bond_prices = [compute_risky_bond_price(face_value, T, r, default_prob, recovery_rate) for default_prob in default_prob_values]

def plot_risky_bond_price_sensitivity(default_prob_values, risky_bond_prices):
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(default_prob_values, risky_bond_prices, label='Risky Bond Price', color='blue')
    plt.title('Sensitivity of Risky Bond Price to Default Probability')
    plt.xlabel('Default Probability')
    plt.ylabel('Risky Bond Price')
    plt.grid()
    plt.legend()
    plt.show()

# Compute the credit spread (difference between risky bond price and risk-free bond price)
def compute_credit_spread(discount_rate, survival_prob, recovery_rate,T,t):
    return -1/(T-t)*np.log(survival_prob * (1-recovery_rate) + recovery_rate)

# Define a function to plot the credit spread as a function of the survival probability
def plot_credit_spread(discount_rate, recovery_rate, T, t):
    survival_prob_values = np.linspace(0.001, 1, 100)  # Varying survival probability from 0.001 to 1
    credit_spreads = [compute_credit_spread(discount_rate, survival_prob, recovery_rate,T,t)*100 for survival_prob in survival_prob_values]

    plt.figure(figsize=(10, 6))
    plt.plot(survival_prob_values, credit_spreads, label='Credit Spread', color='red',ls='--')
    plt.title('Credit Spread as a Function of Survival Probability')
    plt.xlabel('Survival Probability')
    plt.ylabel('Credit Spread (bps)')
    plt.grid()
    plt.legend()
    plt.show()

# Call the function
# plot_risky_bond_price_sensitivity(default_prob_values, risky_bond_prices)

plot_credit_spread(r, recovery_rate, T, 0)
