# Pricing of a T year semi-annual coupon bond with a face value of $1000, 
# a coupon rate of 6%, and a risk-free interest rate given by the term structure%. 
# The bond has a default probability of 10% and a recovery rate of 40%. 
# We will compute the price of the risky bond and the credit spread.

import numpy as np
import matplotlib.pyplot as plt

# define matplotlib parameters for better visualization
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

#-------------------------------------------------------------------
#  DEFINE THE BOND PARAMETERS
#-------------------------------------------------------------------
FACE_VALUE = 1.000  # Face value of the bond
COUPON_RATE = 0.06  # Annual coupon rate
RISK_FREE_RATE = [0.03,0.035]  # Risk-free interest rate
TIME_TO_MATURITY = 1  # Time to maturity in years
CUMPON_FREQUENCY = 2  # Semi-annual coupons
SURVIVAL_PROBABILITY = [0.98,0.94]  # Probability of default
RECOVERY_RATE = 0.4  # Recovery rate in case of default

#-------------------------------------------------------------------
#  CALCULATE THE PRICE OF THE RISKY BOND
#-------------------------------------------------------------------
def compute_the_risky_coupon_bond_price(face_value, coupon_rate, risk_free_rate, time_to_maturity, coupon_frequency, survival_probability, recovery_rate):
    N_payments = int(time_to_maturity * coupon_frequency)
    coupon_payment = face_value * coupon_rate / coupon_frequency

    # Calculate the present value of the coupon payments and the face value
    value = 0
    survival_probability0 = 1

    for i in range(N_payments):
        default_probability = survival_probability0 - survival_probability[i]
        discount_factor = np.exp(-risk_free_rate[i] *(i+1) / coupon_frequency)
        # case1 survival
        pv = discount_factor * coupon_payment*survival_probability[i]
        # print(pv)
        # case2 default
        pv += discount_factor * (default_probability * recovery_rate * face_value)
        # print(pv)
        survival_probability0 = survival_probability[i]
        value += pv
    pv = discount_factor * face_value * survival_probability[-1]
    # print(pv)
    value = value + pv
    return value
    
    
    # Calculate the price of the risky bond
risky_bond_price = compute_the_risky_coupon_bond_price(FACE_VALUE, COUPON_RATE, RISK_FREE_RATE, TIME_TO_MATURITY, CUMPON_FREQUENCY, SURVIVAL_PROBABILITY, RECOVERY_RATE)
print(f"The price of the risky bond is: ${risky_bond_price:.6f}")
