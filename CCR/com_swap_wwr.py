
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import math

# Define matplotlib parameters for better visualization
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = False

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'y'

plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=plt.get_cmap('tab20b').colors
)


# Function to compute the swap value at time s 
def compute_swap_value(S,F_t,P_t,F,s):
    day_s = math.ceil(s*20)
    idx  = math.ceil(s)
    S_idx = S[day_s,:]
    if s <12:
        swap_value = np.sum((np.outer(S_idx,F_t[idx:]/F_t[0]) - F)*P_t[idx:]/P_t[0],axis=1)
    else:
        swap_value = np.zeros(S_idx.shape)
    return swap_value

# Read the data from the file about the commodity swap
data = pd.read_excel("../inputs/commodity_swap.xlsx", sheet_name="Sheet1")
data['Time'] = data['Time']/360

# Function to generate simulation data for the commodity price and the credit factor,
# using Cholesky decomposition to introduce correlation between the two processes

def generate_simulation_data(T, N, M, rho, sigma, sigma2, r, X_init, data):
    F = np.dot(data['F(0, Ti)'], data['P(0,Ti)']) / np.sum(data['P(0,Ti)'])
    print(f"The fair fixed price of the swap is: {F:.4f}")
    P_t = np.array(data['P(0,Ti)'])
    F_t = np.array(data['F(0, Ti)'])
    S0 = F_t[0]
    # Generate two correlated Brownian motions using Cholesky decomposition
    dt = T / N
    t_grid = np.linspace(0, T, N + 1)

    S = np.zeros((N + 1, M))  # N rows, M columns
    X = np.zeros((N + 1, M))
    X[0, :] = X_init
    X1 = np.random.randn(N, M)
    X2 = np.random.randn(N, M)

    Z1 = X1
    Z2 = rho * X1 + np.sqrt(1 - rho**2) * X2

    for i in range(N):
        S[i + 1] = S[i] + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z1[i]
    S = S0 * np.exp(S)

    for i in range(N):
        X[i + 1] = X[i] + (r - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * Z2[i]

    return F, P_t, F_t, S, X


# Function to calculate the CVA series for the commodity swap with WWR, using the simulated data

def calculate_cva_series(T, N, M, rho, sigma, sigma2, r, R, X_init, data):
    # Simulate data
    F, P_t, F_t, S, X = generate_simulation_data(T, N, M, rho, sigma, sigma2, r,X_init, data)

    # Calculate CVA series
    default_event = (X < 0)
    indicator = (np.cumsum(default_event, axis=0) == 0).astype(int)
    S = S * indicator

    t_grid = [t for t in range(0, 13)]
    cva_corr = []
    for s in t_grid:
        swap_values = compute_swap_value(S, F_t, P_t, F, s)
        val = (1 - R) * np.mean(np.maximum(swap_values, 0)) * P_t[s]
        cva_corr.append(val)

    return t_grid, cva_corr

# Sample set of parameters for the commodity swap and the simulation
T = 1.0
N = 240
M = 50000
rho = 0.0
sigma = 0.2
sigma2 = 0.15
r = 0.02
R = 0.4
X_init = 0.1


# Example usage
t_grid, cva_corr = calculate_cva_series(T, N, M, 0.1, sigma, sigma2, r,R, X_init, data)
t_grid, cva_corr2 = calculate_cva_series(T, N, M, 0.3, sigma, sigma2, r,R, X_init, data)
t_grid, cva_corr3 = calculate_cva_series(T, N, M, 0.5, sigma, sigma2, r,R, X_init, data)
t_grid, cva_corr4 = calculate_cva_series(T, N, M, 0.8, sigma, sigma2, r,R, X_init, data)

cva_0_tot = np.sum(cva_corr)
cva_01_tot = np.sum(cva_corr2)
cva_03_tot = np.sum(cva_corr3)
cva_05_tot = np.sum(cva_corr4)

rho = [0.1, 0.3, 0.5, 0.8]
cva_tot = [cva_0_tot, cva_01_tot, cva_03_tot, cva_05_tot]
    
# Final Plotting of the CVA series for different correlation values and the total CVA vs correlation    
fig,ax = plt.subplots(1,2,figsize=(20,7))
fig.suptitle('CVA of the Commodity Swap with WWR',fontsize=30)
ax[0].plot(t_grid,cva_corr,label='CVA, corr = 0.1')
ax[0].plot(t_grid,cva_corr2,label='CVA, corr = 0.3')
ax[0].plot(t_grid,cva_corr3,label='CVA, corr = 0.5')
ax[0].plot(t_grid,cva_corr4,label='CVA, corr = 0.8')

ax[0].set_title('CVA of the Swap with WWR')
ax[0].set_xlabel('Time (Months)')
ax[0].set_ylabel('CVA')
# ax.grid()
ax[0].legend()

ax[1].plot(rho,cva_tot,label='Total CVA',color='k',marker='x',ls='--')
ax[1].set_title('Total CVA vs Correlation')
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Total CVA')
plt.subplots_adjust(hspace=0.5,wspace=0.3)

# ax.grid()
ax[1].legend()  
plt.tight_layout()
plt.show()







