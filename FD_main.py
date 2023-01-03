import os
import time
import matplotlib.pyplot as plt

from FD_functions import *

# Down & Out Call Option Parameters
strike = 100
barrier = 90
maturity = 1

# Heston Model Parameters
S0 = 100
v0 = 0.1
rho = -0.9
kappa = 0.1
theta = 0.5
sigma = 0.1
risk_free_rate = 0.02

# Simulations Parameters
seed = 123
nb_steps = 252
nb_simuls = 1000

# -------------------------------------------------------------------------------------------------------------------- #

# Price
start = time.perf_counter()
price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                   rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print("\nResults")
print(f" - Price: {price} ({round(end - start, 1)}s)")

# Delta
start = time.perf_counter()
delta = DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Delta: {delta} ({round(end - start, 1)}s)")

# Gamma
start = time.perf_counter()
gamma = GammaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Gamma: {gamma} ({round(end - start, 1)}s)")

# Rho
start = time.perf_counter()
rho = RhoFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
            rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Rho: {rho} ({round(end - start, 1)}s)")

# Vega
start = time.perf_counter()
vega = VegaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
            rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Vega: {vega} ({round(end - start, 1)}s)\n")

# Standard Error
start = time.perf_counter()
confidence_int_list = []
nb_simuls_list = range(1, 250)
for my_nb_simuls in nb_simuls_list:
    Y_list = []
    for j in range(1, 10):
        Y_list.append(MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                                 maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                 nb_simuls=my_nb_simuls, seed=j))
    std = StandardError(Y=Y_list, nb_simuls=my_nb_simuls)
    confidence_int_list.append(std*1.96)
end = time.perf_counter()
print(f"Standard Error Evolution Computed ({round(end - start, 1)}s)")

# LSM Dataset
start = time.perf_counter()
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                       rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls,
                       seed=seed)
end = time.perf_counter()
print(f"LSM Dataset Generated ({round(end - start, 1)}s)")

# MC Dataset
start = time.perf_counter()
MC_prices = []
MC_deltas = []
S0_list = np.linspace(10, 200, 30)
for S0 in S0_list:
    MC_prices.append(MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                                maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    MC_deltas.append(DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                             maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                             nb_simuls=nb_simuls, seed=seed))
end = time.perf_counter()
print(f"MC Dataset Generated ({round(end - start, 1)}s)")

# Fig 1: Pricing Function
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.scatter(X, Y, marker="+", color="grey", label='LSM samples')
ax1.plot(S0_list, MC_prices, marker="o", color="green", label='MC pricing')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# Fig 2: Delta Function
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.scatter(X, dYdX, marker="+", color="grey", label='LSM samples')
ax2.plot(S0_list, MC_deltas, marker="o", color="green", label='MC pricing')
ax2.set_xlabel('X')
ax2.set_ylabel('dYdX')
ax2.legend()

# Fig 3: Standard Error
fig3, ax3 = plt.subplots(figsize=(15, 7.5))
ax3.plot(nb_simuls_list, confidence_int_list)
ax3.set_xlabel('Number of paths')
ax3.set_ylabel('Size of Interval Confidence')
ax3.grid()

# Save Figures
if not os.path.exists('results'):
    os.makedirs('results')
fig1.savefig("results/FD_pricing_function.png")
fig2.savefig("results/FD_delta_function.png")
fig3.savefig("results/FD_confidence_interval.png")

# Show Graphs
plt.show()
