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

# Greeks Variance
bump_list = np.linspace(0.05, 0.5, 30)
var_delta_list = []
var_gamma_list = []
var_rho_list = []
for bump in bump_list:
    delta_list = []
    gamma_list = []
    rho_list = []
    for j in range(10):
        delta_list.append(DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                                  maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                  nb_simuls=nb_simuls, seed=j, dS0=bump/100))
        gamma_list.append(GammaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                                  maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                  nb_simuls=nb_simuls, seed=j, dS0=bump/100))
        rho_list.append(RhoFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                              maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                              nb_simuls=nb_simuls, seed=j, dr=bump/10000))
    var_delta_list.append(np.var(delta_list))
    var_gamma_list.append(np.var(gamma_list))
    var_rho_list.append(np.var(rho_list))

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

# Fig 1: Standard Error
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.plot(nb_simuls_list, confidence_int_list)
ax1.set_xlabel('Number of Paths')
ax1.set_ylabel('Size of Interval Confidence')
ax1.grid()

# Fig 2: Delta Variance
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.plot(bump_list, var_delta_list)
ax2.set_xlabel('Bump (in %)')
ax2.set_ylabel('Variance of Delta')
ax2.grid()

# Fig 3: Gamma Variance
fig3, ax3 = plt.subplots(figsize=(15, 7.5))
ax3.plot(bump_list, var_gamma_list)
ax3.set_xlabel('Bump (in %)')
ax3.set_ylabel('Variance of Gamma')
ax3.grid()

# Fig 4: Rho Variance
fig4, ax4 = plt.subplots(figsize=(15, 7.5))
ax4.plot(bump_list/100, var_rho_list)
ax4.set_xlabel('Bump (in %)')
ax4.set_ylabel('Variance of Rho')
ax4.grid()

# Save Figures
if not os.path.exists('results'):
    os.makedirs('results')
fig1.savefig("results/FD_confidence_interval.png")
fig2.savefig("results/FD_delta_variance.png")
fig3.savefig("results/FD_gamma_variance.png")
fig4.savefig("results/FD_rho_variance.png")

# Show Graphs
plt.show()
