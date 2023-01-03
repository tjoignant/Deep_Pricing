import os
import time
import matplotlib.pyplot as plt

from AAD_functions import *

# Down & Out Call Option Parameters
strike = 100
barrier = 90
maturity = 1

# Heston Model Parameters
S0 = torch.tensor(100.0)
v0 = torch.tensor(0.1)
rho = torch.tensor(-0.9)
kappa = torch.tensor(0.1)
theta = torch.tensor(0.5)
sigma = torch.tensor(0.1)
risk_free_rate = torch.tensor(0.02)

# Simulations Parameters
seed = 123
nb_steps = 252
nb_simuls = 1000

# -------------------------------------------------------------------------------------------------------------------- #

# Price
start = time.perf_counter()
price = MC_Pricing(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                   maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                   nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print("\nResults")
print(f" - Price: {price} ({round(end - start, 1)}s)")

# Delta
start = time.perf_counter()
delta = DeltaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                 maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                 nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Delta: {delta} ({round(end - start, 1)}s)")

# Gamma
start = time.perf_counter()
gamma = GammaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                 maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                 nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Gamma: {gamma} ({round(end - start, 1)}s)")

# Rho
start = time.perf_counter()
rho = RhoAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
             maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
             nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Rho: {rho} ({round(end - start, 1)}s)")

# Vega
start = time.perf_counter()
vega = VegaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
               maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
               nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f" - Vega: {vega} ({round(end - start, 1)}s)\n")

# LSM Dataset
start = time.perf_counter()
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                       maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                       nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
end = time.perf_counter()
print(f"LSM Dataset Generated ({round(end - start, 1)}s)")

# MC Dataset
start = time.perf_counter()
MC_prices = []
MC_deltas = []
S0_list = torch.linspace(10, 200, 30)
for S0 in S0_list:
    MC_prices.append(MC_Pricing(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(),
                                risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                                kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    MC_deltas.append(DeltaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(),
                               risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                               kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                               nb_simuls=nb_simuls, seed=seed))
end = time.perf_counter()
print(f"MC Dataset Generated ({round(end - start, 1)}s)")

# Fig 1: Pricing Function
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.scatter(X, Y, marker="+", color="grey", label='LSM samples')
ax1.plot(S0_list, MC_prices, marker="o", color="green", label='MC pricing')
ax1.set_title(f'Heston D&O {barrier} Call {strike} - Pricing Function (Pytorch)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# Fig 2: Delta Function
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.scatter(X, dYdX, marker="+", color="grey", label='LSM samples')
ax2.plot(S0_list, MC_deltas, marker="o", color="green", label='MC pricing')
ax2.set_title(f'Heston D&O {barrier} Call {strike} - Delta Function (Pytorch)')
ax2.set_xlabel('X')
ax2.set_ylabel('dYdX')
ax2.legend()

# Save Figures
if not os.path.exists('results'):
    os.makedirs('results')
fig1.savefig("results/AAD_pricing_function.png")
fig2.savefig("results/AAD_delta_function.png")

# Show Graphs
plt.show()
