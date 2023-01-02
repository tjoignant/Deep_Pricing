import os
import matplotlib.pyplot as plt

from deep_learning_functions import *

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
# PART 1: Pricing and hedging by Monte Carlo

# Heston Diffusion
S_test = GeneratePathsHestonEuler(S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                                  maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                                  sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print(S_test)

# Payoff / Pricing
payoff = Payoff(strike=strike, barrier=barrier, S=S_test[0])
price = MC_Pricing(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                   maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                   nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nDiffusion")
print(f" - Average S_T: {torch.mean(S_test[:, -1])}")
print(f" - Payoff Simul_0: {payoff}")
print(f" - Payoff Price: {price}")

# AAD Greeks
delta = DeltaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                 maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
                 nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
gamma = GammaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                 maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), 
                 nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
rho = RhoAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(), 
             maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), 
             nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
vega = VegaAAD(strike=strike, barrier=barrier, S0=S0.clone(), v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
               maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(),
               nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nADD Greeks")
print(f" - Delta: {delta}")
print(f" - Gamma: {gamma}")
print(f" - Rho: {rho}")
print(f" - Vega: {vega}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# PART 2: Pricing and hedging by differential deep learning

# LSM Dataset
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                       maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                       sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)

# MC Dataset
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
fig1.savefig("results/pricing_function_pytorch.png")
fig2.savefig("results/delta_function_pytorch.png")

# Show Graphs
plt.show()
