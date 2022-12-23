import matplotlib.pyplot as plt

from functions_pytorch import *

# Down & Out Call Option Parameters
strike = 100
barrier = 90
maturity = 1

# Heston Model Parameters
S0 = torch.tensor(100.0, requires_grad=True)
v0 = torch.tensor(0.1, requires_grad=True)
rho = torch.tensor(-0.9)
kappa = torch.tensor(0.1)
theta = torch.tensor(0.5)
sigma = torch.tensor(0.1)
risk_free_rate = torch.tensor(0.02, requires_grad=True)

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
print("\nADD Greeks")
print(f" - Delta: {delta}")
print(f" - Gamma: {gamma}")
print(f" - Rho: {rho}")


# -------------------------------------------------------------------------------------------------------------------- #
# PART 2: Pricing and hedging by differential deep learning

# LSM Dataset
X, Y, dYdX = LSM_dataset(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                         maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                         sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)

# MC Dataset
MC_prices = []
FD_deltas = []
FD_gammas = []
FD_rhos = []
S0_list = torch.linspace(10, 200, 40)
for S_0 in S0_list:
    MC_prices.append(MC_Pricing(strike=strike, barrier=barrier, S0=S_0.clone(), v0=v0.clone(),
                                risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                                kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    FD_deltas.append(DeltaAAD(strike=strike, barrier=barrier, S0=S_0.clone(), v0=v0.clone(),
                                risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                                kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    FD_gammas.append(GammaAAD(strike=strike, barrier=barrier, S0=S_0.clone(), v0=v0.clone(),
                                risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                                kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    FD_rhos.append(RhoAAD(strike=strike, barrier=barrier, S0=S_0.clone(), v0=v0.clone(),
                                risk_free_rate=risk_free_rate.clone(), maturity=maturity, rho=rho.clone(),
                                kappa=kappa.clone(), theta=theta.clone(), sigma=sigma.clone(), nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))

# Fig 1: Pricing Function
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.scatter(X, Y, marker="+", color="grey", label='LSM samples')
ax1.plot(S0_list, MC_prices, marker="o", color="green", label='MC pricing')
ax1.set_title('Heston D&O Call Pricing Function')
ax1.legend()

# Fig 2: AAD Greeks
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.plot(S0_list, FD_deltas, marker="o", label='Delta')
ax2.plot(S0_list, FD_gammas, marker="o", label='Gamma')
ax2.plot(S0_list, FD_rhos, marker="o", label='Rho')
ax2.set_title('AAD Greeks')
ax2.legend()

# Show Graphs
plt.show()
