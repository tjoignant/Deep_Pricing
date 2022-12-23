import matplotlib.pyplot as plt

from functions import *

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
# PART 1: Pricing and hedging by Monte Carlo

# Heston Diffusion
S_test, V_test = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho,
                                          kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls,
                                          seed=seed)
print(S_test)

# Payoff / Pricing
payoff = Payoff(strike=strike, barrier=barrier, S=S_test[0])
price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                   rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nDiffusion")
print(f" - Average S_T: {round(np.mean(S_test[:, -1]), 4)}")
print(f" - Payoff Simul_0: {payoff}")
print(f" - Payoff Price: {round(price, 4)}")

# FD Greeks
delta = DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
gamma = GammaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
rho = RhoFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
            rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nADD Greeks")
print(f" - Delta: {delta}")
print(f" - Gamma: {gamma}")
print(f" - Rho: {rho}")

# ADD Greeks


# -------------------------------------------------------------------------------------------------------------------- #
# PART 2: Pricing and hedging by differential deep learning

# LSM Dataset
X, Y, dYdX = LSM_dataset(strike=strike, barrier=barrier, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                         rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls,
                         seed=seed)

# MC Dataset
MC_prices = []
Delta_values = []
Gamma_values = []
Rho_values = []

for temp_S_0 in np.linspace(10, 200):
    temp_price = MC_Pricing(strike=strike, barrier=barrier, S0=temp_S_0, v0=v0, risk_free_rate=risk_free_rate,
                            maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                            nb_simuls=nb_simuls, seed=seed)
    MC_prices.append(temp_price)

# Plot Pricing Function
plt.scatter(X, Y, marker="+", color="grey", label='LSM samples')
plt.plot(np.linspace(10, 200), MC_prices, marker="o", color="green", label='MC pricing')
plt.title('Heston D&O Call Pricing Function')
plt.legend()
plt.show()
