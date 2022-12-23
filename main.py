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
S_test = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho, kappa=kappa,
                                  theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print(S_test)

# Payoff / Pricing
payoff = Payoff(strike=strike, barrier=barrier, S=S_test[0])
price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                   rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nDiffusion")
print(f" - Average S_T: {np.mean(S_test[:, -1])}")
print(f" - Payoff Simul_0: {payoff}")
print(f" - Payoff Price: {price}")

# FD Greeks
delta = DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
gamma = GammaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
rho = RhoFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
            rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
print("\nFD Greeks")
print(f" - Delta: {delta}")
print(f" - Gamma: {gamma}")
print(f" - Rho: {rho}")


# -------------------------------------------------------------------------------------------------------------------- #
# PART 2: Pricing and hedging by differential deep learning

# LSM Dataset
X, Y, dYdX = LSM_dataset(strike=strike, barrier=barrier, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                         rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls,
                         seed=seed)

# MC Dataset
MC_prices = []
FD_deltas = []
FD_gammas = []
FD_rhos = []
S0_list = np.linspace(10, 200, 40)
for S_0 in S0_list:
    MC_prices.append(MC_Pricing(strike=strike, barrier=barrier, S0=S_0, v0=v0, risk_free_rate=risk_free_rate,
                                maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                nb_simuls=nb_simuls, seed=seed))
    FD_deltas.append(DeltaFD(strike=strike, barrier=barrier, S0=S_0, v0=v0, risk_free_rate=risk_free_rate,
                             maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                             nb_simuls=nb_simuls, seed=seed))
    FD_gammas.append(GammaFD(strike=strike, barrier=barrier, S0=S_0, v0=v0, risk_free_rate=risk_free_rate,
                             maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                             nb_simuls=nb_simuls, seed=seed))
    FD_rhos.append(RhoFD(strike=strike, barrier=barrier, S0=S_0, v0=v0, risk_free_rate=risk_free_rate,
                         maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                         nb_simuls=nb_simuls, seed=seed))

#Interval Confidence
confidenceinterval = []
for simulations in range(1,1000):
    S_test = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho, kappa=kappa,
                                  theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=simulations, seed=seed)
    print(simulations)
    payoffs = []
    for S in S_test:
        payoffs.append(Payoff(strike=strike, barrier=barrier, S=S))
    up,down = StandardError(simulations,payoffs)
    confidenceinterval.append(up-down)

# Fig 1: Pricing Function
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.scatter(X, Y, marker="+", color="grey", label='LSM samples')
ax1.plot(S0_list, MC_prices, marker="o", color="green", label='MC pricing')
ax1.set_title('Heston D&O Call Pricing Function')
ax1.legend()

# Fig 2: FD Greeks
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.plot(S0_list, FD_deltas, marker="o", label='Delta')
ax2.plot(S0_list, FD_gammas, marker="o", label='Gamma')
ax2.plot(S0_list, FD_rhos, marker="o", label='Rho')
ax2.set_title('FD Greeks')
ax2.legend()

# Fig 3: Confidence Interval
fig3, ax3 = plt.subplots(figsize=(15, 7.5))
ax3.plot(np.linspace(1,1000,999),confidenceinterval,marker="o",label= "Taille des intervalles")
ax3.set_title('Confidence Interval')
ax3.legend()
# Show Graphs
plt.show()
