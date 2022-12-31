import torch

from nn_functions import Twin_Network, training
from deep_learning_functions import HestonLSM, normalize_data

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

# Training Set
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                       maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                       sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)