import torch
import matplotlib.pyplot as plt

from nn_class import Twin_Network, training
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

# Neural Network Parameters
nb_hidden_layers = 4
nb_neurones = 20
nb_epochs = 100

# Training Set
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                       maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                       sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)

# Normalized Training Set
X_mean, X_std, X_norm, Y_mean, Y_std, Y_norm, dYdX_norm, lambda_j = normalize_data(X, Y, dYdX)

# Classic Neural Network
classic_nn = Twin_Network(nb_inputs=1, nb_hidden_layer=nb_hidden_layers, nb_neurones=nb_neurones, seed=seed)
classic_nn = training(model=classic_nn, X_norm=X_norm, Y_norm=Y_norm, nb_epochs=nb_epochs)

# Differential Neural Network

# Plot Cost Values
plt.plot(range(1, nb_epochs+1), classic_nn.cost_values, label="Classic")
plt.xlabel("Nb of epochs")
plt.ylabel("Cost value")
plt.title("Training Results")
plt.legend()
plt.grid()

# Show Graphs
plt.show()
