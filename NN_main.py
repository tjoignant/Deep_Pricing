import time
import torch
import matplotlib.pyplot as plt

from NN_functions import Twin_Network, training
from AAD_functions import HestonLSM, normalize_data

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

# FD Step
dS0 = pow(10, -4)

# -------------------------------------------------------------------------------------------------------------------- #

# Generate Normalized Training Set
start = time.perf_counter()
X, Y, dYdX = HestonLSM(strike=strike, barrier=barrier, v0=v0.clone(), risk_free_rate=risk_free_rate.clone(),
                       maturity=maturity, rho=rho.clone(), kappa=kappa.clone(), theta=theta.clone(),
                       sigma=sigma.clone(), nb_steps=nb_steps, nb_simuls=nb_simuls, seed=seed)
X_mean, X_std, X_norm, Y_mean, Y_std, Y_norm, dYdX_mean, dYdX_std, dYdX_norm, lambda_j = normalize_data(X, Y, dYdX)
end = time.perf_counter()
print(f"\nNormalized Training Set Generated ({round(end - start, 1)}s)")

# Train Classic Neural Network
start = time.perf_counter()
classic_nn = Twin_Network(nb_inputs=1, nb_hidden_layer=nb_hidden_layers, nb_neurones=nb_neurones, seed=seed)
classic_nn = training(model=classic_nn, X_norm=X_norm, Y_norm=Y_norm, nb_epochs=nb_epochs)
end = time.perf_counter()
print(f"Classic NN Trained ({round(end - start, 1)}s)")

# Train Differential Neural Network
start = time.perf_counter()
differential_nn = Twin_Network(nb_inputs=1, nb_hidden_layer=nb_hidden_layers, nb_neurones=nb_neurones, seed=seed)
differential_nn = training(model=differential_nn, X_norm=X_norm, Y_norm=Y_norm, nb_epochs=nb_epochs,
                           dYdX_norm=dYdX_norm, lambda_j=lambda_j)
end = time.perf_counter()
print(f"Differential NN Trained ({round(end - start, 1)}s)")

# Price Classic Neural Network
start = time.perf_counter()
classic_nn_price = classic_nn.predict_price(S0, X_mean, X_std, Y_mean, Y_std)
classic_nn_price_up = classic_nn.predict_price(S0+dS0, X_mean, X_std, Y_mean, Y_std)
classic_nn_price_down = classic_nn.predict_price(S0-dS0, X_mean, X_std, Y_mean, Y_std)
classic_nn_delta_FD = (classic_nn_price_up - classic_nn_price_down) / (2*dS0)
end = time.perf_counter()
print(f"Classic NN Pricing ({round(end - start, 1)}s)")

# Price Differentiated Neural Network
start = time.perf_counter()
differential_nn_price, differential_nn_delta_NN = \
    differential_nn.predict_price_and_diffs(S0, X_mean, X_std, Y_mean, Y_std, dYdX_mean, dYdX_std)
differential_nn_price_up, _ = \
    differential_nn.predict_price_and_diffs(S0+dS0, X_mean, X_std, Y_mean, Y_std, dYdX_mean, dYdX_std)
differential_nn_price_down, _ = \
    differential_nn.predict_price_and_diffs(S0-dS0, X_mean, X_std, Y_mean, Y_std, dYdX_mean, dYdX_std)
differential_nn_delta_FD = (differential_nn_price_up - differential_nn_price_down) / (2*dS0)
end = time.perf_counter()
print(f"Differential NN Pricing ({round(end - start, 1)}s)")

# Plot NN Training Cost Evolution
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
ax1.plot(range(1, nb_epochs+1), classic_nn.cost_values, label="Classic")
ax1.plot(range(1, nb_epochs+1), differential_nn.cost_values, label="Differential")
ax1.set_xlabel("Nb of Epochs")
ax1.set_ylabel("Cost Value")
ax1.legend()
ax1.grid()
fig1.savefig("results/NN_training_cost.png")

# Display Results
print("\nResults:")
print(f" - Classic NN Price: {classic_nn_price}")
print(f" - Classic NN FD Delta: {classic_nn_delta_FD}")
print(f" - Classic NN Final Cost: {classic_nn.cost_values[-1]}")
print(f" - Differential NN Price: {differential_nn_price}")
print(f" - Differential NN Delta: {differential_nn_delta_NN}")
print(f" - Differential NN FD Delta: {differential_nn_delta_FD}")
print(f" - Differential NN Final Cost: {differential_nn.cost_values[-1]}")

# Display Graphs
plt.show()
