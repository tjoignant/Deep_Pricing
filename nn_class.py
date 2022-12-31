import torch
import torch.nn as nn
import torch.optim as optim


class Twin_Network(nn.Module):
    def __init__(self, nb_inputs, nb_hidden_layer, nb_neurones, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.nb_inputs = nb_inputs
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_neurones = nb_neurones
        self.layers = nn.ModuleList()
        self.cost_values = []
        self._init_layers()

    def _init_layers(self):
        for i in range(0, self.nb_hidden_layer + 1):
            # If First Layer
            if i == 0:
                layer = nn.Linear(self.nb_inputs, self.nb_neurones)
            # If Output Layer
            elif i == self.nb_hidden_layer:
                layer = nn.Linear(self.nb_neurones, 1)
            # If Hidden Layer
            else:
                layer = nn.Linear(self.nb_neurones, self.nb_neurones)
            # He Initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            # If Output Layer
            if layer == self.layers[-1]:
                # No Activation Function
                x = layer(x)
            # Else
            else:
                # Relu Activation Function
                x = nn.ReLU(layer(x))
        return x

    def predict_price(self, X, X_mean, X_std, Y_mean, Y_std):
        # Forward Propagation
        X_norm = (X - X_mean) / X_std
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        return Y

    def predict_price_and_diffs(self, X, X_mean, X_std, Y_mean, Y_std, dYdX_mean, dYdX_std):
        # Forward Propagation
        X_norm = (X - X_mean) / X_std
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        # Backward Propagation
        torch.autograd.grad(Y, X, retain_graph=True)
        Y.backward()
        dYdX_norm = X.grad
        dYdX = dYdX_norm * dYdX_std + dYdX_mean
        return Y, dYdX


def training(model, X_norm, Y_norm, nb_epochs, dYdX_norm=None, lambda_j=None):
    # Variables Initialization
    loss = None
    if dYdX_norm:
        alpha = 1 / (1 + model.nb_inputs)
    # Cost Function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=0.1)
    # Optimization Loop
    for _ in range(0, nb_epochs):
        optimizer.zero_grad()
        loss = criterion(model.forward(X_norm), Y_norm)
        loss.backward()
        optimizer.step()
    # Store Cost Value
    model.cost_values.append(loss.item())