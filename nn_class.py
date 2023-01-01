import torch
import torch.nn as nn
import torch.optim as optim


class Twin_Network(nn.Module):
    def __init__(self, nb_inputs, nb_hidden_layer, nb_neurones, seed=1):
        super(Twin_Network, self).__init__()
        torch.manual_seed(seed)
        self.nb_inputs = nb_inputs
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_neurones = nb_neurones
        self.layers = nn.ModuleList()
        self.cost_values = []
        self._init_layers()

    def _init_layers(self):
        # For Each Layer
        for i in range(0, self.nb_hidden_layer + 1):
            # If First Hidden Layer
            if i == 0:
                layer = nn.Linear(self.nb_inputs, self.nb_neurones)
            # If Output Layer
            elif i == self.nb_hidden_layer:
                layer = nn.Linear(self.nb_neurones, 1)
            # Else
            else:
                layer = nn.Linear(self.nb_neurones, self.nb_neurones)
            # He Initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')
            self.layers.append(layer)

    def forward(self, x):
        # Set Activation Function
        activation = nn.ReLU()
        # For Each Layer
        for layer in self.layers:
            # If Output Layer
            if layer == self.layers[-1]:
                # Forward
                x = layer(x)
            # Else
            else:
                # Forward + Activation
                x = activation(layer(x))
        return x

    def predict_price(self, X, X_mean, X_std, Y_mean, Y_std):
        # Forward Propagation
        X_norm = torch.div(torch.tensor([X]) - X_mean, X_std)
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        return Y

    def predict_price_and_diffs(self, X, X_mean, X_std, Y_mean, Y_std, dYdX_mean, dYdX_std):
        # Forward Propagation
        X_norm = torch.div(torch.tensor([X]) - X_mean, X_std)
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        # Backward Propagation
        torch.autograd.grad(Y, torch.tensor([X]), retain_graph=True)
        Y.backward()
        dYdX_norm = X.grad
        dYdX = dYdX_norm * dYdX_std + dYdX_mean
        return Y, dYdX


def MSE_standard(model, X_norm, Y_norm):
    loss = torch.tensor(0.0)
    for x, y in zip(X_norm, Y_norm):
        x = torch.tensor([x])
        y_pred = model(x)[0]
        loss += torch.div(torch.square(y - y_pred), len(X_norm))
    return loss


def MSE_differential(model, X_norm, Y_norm, dYdX_norm, lambda_j, alpha):
    loss = alpha * MSE_standard(model, X_norm, Y_norm)
    if alpha != 1:
        for x, z in zip(X_norm, dYdX_norm):
            x = torch.tensor([x], requires_grad=True)
            y_pred = model(x)[0]
            y_pred.backward()
            z_pred = x.grad[0]
            loss += torch.div(torch.square(z - z_pred), len(X_norm)) * lambda_j * (1 - alpha)
    return loss


def training(model, X_norm, Y_norm, nb_epochs, dYdX_norm=None, lambda_j=None):
    # Cost Function
    if dYdX_norm is None:
        alpha = 1
    else:
        alpha = 1 / (1 + model.nb_inputs)
    # Cost Function
    criterion = MSE_differential
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=0.1)
    # Optimization Loop
    for i in range(0, nb_epochs):
        optimizer.zero_grad()
        loss = criterion(model, X_norm, Y_norm, dYdX_norm, lambda_j, alpha)
        # Update Weights
        loss.backward()
        optimizer.step()
        # Store Cost Value
        model.cost_values.append(loss.item())
        print(i+1, " - ", loss.item())
    return model
