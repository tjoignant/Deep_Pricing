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

    def forward(self, x: float):
        """
        Inputs:
         - x              : input (float)
        Outputs:
         - propagated input (float)
        """
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

    def predict_price(self, X: float, X_mean: float, X_std: float, Y_mean: float, Y_std: float):
        """
        Inputs:
         - X              : initial asset spot (float)
         - X_mean         : training samples mean (float)
         - X_std          : training samples stdev (float)
         - Y_mean         : training labels mean (float)
         - Y_std          : training labels stdev (float)
        Outputs:
         - predicted price (float)
        """
        # Forward Propagation
        X = torch.tensor([X])
        X_norm = torch.div(X - X_mean, X_std)
        Y_norm = self.forward(X_norm)[0]
        Y = Y_norm * Y_std + Y_mean
        return Y

    def predict_price_and_diffs(self, X: float, X_mean: float, X_std: float, Y_mean: float, Y_std: float,
                                dYdX_mean: float, dYdX_std: float):
        """
        Inputs:
         - X              : initial asset spot (float)
         - X_mean         : training samples mean (float)
         - X_std          : training samples stdev (float)
         - Y_mean         : training labels mean (float)
         - Y_std          : training labels stdev (float)
         - dYdX_mean      : training differentials mean (float)
         - dYdX_std       : training differentials stdev (float)
        Outputs:
         - predicted payoff (float)
         - predicted delta (float)
        """
        # Forward Propagation
        X = torch.tensor([X], requires_grad=True)
        X_norm = torch.div(X - X_mean, X_std)
        Y_norm = self.forward(X_norm)[0]
        Y = Y_norm * Y_std + Y_mean
        # Backward Propagation
        Y.backward()
        dYdX_norm = X.grad[0]
        dYdX = dYdX_norm * dYdX_std + dYdX_mean
        return Y, dYdX


def MSE_standard(model: Twin_Network, X_norm: list, Y_norm: list):
    """
    Inputs:
     - model          : neural network (Twin_Network())
     - X_norm         : training samples normalized (1D tensor)
     - Y_norm         : training labels normalized (1D tensor)
    Outputs:
     - standard mean squared error (float)
    """
    loss = torch.tensor(0.0)
    for x, y in zip(X_norm, Y_norm):
        x = torch.tensor([x])
        y_pred = model(x)[0]
        loss += torch.div(torch.square(y - y_pred), len(X_norm))
    return loss


def MSE_differential(model: Twin_Network, X_norm: list, Y_norm: list, dYdX_norm: list, lambda_j: float, alpha: float):
    """
    Inputs:
     - model          : neural network (Twin_Network())
     - X_norm         : training samples normalized (1D tensor)
     - Y_norm         : training labels normalized (1D tensor)
     - dYdX_norm      : training differentials normalized (1D tensor)
     - lambda_j       : cost function differential weight (float)
     - alpha          : cost function differential factor (float)
    Outputs:
     - differential mean squared error (float)
    """
    loss = alpha * MSE_standard(model, X_norm, Y_norm)
    if alpha != 1:
        ds = pow(10, -4)
        ds_norm = torch.div(ds - torch.mean(X_norm), torch.std(X_norm))
        for x, z in zip(X_norm, dYdX_norm):
            x_plus = torch.tensor([x+ds_norm])
            x_neg = torch.tensor([x-ds_norm])
            y_pred_plus = model(x_plus)[0]
            y_pred_neg = model(x_neg)[0]
            z_pred = (y_pred_plus - y_pred_neg) / (2*ds_norm)
            loss += torch.div(torch.square(z - z_pred), len(X_norm)) * lambda_j * (1 - alpha)
    return loss


def training(model: Twin_Network, X_norm: list, Y_norm: list, nb_epochs: int, dYdX_norm=None, lambda_j=None):
    """
    Inputs:
     - model          : neural network (Twin_Network())
     - X_norm         : normalized training samples (1D tensor)
     - Y_norm         : normalized training labels (1D tensor)
     - nb_epochs      : number of training iterations (int)
     - dYdX_norm      : normalized pathwise differentials (1D tensor)
     - lambda_j       : cost function differential weight (float)
    Outputs:
     - trained model (Twin_Network())
    """
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
    return model
