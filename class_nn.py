import torch
import torch.nn as nn


class Twin_Network(nn.Module):
    def __init__(self, nb_inputs, nb_hidden_layer, nb_neurones):
        super(Twin_Network, self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_neurones = nb_neurones
        self.layers = []
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
        X_norm = (X - X_mean) / X_std
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        return Y

    def predict_price_and_diffs(self, X, X_mean, X_std, Y_mean, Y_std,  dYdX_mean, dYdX_std):
        X_norm = (X - X_mean) / X_std
        Y_norm = self.forward(X_norm)
        Y = Y_norm * Y_std + Y_mean
        torch.autograd.grad(Y, X, retain_graph=True)
        Y.backward()
        dYdX_norm = X.grad
        dYdX = dYdX_norm * dYdX_std + dYdX_mean
        return Y, dYdX

    def training(self, X_norm, Y_norm, lambda_j, nb_epochs, dYdX_norm=None):
        # Case 1 : Training on Samples & Differentials
        if dYdX_norm:
            return None
        # Case 2 : Training on Samples
        else:
            return None


if __name__ == '__main__':
    torch.manual_seed(123)
    twin_network = Twin_Network(nb_inputs=20, nb_hidden_layer=4, nb_neurones=20)
    print(twin_network.nb_inputs)
