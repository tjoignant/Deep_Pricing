import torch.nn as nn


class Twin_Network(nn.Module):
    def __init__(self, nb_inputs, nb_hidden_layer=4, nb_neurones=20):
        super().__init__()
        # Layers Initialization
        self.layers = []
        for i in range(0, nb_hidden_layer + 1):
            # If First Layer
            if i == 0:
                layer = nn.Linear(nb_inputs, nb_neurones)
            # If Output Layer
            elif i == nb_hidden_layer:
                layer = nn.Linear(nb_neurones, 1)
            # If Hidden Layer
            else:
                layer = nn.Linear(nb_neurones, nb_neurones)
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


twin_network = Twin_Network(20)
print(twin_network)
