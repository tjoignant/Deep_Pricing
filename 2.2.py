import torch
from torch import nn


class Twin_Network(nn.Module):
    def __init__(self,nb_inputs,hidL=4,nb_neurones=20):
        w=torch.empty(3,5)
        nn.init.kaiming_normal_(w,mode='fan_out',nonlinearity='relu')
        super().__init__()
        #(number of inputs, number of hidden layers, number of neurons)

        # Entrées dans la transformation linéaire de la couche cachée
        self.hidden1 = nn.Linear(nb_inputs, nb_neurones)
        self.hidden2 = nn.Linear(nb_neurones,nb_neurones)
        self.hidden3 = nn.Linear(nb_neurones,nb_neurones)
        self.hidden4 = nn.Linear(nb_neurones, nb_neurones)
        self.output=nn.Linear(nb_neurones,1)

        def forward(self, x):
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
            x = F.relu(self.hidden4(x))
            x=self.output(x)

            return x

twin_Network = Twin_Network(20,4,20)
print(twin_Network)
