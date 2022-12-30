import torch
from torch import nn


class Net(nn.Module):
    def __init__(self,nb_inputs,hidL,nb_neurones):
        w=torch.empty(3,5)
        nn.init.kaiming_normal_(w,mode='fan_out',nonlinearity='relu')
        super().__init__()
        #(number of inputs, number of hidden layers, number of neurons)

        # Entrées dans la transformation linéaire de la couche cachée
        self.hidden1 = nn.Linear(nb_inputs, nb_neurones)
        self.hidden2 = nn.Linear(nb_neurones,nb_neurones)
        self.hidden3 = nn.Linear(nb_neurones,nb_neurones)
        self.output = nn.Linear(nb_neurones, 1)

        def forward(self, x):
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
            #x = F.relu(self.output(x))

            return x

net = Net(20,4,20)
print(net)
        # Couche de sortie, 10 unités - une pour chaque digit
        #self.output = nn.Linear(256, 10)

        # Définir l'activation sigmoïde et la sortie softmax
    #    self.relu = nn.ReLU()
    #    self.softmax = nn.ReLU(dim=1)

    #def forward(self, x):
        # Passez le tenseur d'entrée à travers chacune de nos opérations
    #    x = self.hidden(x)
    #    x = self.relu(x)
    #    x = self.output(x)
    #    x = self.softmax(x)

    #    return x