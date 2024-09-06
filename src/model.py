import torch
import torch.nn as nn
from .load_config import load_constants_from_yaml
constants = load_constants_from_yaml('constants.yml')
first_layer_neurons = constants["FIRST_LAYER_NEURONS"]
second_layer_neurons = constants["SECOND_LAYER_NEURONS"]
input_shape = constants["INPUT_SHAPE"]

class Net(nn.Module):
    def __init__(self, input_shape, first_layer_neurons, second_layer_neurons):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, first_layer_neurons)
        self.fc2 = nn.Linear(first_layer_neurons, second_layer_neurons)
        self.output = nn.Linear(second_layer_neurons, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

model = Net(input_shape, first_layer_neurons, second_layer_neurons)