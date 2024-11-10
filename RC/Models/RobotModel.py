import torch
import torch.nn as nn
class RobotDynamicsModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RobotDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size)
        self.fc3 = nn.Linear(4 * input_size, 6 * input_size)
        self.fc4 = nn.Linear(6 * input_size, 2 * input_size)
        self.output_layer = nn.Linear(2 * input_size, output_size)  # Output size is |x| = 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.output_layer(x)
        return x