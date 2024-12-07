import torch
import torch.nn as nn
class RobotDynamicsModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RobotDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size, bias=True)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size, bias=True)
        self.fc3 = nn.Linear(4 * input_size, 6 * input_size, bias=True)
        self.fc4 = nn.Linear(6 * input_size, 2 * input_size, bias=True)
        self.output_layer = nn.Linear(2 * input_size, output_size, bias=True)  # Output size is |x| = 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.output_layer(x)
        return x
    
class RobotDynamicsModel_v2(nn.Module):
    def __init__(self, input_size, output_size):
        super(RobotDynamicsModel_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size, bias=True)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size, bias=True)
        self.fc3 = nn.Linear(4 * input_size, 6 * input_size, bias=True)
        self.fc4 = nn.Linear(6 * input_size, 8 * input_size, bias=True)
        self.fc5 = nn.Linear(8 * input_size, 6 * input_size, bias=True)
        self.fc6 = nn.Linear(6 * input_size, 2 * input_size, bias=True)
        self.output_layer = nn.Linear(2 * input_size, output_size, bias=True)  # Output size is |x| = 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.output_layer(x)
        return x
    