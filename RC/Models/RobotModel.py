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
    

class RobotDynamicsModel_v3(nn.Module):
    def __init__(self, input_size, output_size):
        super(RobotDynamicsModel_v3, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size, bias=True)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size, bias=True)
        self.fc3 = nn.Linear(4 * input_size, 8 * input_size, bias=True)
        self.fc4 = nn.Linear(8 * input_size, 16 * input_size, bias=True)
        self.fc5 = nn.Linear(16 * input_size, 32 * input_size, bias=True)
        self.fc6 = nn.Linear(32 * input_size, 64 * input_size, bias=True)
        self.fc7 = nn.Linear(64 * input_size, 32 * input_size, bias=True)
        self.fc8 = nn.Linear(32 * input_size, 16 * input_size, bias=True)
        self.fc9 = nn.Linear(16 * input_size, 8 * input_size, bias=True)
        self.fc10 = nn.Linear(8 * input_size, 4 * input_size, bias=True)
        self.fc11 = nn.Linear(4 * input_size, 2 * input_size, bias=True)
        self.output_layer = nn.Linear(2 * input_size, output_size, bias=True)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(2 * input_size)
        self.bn2 = nn.BatchNorm1d(4 * input_size)
        self.bn3 = nn.BatchNorm1d(8 * input_size)
        self.bn4 = nn.BatchNorm1d(16 * input_size)
        self.bn5 = nn.BatchNorm1d(32 * input_size)
        self.bn6 = nn.BatchNorm1d(64 * input_size)
        self.bn7 = nn.BatchNorm1d(32 * input_size)
        self.bn8 = nn.BatchNorm1d(16 * input_size)
        self.bn9 = nn.BatchNorm1d(8 * input_size)
        self.bn10 = nn.BatchNorm1d(4 * input_size)
        self.bn11 = nn.BatchNorm1d(2 * input_size)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.dropout(torch.relu(self.bn5(self.fc5(x))))
        x = self.dropout(torch.relu(self.bn6(self.fc6(x))))
        x = self.dropout(torch.relu(self.bn7(self.fc7(x))))
        x = self.dropout(torch.relu(self.bn8(self.fc8(x))))
        x = self.dropout(torch.relu(self.bn9(self.fc9(x))))
        x = self.dropout(torch.relu(self.bn10(self.fc10(x))))
        x = self.dropout(torch.relu(self.bn11(self.fc11(x))))
        x = self.output_layer(x)
        return x