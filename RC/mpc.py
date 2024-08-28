import casadi as ca
import torch
import torch.nn as nn

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a trained model (for the sake of the example, we will use a simple untrained model)
input_dim = 2  # e.g., state dimension (position, velocity)
output_dim = 1  # e.g., control input
nn_model = SimpleNN(input_dim, output_dim)

# Convert PyTorch model to a CasADi function
def torch_to_casadi(nn_model):
    # Define CasADi symbols for state (x) and control input (u)
    x = ca.MX.sym('x', input_dim)
    u = ca.MX.sym('u', output_dim)

    # Convert NN to CasADi function
    def nn_func(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_tensor = nn_model(x_tensor)
        return y_tensor.detach().numpy().squeeze()

    # Create a CasADi function for the dynamics
    dynamics = ca.Function('dynamics', [x, u], [nn_func(ca.vertcat(x, u))])
    return dynamics

# Obtain CasADi dynamics function from NN model
nn_dynamics = torch_to_casadi(nn_model)

# Define MPC problem
T = 10  # time horizon
N = 20  # number of control intervals
dt = T/N  # time step

# Initialize state and control variables
x = ca.MX.sym('x', input_dim)
u = ca.MX.sym('u', output_dim)

# Cost function
cost = 0
x0 = x  # initial state
for k in range(N):
    cost += ca.mtimes(u.T, u)  # quadratic control cost
    x = nn_dynamics(x, u)  # apply the neural network model
    cost += ca.mtimes(x.T, x)  # quadratic state cost

# Constraints
constraints = []

# Define the optimization variables (controls over the horizon)
U = ca.MX.sym('U', output_dim, N)

# Create CasADi NLP solver
nlp = {'x': U, 'f': cost, 'g': ca.vertcat(*constraints)}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Initial guess for control inputs
u0 = ca.DM.zeros(output_dim, N)

# Solve the optimization problem
sol = solver(x0=u0)
optimal_u = sol['x']

# Display the results
print("Optimal control inputs:", optimal_u)

# Simulate the system with the optimal control input
x_sim = x0
x_trajectory = [x0]
for k in range(N):
    u_opt = optimal_u[:, k]
    x_sim = nn_dynamics(x_sim, u_opt)
    x_trajectory.append(x_sim)

# Print the state trajectory
print("State trajectory:", x_trajectory)

