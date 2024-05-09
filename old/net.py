import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        self.n_hidden = 32

        self.fc1 = nn.Linear(in_dim, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.final = nn.Linear(self.n_hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final(x)

        return x