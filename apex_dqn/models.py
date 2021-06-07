import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class PricingDoubleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PricingDoubleDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #the L2 regularizers of each layer and the final layer are
        #dealt during loss calculation and backward propagation
        self.qvals = nn.Sequential(
            nn.Linear(self.input_dim,32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

        self.network_optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001
        )



    def forward(self, state):
        qvals = self.qvals(state)

        return qvals

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_dim)).view(1, -1).size(1)

class ADLDoubleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ADLDoubleDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #the L2 regularizers of each layer and the final layer are
        #dealt during loss calculation and backward propagation
        self.qvals = nn.Sequential(
            nn.Linear(self.input_dim,16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_dim)
        )

        self.network_optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001
        )


    def forward(self, state):
        qvals = self.qvals(state)

        return qvals
