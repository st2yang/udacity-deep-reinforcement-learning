from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.mlp = nn.Sequential(OrderedDict([
            ('mlp-linear0', nn.Linear(state_size, 64)),
            ('mlp-relu0', nn.ReLU(inplace=True)),
            ('mlp-linear1', nn.Linear(64, 64)),
            ('mlp-relu1', nn.ReLU(inplace=True)),
            ('mlp-linear2', nn.Linear(64, action_size))
        ]))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.mlp(state)
