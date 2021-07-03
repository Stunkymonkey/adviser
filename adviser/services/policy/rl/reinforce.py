###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

from typing import List

import torch
import torch.nn as nn


class REINFORCE(nn.Module):
    """ Simple Deep Q-Network """

    def __init__(self, state_dim: int, action_dim: int, hidden_layer_sizes: List[int] = [300, 300],
                 dropout_rate: float = 0.0):
        """ Initialize a REINFORCE Network with an arbitrary amount of linear hidden
            layers """
        super(REINFORCE, self).__init__()
        print("Architecture: REINFORCE")

        # create layers
        self.layers = nn.ModuleList()
        current_input_dim = state_dim
        for layer_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_dim, layer_size))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                self.layers.append(nn.Dropout(p=dropout_rate))
            current_input_dim = layer_size
        # output layer
        self.layers.append(nn.Linear(current_input_dim, action_dim))
        self.layers.append(nn.Softmax(dim=2))

    def forward(self, state_batch: torch.FloatTensor):
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim

        Returns:
            output: tensor of size batch_size x action_dim
        """

        output = state_batch
        for layer in self.layers:
            output = layer(output)
        return output


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs: int, action_dim: int, hidden_layer_sizes: List[int] = [300, 300]):
        super(ValueNetwork, self).__init__()
        self.layers = nn.ModuleList()
        current_input_dim = num_inputs
        for layer_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_dim, layer_size))
            current_input_dim = layer_size
        self.layers.append(nn.Linear(current_input_dim, 1))

    def forward(self, state):
        output = state
        for layer in self.layers:
            output = layer(output)
        return output
