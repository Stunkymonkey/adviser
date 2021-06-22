#!/usr/bin/env python3
import argparse
import gym
import numpy as np
from itertools import count
from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate', type=float, default=1e-2, metavar='G',
                    help='learning rate (default: 0.01)')
parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128], metavar='N N',
                    help='amount of hidden layers (default: 128)')
parser.add_argument('--dropout-rate', type=float, default=0.6, metavar='N',
                    help='dropout rate between hidden layers (default: 0.6)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='enable baseline calculation')
parser.add_argument('--baseline-iterations', type=int, default=1, metavar='N',
                    help='set the amount of iterations to train the V-Value (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class REINFORCE(nn.Module):
    """ Simple Deep Q-Network """

    def __init__(self, state_dim: int, action_dim: int, hidden_layer_sizes: List[int] = [300, 300],
                 dropout_rate: float = 0.0):
        """ Initialize a REINFORCE Network with an arbitrary amount of linear hidden
            layers """
        super(REINFORCE, self).__init__()

        self.dropout_rate = dropout_rate
        self.saved_log_probs = []
        self.rewards = []
        self.states = []

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
        self.layers.append(nn.Softmax(dim=1))

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
    def __init__(self, num_inputs, hidden_layer_sizes: List[int] = [300, 300]):
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


policy = REINFORCE(4, 2, hidden_layer_sizes=args.hidden_layers, dropout_rate=args.dropout_rate)
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
value_function = ValueNetwork(4, hidden_layer_sizes=args.hidden_layers)
value_optimizer = optim.Adam(value_function.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    if args.baseline:
        # loop over this a couple of times
        for _ in range(args.baseline_iterations):
            # calculate loss of value function using mean squared error
            value_estimates = []
            for state in policy.states:
                state = torch.from_numpy(state).float().unsqueeze(0)  # just to make it a Tensor obj
                value_estimates.append(value_function(state))

            value_estimates = torch.stack(value_estimates).squeeze()  # rewards to go for each step of env trajectory

            v_loss = F.mse_loss(value_estimates, returns)
            # update the weights
            value_optimizer.zero_grad()
            v_loss.backward()
            value_optimizer.step()

        # calculate advantage
        advantage = []
        for value, R in zip(value_estimates, returns):
            advantage.append(R - value)

        advantage = torch.Tensor(advantage)

        # caluclate policy loss
        for log_prob, adv in zip(policy.saved_log_probs, advantage):
            policy_loss.append(-log_prob * adv)
    else:
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    policy.rewards.clear()
    policy.saved_log_probs.clear()
    policy.states.clear()


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10_000):  # Don't infinite loop while learning
            policy.states.append(state)
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
