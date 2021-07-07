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

import os
from typing import List, Type
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from services.policy.rl.policy_rl import RLPolicy
from services.policy.rl.reinforce import REINFORCE, ValueNetwork
from services.policy.rl.experience_buffer import Buffer, MonteCarloBuffer
from services.service import Service, PublishSubscribe
from services.simulator.goal import Goal
from utils.beliefstate import BeliefState
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger
from utils.sysact import SysAct, SysActionType
from utils.useract import UserActionType


eps = np.finfo(np.float32).eps.item()


class ReinforcePolicy(RLPolicy, Service):

    def __init__(self, domain: JSONLookupDomain,
                 hidden_layer_sizes: List[int] = [256, 700, 700],
                 lr: float = 0.0001,
                 discount_gamma: float = 0.99,
                 baseline_update_rate: int = 1,
                 replay_buffer_size: int = 64,
                 batch_size: int = 1,
                 buffer_cls: Type[Buffer] = MonteCarloBuffer,
                 l2_regularisation: float = 0.0,
                 p_dropout: float = 0.0,
                 training_frequency: int = 2,
                 train_dialogs: int = 1000,
                 include_confreq: bool = False,
                 logger: DiasysLogger = DiasysLogger(),
                 max_turns: int = 25,
                 summary_writer: SummaryWriter = None,
                 device=torch.device('cpu')):
        RLPolicy.__init__(
            self,
            domain, buffer_cls=buffer_cls,
            buffer_size=replay_buffer_size, batch_size=batch_size,
            discount_gamma=discount_gamma, include_confreq=include_confreq,
            logger=logger, max_turns=max_turns, device=device)

        Service.__init__(self, domain=domain)

        self.writer = summary_writer
        self.training_frequency = training_frequency
        self.train_dialogs = train_dialogs
        self.lr = lr
        self.baseline_update_rate = baseline_update_rate

        # Create network architecture
        self.model = REINFORCE(self.state_dim, self.action_dim,
                               hidden_layer_sizes=hidden_layer_sizes, dropout_rate=p_dropout)

        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_regularisation)
        self.loss_fun = nn.MSELoss()

        # Create network update
        self.baseline_model = None
        if baseline_update_rate >= 1:
            self.logger.info("Update: with baseline")
            self.baseline_model = ValueNetwork(self.state_dim, self.action_dim, hidden_layer_sizes=hidden_layer_sizes)
            self.baseline_optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_regularisation)
        elif self.logger:
            self.logger.info("Update: without baseline")

        self.sim_goal = None
        self.train_call_count = 0
        self.total_train_dialogs = 0
        self.turns = 0
        self.cumulative_train_dialogs = -1

        self.saved_log_probs = []

        self.model = self.model.to(device)
        if self.baseline_model is not None:
            self.baseline_model = self.baseline_model.to(device)

    def dialog_start(self, dialog_start=False):
        self.turns = 0
        self.last_sys_act = None
        if self.is_training:
            self.cumulative_train_dialogs += 1
        self.sys_state = {
            "lastInformedPrimKeyVal": None,
            "lastActionInformNone": False,
            "offerHappened": False,
            'informedPrimKeyValsSinceNone': []}

    def select_action(self, state_vector: torch.FloatTensor):
        """ Select policy.

        Args:
            state_vector (torch.FloatTensor): current state (dimension 1 x state_dim)

        Returns:
            action index for action selected by the agent for the current state
        """
        state = state_vector.unsqueeze(0)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def get_log_prob(self, state_vector: torch.FloatTensor, action_vector: torch.FloatTensor):
        """ Select policy.

        Args:
            state_vector (torch.FloatTensor): current state (dimension 1 x state_dim)

        Returns:
            log_probs for action selected by the agent for the current state and action
        """
        state = state_vector.unsqueeze(0)
        action = action_vector.unsqueeze(0)
        probs = self.model(state)
        m = Categorical(probs)
        return m.log_prob(action)

    @PublishSubscribe(sub_topics=["sim_goal"])
    def end(self, sim_goal: Goal):
        """
            Once the simulation ends, need to store the simulation goal for evaluation

            Args:
                sim_goal (Goal): the simulation goal, needed for evaluation
        """
        self.sim_goal = sim_goal

    def dialog_end(self):
        """
            clean up needed at the end of a dialog
        """
        self.end_dialog(self.sim_goal)
        if self.is_training:
            self.total_train_dialogs += 1
        self.train_batch()

    @PublishSubscribe(sub_topics=["beliefstate"], pub_topics=["sys_act", "sys_state"])
    def choose_sys_act(self, beliefstate: BeliefState = None) -> dict(sys_act=SysAct):
        """
            Determine the next system act based on the given beliefstate

            Args:
                beliefstate (BeliefState): beliefstate, contains all information the system knows
                                           about the environment (in this case the user)

            Returns:
                (dict): dictionary where the keys are "sys_act" representing the action chosen by
                        the policy, and "sys_state" which contains additional informatino which might
                        be needed by the NLU to disambiguate challenging utterances.
        """
        self.num_dialogs = self.cumulative_train_dialogs % self.train_dialogs
        # if self.cumulative_train_dialogs == 0 and self.baseline_model is not None:
        #     # start with same weights for target and online net when a new epoch begins
        #     self.baseline_model.load_state_dict(self.model.state_dict())
        self.turns += 1
        if self.turns == 1 or UserActionType.Hello in beliefstate["user_acts"]:
            # first turn of dialog: say hello & don't record
            out_dict = self._expand_hello()
            out_dict["sys_state"] = {"last_act": out_dict["sys_act"]}
            return out_dict

        if self.turns > self.max_turns:
            # reached turn limit -> terminate dialog
            bye_action = SysAct()
            bye_action.type = SysActionType.Bye
            self.last_sys_act = bye_action
            # self.end_dialog(sim_goal)
            if self.logger:
                self.logger.dialog_turn("system action > " + str(bye_action))
            sys_state = {"last_act": bye_action}
            return {'sys_act': bye_action, "sys_state": sys_state}

        # intermediate or closing turn
        state_vector = self.beliefstate_dict_to_vector(beliefstate)
        next_action_idx = -1

        # check if user ended dialog
        if UserActionType.Bye in beliefstate["user_acts"]:
            # user terminated current dialog -> say bye
            next_action_idx = self.action_idx(SysActionType.Bye.value)
        if next_action_idx == -1:
            # dialog continues
            next_action_idx = self.select_action(state_vector)

        self.turn_end(beliefstate, state_vector, next_action_idx)

        # Update the sys_state
        if self.last_sys_act.type in [SysActionType.InformByName, SysActionType.InformByAlternatives]:
            values = self.last_sys_act.get_values(self.domain.get_primary_key())
            if values:
                # belief_state['system']['lastInformedPrimKeyVal'] = values[0]
                self.sys_state['lastInformedPrimKeyVal'] = values[0]
        elif self.last_sys_act.type == SysActionType.Request:
            if len(list(self.last_sys_act.slot_values.keys())) > 0:
                self.sys_state['lastRequestSlot'] = list(self.last_sys_act.slot_values.keys())[0]

        self.sys_state["last_act"] = self.last_sys_act
        return {'sys_act': self.last_sys_act, "sys_state": self.sys_state}

    def train_batch(self):
        """ Train on a minibatch drawn from the experience buffer. """
        if not self.is_training:
            return

        if self.total_train_dialogs % self.training_frequency == 0:
            self.train_call_count += 1

            s_batch, a_batch, r_batch, s2_batch, t_batch, _, _ = self.buffer.sample()
            s_batch = s_batch.unsqueeze(0)
            a_batch = a_batch.unsqueeze(0)

            # get reward
            R = 0
            returns = []
            for r in reversed(r_batch.tolist()):
                R = r[0] + self.discount_gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # get gradient
            self.optim.zero_grad()
            torch.autograd.set_grad_enabled(True)
            s_batch.requires_grad_()

            # update baseline-model
            if self.baseline_model is not None and self.train_call_count % self.baseline_update_rate == 0:
                # calculate loss of value function
                value_estimates = []
                for state in s_batch[0]:
                    state = state.unsqueeze(0).unsqueeze(0)
                    value_estimates.append(self.baseline_model(state))

                # rewards to go for each step of trajectory
                value_estimates = torch.stack(value_estimates).squeeze()

                v_loss = self.loss_fun(value_estimates, returns)
                # update the weights
                self.baseline_optim.zero_grad()
                v_loss.backward()
                self.baseline_optim.step()

            # get log_probabilities
            log_probs = []
            for s, a in zip(s_batch, a_batch):
                log_probs.append(self.get_log_prob(s, a))

            # calculate loss
            policy_loss = []
            if self.baseline_model is not None:
                # calculate advantage
                advantage = []
                for value, R in zip(value_estimates, returns):
                    advantage.append(R - value)

                advantage = torch.Tensor(advantage)

                # caluclate policy loss
                for log_prob, adv in zip(log_probs, advantage):
                    policy_loss.append(-log_prob * adv)
            else:
                for log_prob, R in zip(log_probs, returns):
                    policy_loss.append(-log_prob * R)

            loss = torch.cat(policy_loss).sum()
            loss.backward()

            # update weights
            self.optim.step()
            current_loss = loss.item()
            torch.autograd.set_grad_enabled(False)

            if self.writer is not None:
                # plot loss
                self.writer.add_scalar('train/loss', current_loss, self.train_call_count)
                # plot min/max gradients
                max_grad_norm = -1.0
                min_grad_norm = 1000000.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        # TODO decide on norm
                        current_grad_norm = torch.norm(param.grad, 2)
                        if current_grad_norm > max_grad_norm:
                            max_grad_norm = current_grad_norm
                        if current_grad_norm < min_grad_norm:
                            min_grad_norm = current_grad_norm
                self.writer.add_scalar('train/min_grad', min_grad_norm, self.train_call_count)
                self.writer.add_scalar('train/max_grad', max_grad_norm, self.train_call_count)

    def save(self, path: str = os.path.join('models', 'reinforce'), version: str = "1.0"):
        """ Save model weights

        Args:
            path (str): path to model folder
            version (str): appendix to filename, enables having multiple models for the same domain
                           (or saving a model after each training epoch)
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        model_file = os.path.join(
            path, "rlpolicy_" + self.domain.get_domain_name() + "_" + version + ".pt")
        torch.save(self.model, model_file)

    def load(self, path: str = os.path.join('models', 'reinforce'), version: str = "1.0"):
        """ Load model weights

        Args:
            path (str): path to model folder
            version (str): appendix to filename, enables having multiple models for the same domain
                           (or saving a model after each training epoch)
        """
        model_file = os.path.join(
            path, "rlpolicy_" + self.domain.get_domain_name() + "_" + version + ".pt")
        if not os.path.isfile(model_file):
            raise FileNotFoundError("Could not find REINFORCE policy weight file ", model_file)
        self.model = torch.load(model_file)
        self.logger.info("Loaded REINFORCE weights from file " + model_file)

    def train(self, train=True):
        """ Sets module and its subgraph to training mode """
        super(ReinforcePolicy, self).train()
        self.is_training = True
        self.model.train()

    def eval(self, eval=True):
        """ Sets module and its subgraph to eval mode """
        super(ReinforcePolicy, self).eval()
        self.is_training = False
        self.model.eval()
