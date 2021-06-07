from copy import deepcopy

import ray
import torch
import torch.nn.functional as F

from common.abstract.learner import Learner
from zmq.sugar.stopwatch import Stopwatch
from torch.nn.utils import clip_grad_norm_

from apex_dqn.microgrids_agents import *


# @ray.remote  # TODO pass this argument for gpus, num_gpus=1
class MultiAgentDQNLearner(Learner):
    def __init__(self, brain, cfg: dict, comm_config: dict):
        super().__init__(brain, cfg, comm_config)
        self.num_step = self.cfg["num_step"]
        self.gamma = self.cfg["gamma"]
        self.tau = self.cfg["tau"]
        self.gradient_clip = self.cfg["gradient_clip"]
        self.q_regularization = self.cfg["q_regularization"]

        self.batchsize = self.cfg["batch_size"]
        self.num_of_agents = 3
        # Microgrids Agent code start

        # number of iterations
        self.total_iterations = 130000

        # constants
        self.state_size = 5
        self.max_battery = 12
        self.max_energy_generated = 12
        self.max_received = 10
        self.min_non_adl = 3
        self.max_non_adl = 6
        self.grid_price = 20

        # penalty for not satisfying a demand
        self.c_values = [10, 20, 30]
        self.c = self.c_values[2]

        self.num_of_agents = 3

        # lambda value for the poisson distribution of each micro grid for each time period of the day
        # Just copy the lambda for which you want to run from the config file and replace it here

        self.lamb = [[2.667e-07, 0.541, 6.5965, 4.3712], [8.8281, 10.2997, 9.8301, 9.7514],
                [8.8281, 10.2997, 9.8301, 9.7514]]
        ## Initialise the agents
        self.agents = []
        self.names = ['g1', 'g2', 'g3']
        for i in range(self.num_of_agents - 1):
            self.agents.append(DoubleDQN_Agent_PER_Price_Constant(self.names[i], self.state_size, self.max_battery, self.max_energy_generated,
                                                             self.max_received,
                                                             self.min_non_adl, self.max_non_adl, self.grid_price,
                                                             total_iterations - 30000, 0,
                                                             self.lamb[i]))
        self.agents.append(
            DoubleDQN_Agent_PER(self.names[2], self.state_size, self.max_battery, self.max_energy_generated, self.max_received, self.min_non_adl,
                                self.max_non_adl,
                                self.grid_price, total_iterations - 30000, 0, self.lamb[2]))

        for i in range(self.num_of_agents):
            self.agents[i].prediction_pricing_model = brain[4 * i]
            self.agents[i].target_pricing_model = brain[4 * i + 1]
            self.agents[i].prediction_adl_model = brain[4 * i + 2]
            self.agents[i].target_adl_model = brain[4 * i + 3]

            self.agents[i].prediction_pricing_model.to(self.device)
            self.agents[i].target_pricing_model.to(self.device)
            self.agents[i].prediction_adl_model.to(self.device)
            self.agents[i].target_adl_model.to(self.device)

            self.agents[i].device = self.device

        self.update_step = 0




    def write_log(self):
        print("TODO: incorporate Tensorboard...")

    # def learning_step(self, data: tuple):
        # states, actions, rewards, next_states, dones, weights, idxes = data
        #
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)
        #
        # # Toggle with comments if not using cuda
        # states.cuda(non_blocking=True)
        # actions.cuda(non_blocking=True)
        # rewards.cuda(non_blocking=True)
        # next_states.cuda(non_blocking=True)
        # dones.cuda(non_blocking=True)
        #
        # curr_q = self.network.forward(states).gather(1, actions.unsqueeze(1))
        # bootstrap_q = torch.max(self.target_network.forward(next_states), 1)[0]
        #
        # bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        # target_q = rewards + (1 - dones) * self.gamma ** self.num_step * bootstrap_q
        # weights = torch.FloatTensor(weights).to(self.device)
        # weights.cuda(non_blocking=True)
        # weights = weights.mean()
        #
        # q_loss = (
        #     weights * F.smooth_l1_loss(curr_q, target_q.detach(), reduction="none")
        # ).mean()
        # dqn_reg = torch.norm(q_loss, 2).mean() * self.q_regularization
        # loss = q_loss + dqn_reg
        #
        # self.network_optimizer.zero_grad()
        # loss.backward()
        # clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        # self.network_optimizer.step()
        #
        # for target_param, param in zip(
        #     self.target_network.parameters(), self.network.parameters()
        # ):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        #
        # new_priorities = torch.abs(target_q - curr_q).detach().view(-1)
        # new_priorities = torch.clamp(new_priorities, min=1e-8)
        # new_priorities = new_priorities.cpu().numpy().tolist()
        #
        # return loss, idxes, new_priorities

    def get_params(self):
        # model = deepcopy(self.network)
        # model = model.cpu()

        # list of 12 network params
        all_agents_params = []
        for i in range(self.num_of_agents):
            all_agents_params.append(self.params_to_numpy(self.agents[i].prediction_pricing_model))
            all_agents_params.append(self.params_to_numpy(self.agents[i].target_pricing_model))
            all_agents_params.append(self.params_to_numpy(self.agents[i].prediction_adl_model))
            all_agents_params.append(self.params_to_numpy(self.agents[i].target_adl_model))

        return all_agents_params
        # return self.params_to_numpy(self.network)
