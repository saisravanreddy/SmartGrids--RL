import numpy as np
import ray
import torch
import torch.nn as nn
import random

from common.abstract.worker import ApeXWorker
from apex_dqn.microgrids_agents import *


# @ray.remote # TODO uncomment for ray multithread
class RollOutWorker(ApeXWorker):
    def __init__(
        self, worker_id: int, worker_brain: nn.Module, cfg: dict, common_config: dict
    ):
        super().__init__(worker_id, worker_brain, cfg, common_config)


        # Non-distributed microgrids code paste start

        # number of iterations
        self.training_step = 1

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
                                                             self.total_iterations - 30000, 0,
                                                             self.lamb[i]))
        self.agents.append(
            DoubleDQN_Agent_PER(self.names[2], self.state_size, self.max_battery, self.max_energy_generated, self.max_received, self.min_non_adl,
                                self.max_non_adl,
                                self.grid_price, self.total_iterations - 30000, 0, self.lamb[2]))

        for i in range(self.num_of_agents):
            self.agents[i].prediction_pricing_model = worker_brain[4 * i]
            self.agents[i].target_pricing_model = worker_brain[4 * i + 1]
            self.agents[i].prediction_adl_model = worker_brain[4 * i + 2]
            self.agents[i].target_adl_model = worker_brain[4 * i + 3]

        self.states_price = []
        self.states_adl = []

        self.rewards = []
        self.battery = []
        self.renewable = []

        self.total_reward_for_display = [[], [], []]
        self.total_prices_for_display = [[], [], []]

        for i in range(self.num_of_agents):
            self.battery.append(0)
            self.renewable.append(self.agents[i].get_renewable(1))

        for i in range(self.num_of_agents):
            self.states_adl.append(
                [self.renewable[i] + self.battery[i] - min(self.agents[i].non_adl), min(self.agents[i].non_adl), 7, 1, self.grid_price])

        self.adl_actions = []
        self.adl_values = []
        # Take the initial adl actions using epsilon greedy
        for i in range(self.num_of_agents):
            act = self.agents[i].adl_action(self.states_adl[i])
            self.adl_actions.append(act)
            self.adl_values.append(self.agents[i].adl_convert_allowed_indices_to_values(act))

        for i in range(self.num_of_agents):
            self.states_price.append(
                [self.states_adl[i][0], self.states_adl[i][1], self.adl_actions[i], self.states_adl[i][3], self.states_adl[i][4]])

        # Non-distributed microgrids code paste end

        self.Iter = 0
        for i in range(self.num_of_agents):
            self.agents[i].device = self.device

    def transaction(self,actions):
        # we will divide the agents as buyers and sellers
        rewards = []
        for i in range(len(actions)):
            rewards.append(0)
        buyers = []
        sellers = []
        prices_dict = {}
        # actions : index, price , ut
        for i in range(len(actions)):
            if actions[i][2] < 0:
                buyers.append(actions[i])
            else:
                sellers.append(actions[i])
                if prices_dict.get(actions[i][1]) == None:
                    prices_dict[actions[i][1]] = 1
                else:
                    prices_dict[actions[i][1]] += 1

        sellers = sorted(sellers, key=lambda x: x[1])  # sort the sellers according to the prices they have quoted

        total_demand = 0
        total_supply = 0
        for i in range(len(sellers)):
            total_supply += sellers[i][2]

        for i in range(len(buyers)):
            total_demand += abs(buyers[i][2])

        sellers_earning = 0
        buyers_spending = 0

        if (total_demand >= total_supply):
            # you have to meet the extra demand using the external supply
            temp = total_demand
            for i in range(len(sellers)):
                buyers_spending += sellers[i][1] * sellers[i][2]
                temp -= sellers[i][2]

            buyers_spending += temp * self.grid_price
            # now we have both sellers_earning and buyers_spending
            for i in range(len(sellers)):
                rewards[sellers[i][0]] += sellers[i][1] * sellers[i][2]
            for i in range(len(buyers)):
                rewards[buyers[i][0]] += 1 * buyers_spending / total_demand * buyers[i][
                    2]  # buyers[i][2] is neagtive thats why not subtracting

        else:

            temp = total_demand
            i = 0
            while (i < len(sellers)):
                p = sellers[i][1]
                val = prices_dict.get(p)
                if val == 1:
                    if (sellers[i][2] <= temp):
                        rewards[sellers[i][0]] += sellers[i][1] * sellers[i][2]
                        buyers_spending += sellers[i][2] * sellers[i][1]
                        temp -= sellers[i][2]
                    elif (temp < sellers[i][2] and temp > 0):
                        rewards[sellers[i][0]] += temp * sellers[i][1]
                        buyers_spending += sellers[i][1] * temp
                        rewards[sellers[i][0]] += (sellers[i][2] - temp) * (self.grid_price - 5)
                        temp -= temp
                    else:
                        rewards[sellers[i][0]] += sellers[i][2] * (self.grid_price - 5)
                    i += 1

                elif val > 1:
                    store = 0
                    reward_common = 0
                    for k in range(i, i + val):
                        store += sellers[k][2]
                    if store != 0:
                        if (store <= temp):
                            reward_common = sellers[i][1] * store
                            buyers_spending += store * sellers[i][1]
                            temp -= store
                        elif (temp < store and temp > 0):
                            reward_common = temp * sellers[i][1]
                            buyers_spending += sellers[i][1] * temp
                            reward_common += (store - temp) * (self.grid_price - 5)
                            temp -= temp
                        else:
                            reward_common = store * (self.grid_price - 5)
                        for k in range(i, i + val):
                            rewards[sellers[k][0]] += (reward_common * sellers[k][2]) / store
                    i = i + val

            for i in range(len(buyers)):
                rewards[buyers[i][0]] += 1 * buyers_spending / total_demand * buyers[i][2]  # buyers[i][2] is negative

        return (rewards)
