import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Deque

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
import zmq
import ray

import tensorflow as tf
# from common.utils.utils import create_env


class Worker(ABC):
    def __init__(
        self, worker_id: int, worker_brain: nn.Module, worker_cfg: dict, comm_cfg: dict
    ):
        self.worker_id = worker_id
        self.cfg = worker_cfg
        self.device = worker_cfg["worker_device"]

        random.seed(self.worker_id)

        self.seed = random.randint(1, 999)

        # unpack communication configs
        self.pubsub_port = comm_cfg["pubsub_port"]
        self.pullpush_port = comm_cfg["pullpush_port"]

        # initialize zmq sockets
        print(f"[Worker {self.worker_id}]: initializing sockets..")
        self.initialize_sockets()

    @abstractmethod
    def collect_data(self) -> list:
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    def synchronize(self, new_brain_params: list):
        # Initializing the agents' networks to the brain
        for i in range(self.num_of_agents):
            """Synchronize worker brain with parameter server"""
            for param, new_param in zip(self.agents[i].prediction_pricing_model.parameters(), new_brain_params[4*i]):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)
            for param, new_param in zip(self.agents[i].target_pricing_model.parameters(), new_brain_params[4*i+1]):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)
            for param, new_param in zip(self.agents[i].prediction_adl_model.parameters(), new_brain_params[4*i+2]):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)
            for param, new_param in zip(self.agents[i].target_adl_model.parameters(), new_brain_params[4*i+3]):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)


    def initialize_sockets(self):
        # for receiving params from learner
        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.pubsub_port}")

        # for sending replay data to buffer
        time.sleep(1)
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.pullpush_port}")

    def send_replay_data(self, replay_data):
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def receive_new_params(self):
        new_brain_params_id = False
        try:
            new_brain_params_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_brain_params_id:
            print("Rollout Worker received new parameters from learner")
            new_brain_params = pa.deserialize(new_brain_params_id)
            self.synchronize(new_brain_params)
            return True

    def run(self):
        # ray.util.pdb.set_trace() # TODO uncomment for ray debugging
        while True:
            local_buffer = self.collect_data()
            self.send_replay_data(local_buffer)
            self.receive_new_params()
            # return  # TODO comment for ray multithreading

class ApeXWorker(Worker):
    """Abstract class for ApeX distrbuted workers """

    def __init__(
        self, worker_id: int, worker_brain: nn.Module, cfg: dict, comm_cfg: dict
    ):
        super().__init__(worker_id, worker_brain, cfg, comm_cfg)
        self.nstep_queue = deque(maxlen=self.cfg["num_step"])
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["num_step"]
        self.total_iterations = self.cfg["total_iterations"]

    def collect_data(self, verbose=True):
        """Fill worker buffer until some stopping criterion is satisfied"""
        local_buffer = [[],[],[]]


        while (len(local_buffer[0]) < self.worker_buffer_size) and (self.Iter < self.total_iterations):
            print("current iteration:" + str(self.Iter))
            self.pricing_actions = []
            self.pricing_values = []
            # take the pricing action using epsilon greedy policy
            for i in range(self.num_of_agents):
                act = self.agents[i].pricing_action(self.states_price[i])
                self.pricing_actions.append(act)
                self.pricing_values.append([i] + self.agents[i].pricing_convert_allowed_indices_to_values(act))

            # adl_logger.info("{} {} {}".format(adl_actions[0], adl_actions[1], adl_actions[2]))
            # renewable_logger.info("{} {} {}".format(renewable[0], renewable[1], renewable[2]))
            # transmission_logger.info(
            #     "{} {} {}".format(pricing_values[0][2], pricing_values[1][2], pricing_values[2][2]))
            # battery_logger.info("{} {} {}".format(battery[0], battery[1], battery[2]))
            # price_logger.info("{} {} {}".format(pricing_values[0][1], pricing_values[1][1], pricing_values[2][1]))
            # nd_logger.info("{} {} {}".format(states_price[0][0], states_price[1][0], states_price[2][0]))

            # Rewards from trading given the actions
            rewards = self.transaction(self.pricing_values)

            # Rewards for not satisfying the non ADL demands and updation of the battery of each microgrid
            for i in range(self.num_of_agents):
                if (self.states_price[i][0] - self.adl_values[i] - self.pricing_values[i][2] < 0):
                    rewards[i] += self.c * (self.states_price[i][0] - self.adl_values[i] - self.pricing_values[i][2])
                    self.battery[i] = 0
                else:
                    self.battery[i] = self.states_price[i][0] - self.adl_values[i] - self.pricing_values[i][2]
                    self.battery[i] = min(self.battery[i], self.max_battery)

            self.adl_states = []
            # Rewards for not satisfying the ADL demands and updation of the ADL state
            for i in range(self.num_of_agents):
                penalty, updated_adl_state = self.agents[i].update_adl(self.adl_actions[i], self.states_adl[i][3])
                rewards[i] += self.c * -1 * penalty
                self.adl_states.append(updated_adl_state)
                self.total_reward_for_display[i].append(rewards[i])
                self.total_prices_for_display[i].append(self.pricing_values[i][1])

            temp_states_adl = []
            # Get the next state of the Microgrid, stores the state
            for i in range(self.num_of_agents):
                self.renewable[i] = self.agents[i].get_renewable((self.Iter + 1) % 4 + 1)
                temp = self.agents[i].get_non_adl_demand((self.Iter + 1) % 4 + 1)
                temp_states_adl.append(
                    [self.renewable[i] + self.battery[i] - temp, temp, self.adl_states[i], (self.Iter + 1) % 4 + 1, self.grid_price])

            temp_adl_actions = []
            temp_adl_values = []
            temp_states_price = []
            # Take the ADL action using epsilon greedy policy; to be used in the next iteration. Store it in temp_adl_action. At the end of the main loop we will make adl_action = temp_adl_action

            for i in range(self.num_of_agents):
                act = self.agents[i].adl_action(temp_states_adl[i])
                temp_adl_actions.append(act)
                temp_adl_values.append(self.agents[i].adl_convert_allowed_indices_to_values(act))

            for i in range(self.num_of_agents):
                temp_states_price.append(
                    [temp_states_adl[i][0], temp_states_adl[i][1], temp_adl_actions[i], temp_states_adl[i][3],
                     temp_states_adl[i][4]])

            # Store it in replay buffer.  states_adl holds the current state for the ADL network, states_price holds the current state for the price network
            # adl_actions, pricing_actions stores the actions for the ADL network and Pricing network. temp_states_adl and temp_states_price holds the next state
            # for the ADL Network and the Pricing Network.

            for i in range(self.num_of_agents):
                # self.agents[i].remember(states_adl[i], states_price[i], adl_actions[i], pricing_actions[i], rewards[i],
                #                    temp_states_adl[i], temp_states_price[i])

                local_buffer[i].append((self.states_adl[i], self.states_price[i], self.adl_actions[i], self.pricing_actions[i], rewards[i],
                                    temp_states_adl[i], temp_states_price[i]))

            for i in range(self.num_of_agents):
                # loss[i] = agents[i].replay(minibatchsize)
                # reward_summary = tf.Summary()
                # reward_summary.value.add(tag='Reward', simple_value=float(sum(self.total_reward_for_display[i])/len(self.total_reward_for_display[i])))
                # self.agents[i].reward_summary_writer.add_summary(reward_summary, self.training_step)
                with self.agents[i].reward_summary_writer.as_default():
                    tf.summary.scalar('Reward',float(sum(self.total_reward_for_display[i])/len(self.total_reward_for_display[i])), self.Iter)
            # agents[i].adl_summary_writer.flush()
            # logger_loss1.info("{} {} {}".format(loss[0][0], loss[1][0], loss[2][0]))
            # logger_loss2.info("{} {} {}".format(loss[0][1], loss[1][1], loss[2][1]))

            self.adl_values = temp_adl_values
            self.states_adl = temp_states_adl
            self.states_price = temp_states_price
            self.adl_actions = temp_adl_actions


            if (self.Iter + 1) % 10000 == 0:
                print("Number of iterations completed = {} and reaming = {}".format(self.Iter + 1,
                                                                                    self.total_iterations - self.Iter - 1))
                # logger_updates.info("Iteration number {}".format(Iter + 1))
                # logger_updates.info(
                #     'The average reward for agent 1 after is {}'.format(mean(total_reward_for_display[0])))
                # logger_updates.info(
                #     'The average reward for agent 2 after is {}'.format(mean(total_reward_for_display[1])))
                # logger_updates.info(
                #     'The average reward for agent 3 after is {}'.format(mean(total_reward_for_display[2])))
                # logger_updates.info(
                #     'The average prices for agent 1 after is {}'.format(mean(total_prices_for_display[0])))
                # logger_updates.info(
                #     'The average prices for agent 2 after is {}'.format(mean(total_prices_for_display[1])))
                # logger_updates.info(
                #     'The average prices for agent 3 after is {}'.format(mean(total_prices_for_display[2])))
                self.total_reward_for_display = [[], [], []]
                self.total_prices_for_display = [[], [], []]

            self.Iter += 1

        return local_buffer
