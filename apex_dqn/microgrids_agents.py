import tensorflow as tf
import torch
import os
import collections
import random
import numpy as np
import math
import copy
from datetime import datetime
from replay_memory.prioritized_memory import Memory
from abc import ABC, abstractmethod

minibatchsize = 32  #TODO transfer these agent parameters to the trainer.py files
targetupdatefrequency = 200
replaybuffersize = 1000000
# number of iterations
total_iterations = 130000


class Microgrid_Agent(ABC):
    def __init__(self, name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl,
                 grid_price, total_iterations, current_iteration, lam):

        self.name = name
        self.state_size = state_size
        self.max_battery = max_battery
        self.max_energy_generated = max_energy_generated
        self.max_received = max_received
        self.grid_price = grid_price
        self.action_size_pricing = (max_battery + max_energy_generated) * 6 + max_received + 1  # 6 for grid price to (grid price - 5) and 1 for the zeroth state.
        self.action_size_adl = 8
        self.total_iterations = total_iterations
        self.current_iteration = current_iteration
        self.memory = Memory(replaybuffersize)
        self.gamma = 0.90  # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0
        self.regularizer_loss = 50  # Loss for penalising the impossible actions
        self.regularizer_factor = 0.07
        self.reward_summary_writer = tf.summary.create_file_writer(
            "./logs/ApeXDoubleDQN_PER-" + datetime.now().strftime("%Y%m%d-%H%M%S") +
            '-itrs-' + str(total_iterations) +
            '-ms-' + str(minibatchsize) +
            '-tuf-' + str(targetupdatefrequency) +
            '-rbsize-'+str(replaybuffersize)+
            '-alpha-' + str(self.memory.a) +
            '-priority-epsilon-' + str(self.memory.e) +
            '-beta-' + str(self.memory.beta) +
            '-beta-inc-' + str(self.memory.beta_increment_per_sampling) +
            "/train_reward-" + str(self.name))
        self.pricing_summary_writer = tf.summary.create_file_writer(
            "./logs/ApeXDoubleDQN_PER-" + datetime.now().strftime("%Y%m%d-%H%M%S") +
            '-itrs-' + str(total_iterations) +
            '-ms-' + str(minibatchsize) +
            '-tuf-' + str(targetupdatefrequency) +
            '-rbsize-'+str(replaybuffersize)+
            '-alpha-' + str(self.memory.a) +
            '-priority-epsilon-' + str(self.memory.e) +
            '-beta-' + str(self.memory.beta) +
            '-beta-inc-' + str(self.memory.beta_increment_per_sampling) +
            "/train_pricing-" + str(self.name))
        self.adl_summary_writer = tf.summary.create_file_writer(
            "./logs/ApeXDoubleDQN_PER-" + datetime.now().strftime("%Y%m%d-%H%M%S") +
            '-itrs-' + str(total_iterations) +
            '-ms-' + str(minibatchsize) +
            '-tuf-' + str(targetupdatefrequency) +
            '-rbsize-'+str(replaybuffersize)+
            '-alpha-' + str(self.memory.a) +
            '-priority-epsilon-' + str(self.memory.e) +
            '-beta-' + str(self.memory.beta) +
            '-beta-inc-' + str(self.memory.beta_increment_per_sampling) +
            "/train_adl-" + str(self.name))

        self.non_adl = [3, 4, 5, 6]
        self.prob_non_adl = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.1, 0.3, 0.4, 0.2], [0.2, 0.3, 0.1, 0.4]]
        self.adl_value = [[1, 2], [1, 3], [2, 4]]
        self.lam = lam
        self.adl_state = 7

    # This functions updates target models weights = prediction models
    def update_target_models(self):
        # self.target_pricing_model.set_weights(self.prediction_pricing_model.get_weights())
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.target_pricing_model.parameters(), self.prediction_pricing_model.parameters()):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
        for param, new_param in zip(self.target_adl_model.parameters(), self.prediction_adl_model.parameters()):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
        # self.target_adl_model.set_weights(self.prediction_adl_model.get_weights())

    # This is the function that returns the renewable power generated by the Microgrid at the particular time. It samples from a poisson
    # distribution. We get the lamda value for the poisson distribution by preprocessing a data set.
    def get_renewable(self, time):
        energy = np.random.poisson(lam=self.lam[time - 1], size=1)
        energy = min([12, energy])  # clipping the value so that it can't exceede 8
        energy = int(math.floor(energy))
        return energy

    # This is the function that returns the non adl demand for the given grid at the particular time instant.
    def get_non_adl_demand(self, time):
        demand = np.random.choice(self.non_adl, p=self.prob_non_adl[time - 1])
        return int(demand)

    # Custom Loss function for the neural network
    def custom_loss(self, y_true, y_pred):
        loss = torch.mean(
             torch.sum((y_true - y_pred).pow(2)))  # need to put axis
        return loss

    def summary(self):
        self.prediction_pricing_model.summary()
        print('-------------------')
        self.prediction_adl_model.summary()
        print('-------------------')

    # Gives the range of allowed pricing actions given the state of the microgrid. Her we impose the design constraints
    # to get the range of possible actions. The design constraints give us the range of electricty values that can be traded. Negative values means that a grid is buying
    # while positve values mean that a grid is selling. This range is converted to represent range of pricing action indexes .Each index represents an action pair of the amount of
    # electricty to be traded and the price to be traded at. The function returns a tuple which has the lower bound index and the upper bound index on pricing actions that can be taken.
    # Whenever a Microgrid takes a decision to buy electricty we associate no price with it. When it decides to sell electricty
    # there are 6 possible prices that can be associated with it from GP to GP-5. So negative values(buying) have a single pricing action index associated with them
    # while each positve value of energy to be traded(selling) has six pricing action indexes associated with them.

    def pricing_convert_constraint_values_to_allowed_action_indices(self, state):

        # State contains nd, d, adl, t, gp
        nd = state[0]
        d = state[1]
        adl = self.adl_convert_allowed_indices_to_values(state[2])
        lower_bound = max(-1 * self.max_received, nd - self.max_battery - adl)
        upper_bound = nd + d - adl

        if (lower_bound <= 0):
            lower_bound_index = lower_bound + self.max_received
        else:
            lower_bound_index = self.max_received + (lower_bound - 1) * 6 + 1  # In order to account for the zero state

        if (upper_bound <= 0):
            upper_bound_index = upper_bound + self.max_received
        else:
            upper_bound_index = self.max_received + upper_bound * 6

        return lower_bound_index, upper_bound_index

    def batch_pricing_convert_constraint_values_to_allowed_action_indices(self, states_batch):

        # State contains nd, d, adl, t, gp
        nd_batch = states_batch[:,0]
        # nd = state[0]
        d_batch = states_batch[:,1]
        # d = state[1]
        adl_batch = self.batch_adl_convert_allowed_indices_to_values(states_batch[:,2])
        # adl = self.batch_adl_convert_allowed_indices_to_values(state[2])
        batchsize = adl_batch.shape[0]
        lower_bound_batch = torch.max(-1 * self.max_received * torch.ones(batchsize), nd_batch - self.max_battery - adl_batch)
        upper_bound_batch = nd_batch + d_batch - adl_batch

        # if else condition is changed into pytorch based boolean addition
        lower_bound_index_batch = \
            (lower_bound_batch <= 0) * (lower_bound_batch + self.max_received) \
            + \
            (lower_bound_batch > 0) * (self.max_received + (lower_bound_batch - 1) * 6 + 1)

        upper_bound_index_batch = \
            (upper_bound_batch <= 0) * (upper_bound_batch + self.max_received) \
            + \
            (upper_bound_batch > 0) * (self.max_received + upper_bound_batch * 6)

        return lower_bound_index_batch, upper_bound_index_batch


    # convert an pricing action index to the respective price quoted and energy to be traded
    def pricing_convert_allowed_indices_to_values(self, action):
        # return price and ut
        if (action <= self.max_received):
            return ([0, action - self.max_received])
        else:
            action = action - self.max_received - 1
            return ([action % 6 + self.grid_price - 5, action // 6 + 1])

    # Takes a pricing action based on the available pricing action indexes using an epsilon greedy policy(either choose a random action with epsilon proabilty
    #  or choose an action that maximises your expected reward)

    def pricing_action(self, state):

        lower_bound_index, upper_bound_index = self.pricing_convert_constraint_values_to_allowed_action_indices(state)
        check = random.uniform(0, 1)
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) != self.pricing_unallowed_number:
                possible_actions.append(i)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            # output = self.prediction_pricing_model.predict(state)
            output = self.prediction_pricing_model.forward(torch.FloatTensor(state)).cpu().detach().numpy()
            output = output + 1.0 - np.min(output)
            action_FLAGS = np.zeros(len(output[0]))
            np.put(action_FLAGS, possible_actions, 1)
            output = np.multiply(output, action_FLAGS)
            index = np.argmax(output)

            return index

    def argmax_price_Q_predictionNetwork_given_State(self, state):
        """

        :param state:
        :return: argmax(over action) Q(state, action , theta)
        """

        lower_bound_index, upper_bound_index = self.pricing_convert_constraint_values_to_allowed_action_indices(state)
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) != self.pricing_unallowed_number:
                possible_actions.append(i)
        # output = self.prediction_pricing_model.predict(state)
        output = self.prediction_pricing_model.forward(torch.FloatTensor(state)).cpu().detach().numpy()
        output = output + 1.0 - np.min(output)
        action_FLAGS = np.zeros(len(output[0]))
        np.put(action_FLAGS, possible_actions, 1)
        output = np.multiply(output, action_FLAGS)
        index = np.argmax(output)

        return index

    def batch_argmax_price_Q_predictionNetwork_given_State(self, states_batch):
        """

        :param state:
        :return: argmax(over action) Q(state, action , theta)
        """

        lower_bound_index_batch, upper_bound_index_batch = self.batch_pricing_convert_constraint_values_to_allowed_action_indices(states_batch)
        # state = np.array([state])
        # possible_actions = []
        # for i in range(lower_bound_index, upper_bound_index + 1):
        #     if i <= self.max_received:
        #         possible_actions.append(i)
        #     elif ((i - self.max_received) % 6) != self.pricing_unallowed_number:
        #         possible_actions.append(i)
        # output = self.prediction_pricing_model.predict(state)
        # output = self.prediction_pricing_model.forward(torch.FloatTensor(state)).cpu().detach().numpy()
        output = self.prediction_pricing_model.forward(states_batch)
        output = output + 1.0 - torch.min(output, dim=1).values.unsqueeze_(1)
        batchsize = states_batch.shape[0]
        actionsize = output.shape[1]
        lower_bound_index_batch_FULL = lower_bound_index_batch.unsqueeze_(1).repeat([1,actionsize])
        upper_bound_index_batch_FULL = upper_bound_index_batch.unsqueeze_(1).repeat([1,actionsize])
        action_indices = torch.arange(0,actionsize).unsqueeze_(0).repeat([batchsize,1])

        action_FLAGS_batch = (action_indices >= lower_bound_index_batch_FULL) * \
        (action_indices <= upper_bound_index_batch_FULL) * \
        ((action_indices <= self.max_received * torch.ones([batchsize, actionsize])) + \
          (action_indices - self.max_received * torch.ones([batchsize, actionsize])) % 6 != \
          self.pricing_unallowed_number * torch.ones([batchsize, actionsize]))

        output = output * action_FLAGS_batch
        indexes_batch = torch.argmax(output, dim=1)

        return indexes_batch


    # given the state and the possible pricing actions returns the maximum expected reward over all the possible pricing actions as predicted by the Pricing neural network
    def price_Q_targetNetwork_given_State_and_ActionIndex(self, state, index):
        """
        :param state:
        :return: Q(state,actionIndex,theta-minus)
        """
        # Qvector = self.target_pricing_model.predict(np.array([state]))
        Qvector = np.expand_dims(self.target_pricing_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),
                                axis=0)

        return Qvector[0][index]

    def batch_price_Q_targetNetwork_given_State_and_ActionIndex(self, states_batch, indexes_batch):
        """
        :param state:
        :return: Q(state,actionIndex,theta-minus)
        """
        # Qvector = self.target_pricing_model.predict(np.array([state]))
        # Qvector = np.expand_dims(self.target_pricing_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),
        #                        axis=0)

        Qvector = self.target_pricing_model.forward(states_batch)

        return Qvector.gather(1, indexes_batch.long().view(-1,1))


    # Returns an array of possible ADL actions. Given the ADL state returns an array of all the possible ADL actions taht can be taken
    # The ADL states are represented using a binary encoding. Each ADL state is an integer value between 0-7 where the integer's binary
    # form has a meaning associated with it. Take an ADL state of 6. 6 in binary is 110. Here ith index represents if the ith ADL demand has been
    # scheduled or not. 1 means that it has not been scheduled till now and 0 means that it has been completed. Similarly each ADL action
    # is also a integer between 0-7 with the binary form associated with it has a meaning. Take an ADL action of 6. 6 in binary is 110. Here ith
    # index means that the agent has decided to schedule the ith demand. So 110 means that the agent has decided to schedule the 3rd and 2nd ADL demand.
    def adl_give_possible_actions(self, state):

        possible_actions = []
        possible_actions.append(0)
        for i in range(3):
            if (state & 2 ** i):
                temp = copy.deepcopy(possible_actions)
                for j in range(len(temp)):
                    temp[j] += 2 ** i

                possible_actions.extend(temp)
        return possible_actions

    def batch_adl_give_possible_actions(self, states_batch):

        possible_actions = []
        possible_actions.append(0)
        for i in range(3):
            if (state & 2 ** i):
                temp = copy.deepcopy(possible_actions)
                for j in range(len(temp)):
                    temp[j] += 2 ** i

                possible_actions.extend(temp)
        return possible_actions


    # given an adl action taken returns the amount of energy that is needed to satisfy that ADL action
    def adl_convert_allowed_indices_to_values(self, action):
        adl = 0
        temp = action
        for j in range(3):
            if (temp % 2 == 1):
                adl += self.adl_value[j][0]
            temp = temp // 2
        return adl

    # given an adl action taken returns the amount of energy that is needed to satisfy that ADL action
    def batch_adl_convert_allowed_indices_to_values(self, actions_batch):
        adl_batch = 0
        temp_batch = actions_batch
        for j in range(3):
            adl_batch += (temp_batch % 2 ==1) * self.adl_value[j][0]
            temp_batch = temp_batch // 2
        return adl_batch


    # given a state returns the adl action using an epsilon greedy policy (take a random ADL actions or an action that maximises the predicted
    # expected reward )
    def adl_action(self, state):

        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        check = random.uniform(0, 1)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            # output = self.prediction_adl_model.predict(np.asarray([state]))
            output = np.expand_dims(self.prediction_adl_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),axis=0)
            output = output + 1.0 - np.min(output)
            action_FLAGS = np.zeros(len(output[0]))
            np.put(action_FLAGS, possible_actions, 1)
            output = np.multiply(output, action_FLAGS)
            index = np.argmax(output)

            return index

            # Returns the max expected rewards for all the possible ADL actions given the state
    def argmax_adl_Q_predictionNetwork_given_State(self, state):
        """

        :param state:
        :return: argmax(over action) Q(state, action, theta)
        """
        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        # output = self.prediction_adl_model.predict(np.asarray([state]))
        output = np.expand_dims(self.prediction_adl_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),
                                axis=0)
        output = output + 1.0 - np.min(output)
        action_FLAGS = np.zeros(len(output[0]))
        np.put(action_FLAGS, possible_actions, 1)
        output = np.multiply(output, action_FLAGS)
        index = np.argmax(output)

        return index

    def batch_argmax_adl_Q_predictionNetwork_given_State(self, states_batch):
        """

        :param state:
        :return: argmax(over action) Q(state, action, theta)
        """
        # possible_actions = sorted(self.batch_adl_give_possible_actions(state[2]))
        # output = np.expand_dims(self.prediction_adl_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),
        #                         axis=0)
        output = self.prediction_adl_model.forward(states_batch)
        output = output + 1.0 - torch.min(output, dim=1).values.unsqueeze_(1)
        batchsize = output.shape[0]
        actionsize = output.shape[1]
        action_indices = torch.arange(0, actionsize).unsqueeze_(0).repeat([batchsize,1])

        # only when an action index bit specifies 1 and this adl load is already satisfied, then this
        # action is not possible so its FLAG is false
        adl_states_batch = states_batch[:,2]
        action_FLAGS_batch = (action_indices & \
        torch.bitwise_not(adl_states_batch.long().unsqueeze_(1).repeat([1,actionsize])) == 0)

        output = output * action_FLAGS_batch
        indexes_batch = torch.argmax(output, dim=1)

        return indexes_batch


    def adl_Q_targetNetwork_given_State_and_ActionIndex(self, state, index):
        """

        :param state:
        :param index:
        :return: Q(state,actionIndex,theta-minus)
        """
        # Qvector = self.target_adl_model.predict(np.array([state]))
        Qvector = np.expand_dims(self.target_adl_model.forward(torch.FloatTensor(state)).cpu().detach().numpy(),
                                axis=0)
        return Qvector[0][index]

    def batch_adl_Q_targetNetwork_given_State_and_ActionIndex(self, states_batch, indexes_batch):
        """

        :param state:
        :param index:
        :return: Q(state,actionIndex,theta-minus)
        """
        Qvector = self.target_adl_model.forward(states_batch)

        return Qvector.gather(1, indexes_batch.long().view(-1,1))



        # Given the ADL action updates the ADL state. It returns the penalty if there is one for not satisfying the ADL demand.

    def update_adl(self, adl_action, time):
        self.adl_state = self.adl_state & (~adl_action)
        penalty = 0
        if time == 2 and (self.adl_state & 1):
            self.adl_state = self.adl_state & (~1)
            penalty = 1
        elif time == 3 and (self.adl_state & 2):
            self.adl_state = self.adl_state & (~2)
            penalty = 1
        elif time == 4 and (self.adl_state & 4):
            penalty = 2
            self.adl_state = 7
        if time == 4:
            self.adl_state = 7
        new_adl_state = copy.deepcopy(self.adl_state)
        return penalty, new_adl_state

        # Stores the action into the repaly memory buffer

    def remember(self, state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price):
        #self.memory.append((state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price))
        self.memory.addSample((state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price))

    def load_model(self, path_pricing, path_adl):
        self.prediction_pricing_model.load_weights(path_pricing)
        self.prediction_adl_model.load_weights(path_adl)
        self.update_target_models()

    # Samples a mini batch from the replay buffer and fits the neural networks on the mini batch
    def replay(self, replay_data, batch_size):

        # pytorch version of replay code start
        minibatch, idxs, is_weights = replay_data
        # TODO do all the replay operations on Tensors on GPU CUDA incase it is slow

        states_batch_adl = [sample[0] for sample in minibatch]
        states_batch_adl = torch.FloatTensor(states_batch_adl).to(self.device)
        states_batch_pricing = [sample[1] for sample in minibatch]
        states_batch_pricing = torch.FloatTensor(states_batch_pricing).to(self.device)
        actions_batch_adl = [sample[2] for sample in minibatch]
        actions_batch_adl = torch.FloatTensor(actions_batch_adl).to(self.device)
        actions_batch_pricing = [sample[3] for sample in minibatch]
        actions_batch_pricing = torch.FloatTensor(actions_batch_pricing).to(self.device)
        rewards_batch = [sample[4] for sample in minibatch]
        rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
        next_states_batch_adl = [sample[5] for sample in minibatch]
        next_states_batch_adl = torch.FloatTensor(next_states_batch_adl).to(self.device)
        next_states_batch_pricing = [sample[6] for sample in minibatch]
        next_states_batch_pricing = torch.FloatTensor(next_states_batch_pricing).to(self.device)

        if(self.device == 'cuda'):
            states_batch_adl.cuda(non_blocking=True)
            states_batch_pricing.cuda(non_blocking=True)
            actions_batch_adl.cuda(non_blocking=True)
            actions_batch_pricing.cuda(non_blocking=True)
            rewards_batch.cuda(non_blocking=True)
            next_states_batch_adl.cuda(non_blocking=True)
            next_states_batch_pricing.cuda(non_blocking=True)

        # getting the network values Q(s) and only targetting the value Q(s,a)
        targets_batch_pricing = self.prediction_pricing_model.forward(states_batch_pricing)
        targets_batch_action_pricing = rewards_batch / 180.0 + self.gamma * self.batch_price_Q_targetNetwork_given_State_and_ActionIndex(next_states_batch_pricing, self.batch_argmax_price_Q_predictionNetwork_given_State(next_states_batch_pricing)).squeeze_(1)
        pricing_batch_TD_error = torch.abs(targets_batch_pricing.gather(1,actions_batch_pricing.long().view(-1,1)).squeeze_(1) - targets_batch_action_pricing)
        targets_batch_pricing[torch.arange(batch_size), actions_batch_pricing.long()] = targets_batch_action_pricing

        targets_batch_adl = self.prediction_adl_model.forward(states_batch_adl)
        targets_batch_action_adl = rewards_batch / 180.0 + self.gamma * self.batch_adl_Q_targetNetwork_given_State_and_ActionIndex(next_states_batch_adl, self.batch_argmax_adl_Q_predictionNetwork_given_State(next_states_batch_adl)).squeeze_(1)
        adl_batch_TD_error = torch.abs(targets_batch_adl.gather(1,actions_batch_adl.long().view(-1,1)).squeeze_(1)- targets_batch_action_adl)
        targets_batch_adl[torch.arange(batch_size), actions_batch_adl.long()] = targets_batch_action_adl

        errors = pricing_batch_TD_error + adl_batch_TD_error

        # Pytorch version of replay code end

        """
        minibatch, idxs, is_weights = replay_data
        # minibatch, idxs, is_weights = self.memory.sample(batch_size)
        minibatch = np.array(minibatch)
        states_batch_pricing, targets_batch_pricing, regularizer_batch = [], [], []
        states_batch_adl, targets_batch_adl = [], []
        y_true = []
        errors = []

        for state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price in minibatch:
            dl_state, dl_next_state = np.array([state_price]), np.array([next_state_price])
            target_price = reward / 180.0 + self.gamma * self.price_Q_targetNetwork_given_State_and_ActionIndex(next_state_price, self.argmax_price_Q_predictionNetwork_given_State(next_state_price))
            # here use the prediction network should try to reach the target value only for the taken action, the other actions Q-values targets should be same as the prediction network
            # target_array_price = self.prediction_pricing_model.predict(dl_state)
            target_array_price = self.prediction_pricing_model.forward(torch.FloatTensor(dl_state)).cpu().detach().numpy()
            price_TD_error = abs(target_array_price[0][action_price] - target_price)
            target_array_price[0][action_price] = target_price

            dl_adl_state, dl_adl_next_state = np.array([state_adl]), np.array([next_state_adl])
            target_adl = reward / 180.0 + self.gamma * self.adl_Q_targetNetwork_given_State_and_ActionIndex(next_state_adl, self.argmax_adl_Q_predictionNetwork_given_State(next_state_adl))
            # here use the prediction network should try to reach the target value only for the taken action, the other actions Q-values targets should be same as the prediction network
            # target_array_adl = self.prediction_adl_model.predict(dl_adl_state)
            target_array_adl = self.prediction_adl_model.forward(torch.FloatTensor(dl_adl_state)).cpu().detach().numpy()
            adl_TD_error = abs(target_array_adl[0][action_adl] - target_adl)
            target_array_adl[0][action_adl] = target_adl

            errors.append(price_TD_error + adl_TD_error)

            states_batch_pricing.append(state_price)
            targets_batch_pricing.append(target_array_price[0])
            states_batch_adl.append(state_adl)
            targets_batch_adl.append(target_array_adl[0])

        # history_pricing = self.prediction_pricing_model.fit(np.array(states_batch_pricing),
        #                                                     np.array(targets_batch_pricing), epochs=1, verbose=0)
        # history_adl = self.prediction_adl_model.fit(np.array(states_batch_adl), np.array(targets_batch_adl), epochs=1,
        #                                             verbose=0)
        """
        # Keras fit equivalent pytorch training loop start
        pricing_y_true = torch.FloatTensor(targets_batch_pricing)
        pricing_y_pred = self.prediction_pricing_model.forward(torch.FloatTensor(states_batch_pricing))

        pricing_loss = self.custom_loss(pricing_y_true, pricing_y_pred)

        pricing_l2_lambda = 2e-4
        l2_reg = torch.tensor(0.)
        for param in self.prediction_pricing_model.parameters():
            l2_reg += torch.norm(param)
        pricing_loss += pricing_l2_lambda * l2_reg

        self.prediction_pricing_model.network_optimizer.zero_grad()
        pricing_loss.backward()
        # clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.prediction_pricing_model.network_optimizer.step()

        # Keras fit equivalent pytorch training loop start
        adl_y_true = torch.FloatTensor(targets_batch_adl)
        adl_y_pred = self.prediction_adl_model.forward(torch.FloatTensor(states_batch_adl))

        # loss = torch.mean(
        #     torch.sum(torch.square(y_true - y_pred), axis=1))  # need to put axis

        adl_loss = self.custom_loss(adl_y_true, adl_y_pred)

        adl_l2_lambda = 2e-4
        l2_reg = torch.tensor(0.)
        for param in self.prediction_adl_model.parameters():
            l2_reg += torch.norm(param)
        adl_loss += adl_l2_lambda * l2_reg

        self.prediction_adl_model.network_optimizer.zero_grad()
        adl_loss.backward()
        # clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.prediction_adl_model.network_optimizer.step()

        # loss_pricing = history_pricing.history['loss'][0]
        # loss_adl = history_adl.history['loss'][0]
        # self.epsilon = max(self.epsilon_min, (0.8 - self.current_iteration / self.total_iterations))

        self.current_iteration += 4

        # update priority
        # for i in range(batch_size):
        #     idx = idxs[i]
        #     self.memory.update(idx, errors[i])

        # return loss_adl, loss_pricing

        return idxs, errors, adl_loss, pricing_loss

    def save_model(self):
        self.prediction_pricing_model.save_weights(
            './saved/' + self.name + '_prediction_pricing_model_adl_pricing' + '.h5')
        self.prediction_adl_model.save_weights('./saved/' + self.name + '_prediction_adl_model_adl_pricing' + '.h5')


# This is the class which implements the Microgrid
class DoubleDQN_Agent_PER(Microgrid_Agent):
    # Initialise the Microgrid object and its variables
    def __init__(self, name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl,
                 grid_price, total_iterations, current_iteration, lam):
        super().__init__(name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl,
                 grid_price, total_iterations, current_iteration, lam)

        self.pricing_unallowed_number = 7


# This is similar as DQN_Agent but the agent can only quote a single price for selling
class DoubleDQN_Agent_PER_Price_Constant(Microgrid_Agent):

    def __init__(self, name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl,
                 grid_price, total_iterations, current_iteration, lam):
        super().__init__(name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl,
                 grid_price, total_iterations, current_iteration, lam)

        self.pricing_unallowed_number = 0