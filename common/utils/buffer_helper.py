from typing import Deque

import pyarrow as pa
import ray
import zmq

# from common.utils.buffer import PrioritizedReplayBuffer
from replay_memory.prioritized_memory import Memory


# @ray.remote # TODO uncomment ray.remote()
class PrioritizedReplayBufferHelper(object):
    def __init__(self, buffer_cfg: dict, comm_cfg: dict):
        self.cfg = buffer_cfg

        # unpack buffer configs
        self.max_num_updates = self.cfg["max_num_updates"]
        self.priority_alpha = self.cfg["priority_alpha"]
        self.priority_beta = self.cfg["priority_beta_start"]
        self.priority_beta_end = self.cfg["priority_beta_end"]
        self.priority_beta_increment = (
            self.priority_beta_end - self.priority_beta
        ) / self.max_num_updates

        self.batch_size = self.cfg["batch_size"]
        self.worker_buffer_size = self.cfg["worker_buffer_size"]

        # self.buffer = PrioritizedReplayBuffer(
        #     self.cfg["buffer_max_size"], self.priority_alpha
        # )

        self.minimum_buffersize_learning = self.cfg["minimum_buffersize_learning"]

        self.replaybuffersize = self.cfg["replaybuffersize"]


        self.buffers = []
        self.num_of_agents = 3
        for i in range(self.num_of_agents):
            self.buffers.append(Memory(self.replaybuffersize))



        # unpack communication configs
        self.repreq_port = comm_cfg["repreq_port"]
        self.pullpush_port = comm_cfg["pullpush_port"]

        # initialize zmq sockets
        print("[Buffer]: initializing sockets..")
        self.initialize_sockets()

    def initialize_sockets(self):
        # for sending batch to learner and retrieving new priorities
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REQ)
        self.rep_socket.connect(f"tcp://127.0.0.1:{self.repreq_port}")

        # for receiving replay data from workers
        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.pullpush_port}")

    def send_batch_recv_priors(self):
        batch_data = [[],[],[]]
        # send batch and request priorities (blocking recv)
        for i in range(self.num_of_agents):
            minibatch, idxs, is_weights = self.buffers[i].sample(self.batch_size)
            batch_data[i] = [minibatch, idxs, is_weights]
        batch_id = pa.serialize(batch_data).to_buffer()
        self.rep_socket.send(batch_id)

        # self.learner.run() # TODO comment this for ray multithreading
        # receive and update priorities
        new_priors_id = self.rep_socket.recv()
        print("Buffer received updated batch priorities from learner")
        all_agents_idxs, all_agents_errors = pa.deserialize(new_priors_id)

        for i in range(self.num_of_agents):
            idxs = all_agents_idxs[i]
            errors = all_agents_errors[i]
            for j in range(self.batch_size):
                self.buffers[i].update(idxs[j], errors[j])

        # update priority
        # for i in range(self.batch_size):
        #     idx = idxs[i]
        #     self.memory.update(idx, errors[i])




    def recv_data(self):
        new_replay_data_id = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass

        if new_replay_data_id:
            print("Buffer received action-reward pairs from worker")
            new_replay_data = pa.deserialize(new_replay_data_id)
            for i in range(self.num_of_agents):
                for replay_data in new_replay_data[i]:
                    self.buffers[i].addSample(replay_data)



    def run(self):
        # ray.util.pdb.set_trace()   # TODO uncomment for ray debugging
        while True:
            self.recv_data()
            if self.buffers[0].tree.n_entries >= self.minimum_buffersize_learning:
                self.send_batch_recv_priors()
            else:
                pass

            # return # TODO comment this for multithreading ray
