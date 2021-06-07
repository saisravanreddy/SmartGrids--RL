from typing import Deque, Union

import numpy as np
import ray
import torch.nn as nn

from common.abstract.architecture import Architecture
from common.utils.buffer_helper import PrioritizedReplayBufferHelper


class ApeX(Architecture):
    def __init__(
        self,
        worker_cls: type,
        learner_cls: type,
        brain: Union[tuple, nn.Module],
        cfg: dict,
        comm_cfg: dict,
    ):
        self.cfg = cfg
        self.comm_cfg = comm_cfg
        super().__init__(worker_cls, learner_cls, self.cfg)

        self.brain = brain

        # TODO in the original code, target dqns are ignored as above
        # only prediction DQN is sent whereas DoubleDQN the action selection
        # is done by prediction DQN is used and q value from target dqn is used
        self.worker_brain = self.brain

    def spawn(self):
        # Spawn all components
        # TODO uncomment this for enabling ray multiprocessing
        # self.workers = [
        #     self.worker_cls.remote(n, self.worker_brain, self.cfg, self.comm_cfg)
        #     for n in range(1, self.num_workers + 1)
        # ]
        # self.learner = self.learner_cls.remote(self.brain, self.cfg, self.comm_cfg)
        # self.global_buffer = PrioritizedReplayBufferHelper.remote(
        #     self.cfg, self.comm_cfg
        # )
        # self.all_actors = self.workers + [self.learner] + [self.global_buffer]

        # TODO selectively choose the actors from one of these during debugging multiple threads
        import sys
        if(sys.argv[1] == 'workers'):
            self.workers = [
                self.worker_cls(n, self.worker_brain, self.cfg, self.comm_cfg)
                for n in range(1, self.num_workers + 1)
            ]
            self.all_actors = self.workers
        if(sys.argv[1] == 'learner'):
            self.learner = self.learner_cls(self.brain, self.cfg, self.comm_cfg)
            self.all_actors = [self.learner]
        if(sys.argv[1] == 'buffer'):
            self.global_buffer = PrioritizedReplayBufferHelper(
                self.cfg, self.comm_cfg
            )
            self.all_actors = [self.global_buffer]
        print ("Initialized the actor to the required one",self.all_actors)

    def train(self):
        # TODO: implement a safer exit
        print("Running main training loop...")
        # self.global_buffer.learner = self.learner
        # while True:
        #     self.workers[0].run()
        #     self.global_buffer.run()
        #     #self.learner.run()

        [actor.run() for actor in self.all_actors]
        # ray.wait([actor.run.remote() for actor in self.all_actors]) # TODO uncomment ray.wait
