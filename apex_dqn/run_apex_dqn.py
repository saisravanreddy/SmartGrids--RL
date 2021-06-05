# import ray # TODO uncomment for ray multithreading
import torch
import pickle

from apex_dqn.MultiAgent_dqn_learner import MultiAgentDQNLearner
from apex_dqn.rollout_worker import RollOutWorker
from apex_dqn.models import PricingDoubleDQN,ADLDoubleDQN
from architectures.apex import ApeX
from common.utils.buffer_helper import PrioritizedReplayBufferHelper
from common.utils.utils import read_config

# ray.init() # TODO uncomment for ray multithreading


if __name__ == "__main__":

    cfg, comm_cfg = read_config("./apex_dqn/config.yml")

    # TODO uncomment below part for ray multithreading enable
    pricing_dqn_1 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    """
    pricing_target_dqn_1 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    adl_dqn_1 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])
    adl_target_dqn_1 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])

    pricing_dqn_2 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    pricing_target_dqn_2 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    adl_dqn_2 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])
    adl_target_dqn_2 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])

    pricing_dqn_3 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    pricing_target_dqn_3 = PricingDoubleDQN(cfg["pricing_obs_dim"], cfg["pricing_action_dim"])
    adl_dqn_3 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])
    adl_target_dqn_3 = ADLDoubleDQN(cfg["adl_obs_dim"], cfg["adl_action_dim"])

    brain = (
        pricing_dqn_1,pricing_target_dqn_1,
        adl_dqn_1,adl_target_dqn_1,
        pricing_dqn_2, pricing_target_dqn_2,
        adl_dqn_2, adl_target_dqn_2,
        pricing_dqn_3, pricing_target_dqn_3,
        adl_dqn_3, adl_target_dqn_3
        )

    with open("brain.pkl", 'wb') as output:
        pickle.dump(brain, output, pickle.HIGHEST_PROTOCOL)
    """

    with open("brain.pkl", 'rb') as input:
        brain = pickle.load(input)


    ApeXDQN = ApeX(RollOutWorker, MultiAgentDQNLearner, brain, cfg, comm_cfg)
    ApeXDQN.spawn()
    ApeXDQN.train()
