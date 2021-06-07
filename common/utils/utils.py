from typing import Deque, Union

import numpy as np
import torch
import yaml



def read_config(config_path: str):
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    cfg["pricing_obs_dim"] = cfg["state_size"]
    cfg["pricing_action_dim"] = (cfg["max_battery"]+ cfg["max_energy_generated"]) * 6 + cfg["max_received"] + 1
    cfg["adl_obs_dim"] = cfg["state_size"]
    cfg["adl_action_dim"] = 8

    comm_cfg = {}
    comm_cfg["pubsub_port"] = cfg["pubsub_port"]
    comm_cfg["repreq_port"] = cfg["repreq_port"]
    comm_cfg["pullpush_port"] = cfg["pullpush_port"]

    return cfg, comm_cfg

def params_to_numpy(model):
    params = []
    state_dict = model.cpu().state_dict()
    for param in list(state_dict):
        params.append(state_dict[param])
    return params
