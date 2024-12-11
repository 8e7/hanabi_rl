import os
import time
from collections import OrderedDict
import json
import torch
import numpy as np
from sad_code import r2d2

def load_sad_model(weight_file, device):
    state_dict = torch.load(weight_file, map_location=device)
    input_dim = state_dict["net.0.weight"].size()[1]
    hid_dim = 512
    output_dim = state_dict["fc_a.weight"].size()[0]

    agent = r2d2.R2D2Agent(
        False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, False
    ).to(device)
    load_weight(agent.online_net, weight_file, device)
    return agent


def load_op_model(weight_file, idx, device):
    """load op models, op models was trained only for 2 player
    """
    if idx >= 0 and idx < 3:
        num_fc = 1
        skip_connect = False
    elif idx >= 3 and idx < 6:
        num_fc = 1
        skip_connect = True
    elif idx >= 6 and idx < 9:
        num_fc = 2
        skip_connect = False
    else:
        num_fc = 2
        skip_connect = True
    if not os.path.exists(weight_file):
        print(f"Cannot find weight at: {weight_file}")
        assert False

    state_dict = torch.load(weight_file)
    input_dim = state_dict["net.0.weight"].size()[1]
    hid_dim = 512
    output_dim = state_dict["fc_a.weight"].size()[0]
    agent = r2d2.R2D2Agent(
        False,
        3,
        0.999,
        0.9,
        device,
        input_dim,
        hid_dim,
        output_dim,
        2,
        5,
        False,
        num_fc_layer=num_fc,
        skip_connect=skip_connect,
    ).to(device)
    load_weight(agent.online_net, weight_file, device)
    return agent


def load_weight(model, weight_file, device):
    state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()
    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded" % k)
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            # print(target_state_dict.keys())
            print("removing: %s not used" % k)
            # state_dict.pop(k)
        else:
            source_state_dict[k] = state_dict[k]

    # if "pred.weight" in state_dict:
    #     state_dict.pop("pred.bias")
    #     state_dict.pop("pred.weight")

    model.load_state_dict(source_state_dict)
    return

