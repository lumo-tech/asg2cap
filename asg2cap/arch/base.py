import math

import numpy as np
import torch

BOS = 0
EOS = 1
UNK = 2


def l2norm(inputs, dim=-1):
    # inputs: (batch, dim_ft)
    norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
    inputs = inputs / norm.clamp(min=1e-10)
    return inputs


def gen_order_embeds(max_len, dim_ft):
    order_embeds = np.zeros((max_len, dim_ft))
    position = np.expand_dims(np.arange(0, max_len - 1).astype(np.float32), 1)
    div_term = np.exp(np.arange(0, dim_ft, 2) * -(math.log(10000.0) / dim_ft))
    order_embeds[1:, 0::2] = np.sin(position * div_term)
    order_embeds[1:, 1::2] = np.cos(position * div_term)
    return order_embeds
