
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import d4rl




skill_model = SkillModel() # TODO
# Load trained skill model

# get low-level polic
ll_policy = skill_model.decoder.ll_policy

# create env
env = 'maze2d-large-v1'  # maze whatever
env = gym.make(env)


# sample a skill vector from prior
z_sampled = 

# simulate low-level policy in env
state = env.reset()
for i in range(H):
    env.render()  # for visualization
    state = torch.tensor(state) # probably need to put this on the GPU and reshape it
    action = skill_model(state,z_sampled)
    state,_,_,_ = env.step(action)
