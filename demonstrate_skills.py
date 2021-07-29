
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import d4rl
import gym




skill_model = SkillModel() # TODO
# Load trained skill model

# get low-level polic
ll_policy = skill_model.decoder.ll_policy

# create env
env = 'maze2d-large-v1'  # maze whatever
env = gym.make(env)
data = env.get_dataset()
states = data['observations'] # shape = (4000000,4)
actions = data['actions']
N = states.shape[0]


# sample a skill vector from prior
_,_,_,_, z_post_means, z_post_sigs = SkillModel.forward(states, actions)
z_prior_means = torch.zeros_like(z_post_means)
z_prior_sigs = torch.ones_like(z_post_sigs)
z_sampled = SkillModel.reparameterize(z_prior_means, z_prior_sigs)

# simulate low-level policy in env
state = env.reset()
for i in range(N):
    env.render()  # for visualization
    state = torch.tensor(state) # probably need to put this on the GPU and reshape it
    action = skill_model(state,z_sampled)
    state,_,_,_ = env.step(action)
