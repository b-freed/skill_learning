from comet_ml import Experiment

experiment = Experiment(
    api_key="yQQo8E8TOCWYiVSruS7nxHaB5",
    project_name="general",
    workspace="anirudh-27",
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import gym
from pointmass_env import PointmassEnv

filename = 'log.pth'
H = 20
PATH = 'checkpoints/'+filename

state_dim = 4
a_dim = 2
h_dim = 128
skill_model = SkillModel(state_dim, a_dim, 20, h_dim).cuda() # TODO
# Load trained skill model
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# get low-level polic
ll_policy = skill_model.decoder.ll_policy

# create env
env = PointmassEnv()

z_dim = 20

# sample a skill vector from prior
input = torch.empty(1,H,z_dim)
z_prior_means = torch.zeros_like(input).cuda()
z_prior_sigs = torch.ones_like(input).cuda()
z_sampled = skill_model.reparameterize(z_prior_means, z_prior_sigs)

# simulate low-level policy in env
state = env.reset() # TODO: consider trying to reset to the same initial state every time
for i in range(H):
    env.render()  # for visualization
    state = torch.tensor(state).cuda() # probably need to put this on the GPU and reshape it
    action = ll_policy(state,z_sampled)
    state = env.step(action)
