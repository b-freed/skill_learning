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

device = torch.device('cuda:0')

filename = 'log.pth'
H = 100
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
z_prior_means = torch.zeros((1,1,z_dim),device=device)
z_prior_sigs   = torch.ones((1,1,z_dim),device=device)
z_sampled = skill_model.reparameterize(z_prior_means, z_prior_sigs)

# simulate low-level policy in env

state = env.reset() # TODO: consider trying to reset to the same initial state every time

# states is going to be a growing sequence of individual states.  So it will be 1xtxstate_dim
for i in range(H):
    #env.render()  # for visualization
    state = torch.reshape(torch.tensor(state,device=device),(1,1,-1))
    action = ll_policy(state,z_sampled)
    state = env.step(action)
