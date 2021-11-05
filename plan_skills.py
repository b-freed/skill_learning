'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoits'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
H = 100
state_dim = 4
a_dim = 2
h_dim = 128
z_dim = 20
batch_size = 100
epochs = 10000

PATH = 'checkpoints/'+filename

skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
skill_seq = torch.randn((1,H,z_dim), device=device)
s0 = torch.zeros((batch_size,1,state_dim), device=device)
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=0.002)
# determine waypoints
goal_seq = 2*torch.rand((1,H,state_dim), device=device) - 1

total_cost = 0

for e in range(epochs):
  # Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
  exp_cost = skill_model.get_expected_cost(s0, skill_seq, goal_seq)
  seq_optimizer.zero_grad()
  exp_cost.backward()
  seq_optimizer.step()

  # Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment

  # compute test and train plan cost, plot so we can see what they;re doing
