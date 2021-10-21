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
s_dim = 4
epochs = 100

# initialize skill sequence
skill_seq = torch.randn((1,H,s_dim), device=device)
s0 = skill_seq[:,0:1,:]
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=0.002)
# determine waypoints
goal_seq = 2*torch.rand((1,H,s_dim)) - 1

total_cost = 0

for e in range(epochs):
  # Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
  exp_cost = SkillModel.get_expected_cost(s0, skill_seq, goal_seq)
  seq_optimizer.zero_grad()
  seq_optimizer.step()

  total_cost += exp_cost
  # Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment

  # compute test and train plan cost, plot so we can see what they;re doing
