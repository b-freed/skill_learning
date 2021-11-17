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
epochs = 100000
skill_seq_len = 10

PATH = 'checkpoints/'+filename

skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
s0 = torch.zeros((batch_size,1,state_dim), device=device)
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=1e-4)
# determine waypoints
goal_seq = 2*torch.rand((1,skill_seq_len,state_dim), device=device) - 1

total_cost = 0

for e in range(epochs):
  # Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
  exp_cost = skill_model.get_expected_cost(s0, skill_seq, goal_seq)
  seq_optimizer.zero_grad()
  exp_cost.backward()
  seq_optimizer.step()

# Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment
ll_policy = skill_model.decoder.ll_policy

env = PointmassEnv()

state = env.reset()
states = []
for i in range(skill_seq_len):
  # get the skill
  z = skill_seq[:,i:i+1,:]
  skill_seq_states = []
  # run skill for H timesteps
  for j in range(H):
    action = ll_policy.numpy_policy(state,z)
    state = env.step(action)
    
    skill_seq_states.append(state)
  
  states.append(skill_seq_states)

states = np.stack(states)

goals = goal_seq.detach().cpu().numpy()
goals = np.stack(goals)

plt.figure()
plt.scatter(states[:,:,0],states[:,:,1])
plt.scatter(states[:,0,0],states[:,0,1])
plt.scatter(states[skill_seq_len-1,H-1,0],states[skill_seq_len-1,H-1,1])
plt.scatter(goals[:,:,0],goals[:,:,1])
plt.show()


# compute test and train plan cost, plot so we can see what they;re doing






