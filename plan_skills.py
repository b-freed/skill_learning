'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoints'''

from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
import d4rl
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi

device = torch.device('cuda:0')

#env = 'antmaze-medium-diverse-v0'
env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
epochs = 100000
skill_seq_len = 10
lr = 5e-5
wd = 0
state_dependent_prior = True
'''
if not state_dependent_prior:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
else:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
'''
filename = 'maze2d_H'+str(H)+'_log_best.pth'

PATH = 'checkpoints/'+filename

if not state_dependent_prior:
  	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
  	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
s0 = torch.zeros((batch_size,1,state_dim), device=device)
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=lr)
# determine waypoints
goal_seq = 2*torch.rand((1,skill_seq_len,state_dim), device=device) - 1

experiment = Experiment(api_key = 'yQQo8E8TOCWYiVSruS7nxHaB5', project_name = 'skill-learning', workspace="anirudh-27")
experiment.add_tag('Skill PLanning for '+env_name)
experiment.log_parameters({'lr':lr,
							   'h_dim':h_dim,
							   'state_dependent_prior':state_dependent_prior,
							   'z_dim':z_dim,
                 				           'skill_seq_len':skill_seq_len,
			  				   'H':H,
			  				   'a_dim':a_dim,
			  				   'state_dim':state_dim,
			  				   'l2_reg':wd})
#experiment.log_metric('Goals', goal_seq)

for e in range(epochs):
	# Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
	exp_cost = skill_model.get_expected_cost(s0, skill_seq, goal_seq)
	seq_optimizer.zero_grad()
	exp_cost.backward()
	seq_optimizer.step()
	print(e)
	print("Cost: ", exp_cost)
	experiment.log_metric("Cost", exp_cost, step=e)

# Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment
ll_policy = skill_model.decoder.ll_policy

state = env.reset()
states = []
for i in range(skill_seq_len):
  	# get the skill
  	z = skill_seq[:,i:i+1,:]
  	skill_seq_states = []
  	# run skill for H timesteps
  	for j in range(H):
    		action = ll_policy.numpy_policy(state,z)
    		state,_,_,_ = env.step(action)
    
    		skill_seq_states.append(state)
  
  	states.append(skill_seq_states)

states = np.stack(states)

goals = goal_seq.detach().cpu().numpy()
goals = np.stack(goals)

plt.figure()
plt.scatter(states[:,:,0],states[:,:,1], label='Trajectory')
plt.scatter(states[:,0,0],states[:,0,1], label='Initial State')
plt.scatter(states[skill_seq_len-1,H-1,0],states[skill_seq_len-1,H-1,1], label='Final State')
plt.scatter(goals[:,:,0],goals[:,:,1], label='Goals')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol= 3)

if not state_dependent_prior:
	plt.savefig('Planned Skills (No State Dependent Prior)')
else:
	plt.savefig('Planned skills (State Dependent Prior)')


# compute test and train plan cost, plot so we can see what they;re doing






