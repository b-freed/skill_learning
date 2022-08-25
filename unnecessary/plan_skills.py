'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoints'''

from tokenize import ContStr
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
import random
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
env = 'antmaze-large-diverse-v0'
env_name = env
env = gym.make(env)
data = env.get_dataset()

H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
epochs = 300000
skill_seq_len = 5
lr = 1e-4
wd = 0
state_dependent_prior = True
'''
if not state_dependent_prior:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
else:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
'''
filename = 'maze2d_H'+str(H)+'_log_best.pth'

PATH = 'checkpoints/canebrake/Apr_16/T_None_40_slp_None__r1/best.pth'
# PATH = 'checkpoints/'+filename

if not state_dependent_prior:
  	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
  	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()

checkpoint = torch.load(PATH, map_location='cpu')
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
s0 = env.reset()
# initial_loc = s0[:2]
# env.env.reset_to_location
print('s0: ', s0)
s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape(1,1,-1)

# z_mean,z_sig = skill_model.prior(s0_torch)
# print('z_mean: ', z_mean)
# print('z_sig: ', z_sig)
# skill_seq = z_mean.detach() + z_sig.detach() * torch.randn((1,skill_seq_len,z_dim), device=device)
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device)
skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

s0_torch = torch.stack(batch_size*[torch.tensor(s0,dtype=torch.float32).cuda().unsqueeze(0)])
# s0 = torch.zeros((batch_size,1,state_dim), device=device)

# s0_torch = torch.stack([env.reset_to_location(initial_loc)])
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=lr)
# determine waypoints
goal_seq = torch.tensor([[random.choice(data['observations'])]], device=device)
#goal_seq = 2*torch.rand((1,skill_seq_len,state_dim), device=device) - 1

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

def run_skill_seq(env,s0,model):
	'''
	'''
	env.env.set_state(s0[:2],s0[2:])
	state = s0
	# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape((1,1,-1))

	pred_states = []
	pred_sigs = []
	states = []
	for i in range(skill_seq_len):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
		z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		skill_seq_states = []
		state_torch = torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_states.append(s_mean.squeeze().detach().cpu().numpy())
		pred_sigs.append(s_sig.squeeze().detach().cpu().numpy())
		# run skill for H timesteps
		for j in range(H):
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
		states.append(skill_seq_states)

	states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)



	plt.figure()
	plt.scatter(states[:,:,0],states[:,:,1], label='Trajectory')
	plt.scatter(s0[0],s0[1], label='Initial States')
	# plt.scatter(states[:,-1,0],states[:,-1,1], label='Predicted Final States')
	plt.scatter(goals[:,:,0],goals[:,:,1], label='Goals')
	# print('pred_states: ', pred_states)
	# print('pred_states.shape: ', pred_states.shape)
	plt.scatter(pred_states[:,0],pred_states[:,1], label='Pred states')
	# plt.errorbar(pred_states[:,0],pred_states[:,1],xerr=pred_sigs[:,0],yerr=pred_sigs[:,1])
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol= 4)

	if not state_dependent_prior:
		plt.title('Planned Skills (No State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'_sdp_'+'false'+'.png')
	else:
		plt.title('Planned skills (State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'.png')

for e in range(epochs):
	# Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
	# print('goal_seq: ', goal_seq)
	if e % 100 == 0:
		run_skill_seq(env, s0, skill_model)

	exp_cost,pred_states = skill_model.get_expected_cost(s0_torch, skill_seq, goal_seq)
	seq_optimizer.zero_grad()
	exp_cost.backward()
	seq_optimizer.step()
	# print(e)
	# print("Cost: ", exp_cost)
	# ipdb.set_trace()
	if e % 100 == 0:
		experiment.log_metric("Cost", exp_cost, step=e)
		print('cost: ', exp_cost)
		print('e: ', e)
		plt.figure()
		plt.plot(pred_states[:,:,0].T.detach().cpu().numpy(),pred_states[:,:,1].T.detach().cpu().numpy())
		plt.scatter(goal_seq.flatten().detach().cpu().numpy()[0],goal_seq.flatten().detach().cpu().numpy()[1])
		plt.savefig('heeeere figgy figgy')
		

		
		
		# if exp_cost < 1.0:
		# 	break

# run_skill_seq(env,s0,skill_model.decoder.ll_policy)

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
plt.scatter(states[:,0,0],states[:,0,1], label='Initial States')
plt.scatter(states[:,-1,0],states[:,-1,1], label='Predicted Final States')
plt.scatter(goals[:,:,0],goals[:,:,1], label='Goals')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol= 4)



# compute test and train plan cost, plot so we can see what they;re doing






