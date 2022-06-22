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
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem, cem_variable_length
from mppi import mppi
# from utils import make_video, save_frames_as_gif
# from gym.wrappers.monitoring import video_recorder
from utils import make_gif,make_video
from statsmodels.stats.proportion import proportion_confint

device = torch.device('cuda:0')

#env = 'antmaze-large-diverse-v0'
env = 'antmaze-medium-diverse-v0'
#env = 'maze2d-large-v1'

env_name = env
env = gym.make(env)
data = env.get_dataset()

skill_seq_len = 3
H = 30#10#10#5#10
replan_freq = H * 5
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 1000
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig = 0.0#None (Try None for antmaze)
n_iters = 10#10
# n_iters = 200
a_dist = 'normal'
keep_frac = 0.2
per_element_sigma = True#False
use_epsilon = True
max_ep = None
cem_l2_pen = 10.0
var_pen = 0.0
render = False
variable_length = False
max_replans = 2000 // H # run max 200 timesteps
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True
random_goal = False#True # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment

#filename = 'EM_model_05_08_22_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_a_dist_normal_log_best_sT.pth'
filename = 'EM_model_06_08_22_antmaze-medium-diverse-v0_batch_size_100state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_lr_5e-05_a_1.0_b_1.0_per_el_sig_True_a_dist_normal_log_best.pth'

PATH = 'checkpoints/antmaze_medium_skill_models/'+filename


if term_state_dependent_prior:
	skill_model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()
else:
	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

experiment = Experiment(api_key = 'wb7Q8i6WX2nuxXfHJdBSs9PUV', project_name = 'skill-learning')


s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True



def convert_epsilon_to_z(epsilon,s0,model):

	s = s0
	z_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_z, sigma_z = model.prior(s)
		z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		z_seq.append(z_i)
		s_mean,_ = model.decoder.abstract_dynamics(s,z_i)
		s = s_mean

	return torch.cat(z_seq,dim=1)


def run_skill_seq(skill_seq,env,s0,model,use_epsilon):
	state = s0

	pred_states = []
	pred_sigs = []
	states = []
	# plt.figure()
	for i in range(skill_seq.shape[1]):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		else:
			z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		state_torch = torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_state = s_mean.squeeze().detach().cpu().numpy()
		pred_sig = s_sig.squeeze().detach().cpu().numpy()
		pred_states.append(pred_state)
		pred_sigs.append(pred_sig)
		
		# run skill for H timesteps
		for j in range(H):
			#env.render()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
			plt.scatter(state[0],state[1], label='Trajectory',c='b')
		states.append(skill_seq_states)

	states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)

	return state,states

execute_n_skills = 1

min_dists_list = []
for j in range(1000):
	env.set_target() # this randomizes goal locations between trials, so that we're actualy averaging over the goal distribution
	# otherwise, same goal is kept across resets
	if not random_goal:
		
		goal_state = np.array(env.target_goal)#random.choice(data['observations'])
		print('goal_state: ', goal_state)
	else:
		N = data['observations'].shape[0]
		ind = np.random.randint(low=0,high=N)
		goal_state = data['observations'][ind,:]
	goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)

	state = env.reset()
	goal_loc = goal_state[:2]
	min_dist = 10**10
	for i in range(max_replans):
		if(i%50==0):
			print(i,'/',max_replans)
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
		skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
		skill_seq = skill_seq[:execute_n_skills,:]
		skill_seq = skill_seq.unsqueeze(0)	
		skill_seq = convert_epsilon_to_z(skill_seq,s_torch[:1,:,:],skill_model)
		state,states = run_skill_seq(skill_seq,env,state,skill_model,use_epsilon=False)

		dists = np.sqrt(np.sum((states[0,:,:2] - goal_loc)**2,axis=-1))

		if np.min(dists) < min_dist:
			min_dist = np.min(dists)

		if min_dist <= 0.5:
			break
		if(i%10==0):
			print(min_dist)
	min_dists_list.append(min_dist)

	p_succ = 0 #Incase need to resume experiments
	p_n_tot = 0 #Incase need to resume experiments
	n_success = np.sum(np.array(min_dists_list) <= 0.5)
	n_tot = len(min_dists_list)

	ci = proportion_confint(n_success+p_succ,n_tot+p_n_tot)
	print('ci: ', ci)
	print('mean: ',(n_success+p_succ)/(n_tot+p_n_tot))
	print('N = ',n_tot+p_n_tot)
	print('n_success = ,',n_success+p_succ)