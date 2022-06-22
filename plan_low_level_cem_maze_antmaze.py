from tokenize import ContStr
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import LowLevelDynamicsFF, Prior
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
from cem import cem
from utils import make_gif,make_video
from statsmodels.stats.proportion import proportion_confint

device = torch.device('cuda:0')

#env = 'antmaze-large-diverse-v0'
#env = 'antmaze-medium-diverse-v0'
#env = 'maze2d-large-v1'
env = 'maze2d-medium-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

action_seq_len = 20#40 maze
H = 5
replan_freq = H * 5
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 512
batch_size = 100#1000
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig =  0.0
n_iters = 5
# n_iters = 200
a_dist = 'normal'
keep_frac = 0.2

# background_img = mpimg.imread('maze_medium.png')
use_epsilon = True
goal_conditioned = False
max_ep = None
cem_l2_pen = 0.01 #(maze2d)
#cem_l2_pen = 10 #(antmaze)
var_pen = 0.0
render = False
variable_length = False
max_replans = 2000//H
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True

random_goals = True

#PATH_DYNAMICS = 'checkpoints/ll_dynamics_maze2d-medium-v1_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
#PATH_PRIOR = 'checkpoints/ll_prior_maze2d-medium-v1_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
#PATH_DYNAMICS = 'checkpoints/ll_dynamics_maze2d-large-v1_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
#PATH_PRIOR = 'checkpoints/ll_prior_maze2d-large-v1_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
#PATH_DYNAMICS = 'checkpoints/ll_dynamics_antmaze-large-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
#PATH_PRIOR = 'checkpoints/ll_prior_antmaze-large-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
PATH_DYNAMICS = 'checkpoints/ll_dynamics_antmaze-medium-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
PATH_PRIOR = 'checkpoints/ll_prior_antmaze-medium-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'

dynamics_model = LowLevelDynamicsFF(state_dim,a_dim,h_dim,deterministic=False).cuda()
checkpoint = torch.load(PATH_DYNAMICS)
dynamics_model.load_state_dict(checkpoint['model_state_dict'])
if(use_epsilon):
	prior = Prior(state_dim,a_dim,h_dim,goal_conditioned).cuda()
	checkpoint = torch.load(PATH_PRIOR)
	prior.load_state_dict(checkpoint['model_state_dict'])
	dynamics_model.prior = prior

goal_state_original = env.get_target()

def convert_epsilon_to_a(epsilon,s0,goal_seq,model):

	s = s0
	a_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_a, sigma_a = model.prior(s)
		a_i = mu_a + sigma_a*action_seq[:,i:i+1,:]
		#print(sigma_a)
		a_seq.append(a_i)
		s_mean,_ = model(s,a_i)
		s = s_mean

	return torch.cat(a_seq,dim=1)[0]

N_TRIALS = 300
N_SUCCESS = 0

dataset_states = data['observations']

for trials in range(N_TRIALS):
	#env.set_target()
	print('TRIALS DONE: ',trials)
	print('SUCCESSES: ',N_SUCCESS)
	print('SUCCESS RATE: ',N_SUCCESS,'/',trials)
	state_idx = np.random.randint(0,dataset_states.shape[0])
	if(random_goals):
		goal_state = dataset_states[state_idx,:2]
		env.set_target(goal_state)
	else:
		goal_state = np.array(env.get_target())
	goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)
	action_seq = torch.zeros((1,action_seq_len,a_dim),device=device)
	success_flag = False
	state = env.reset()
	goal_loc = goal_state[:2]
	print('NEW GOAL = ',goal_state)

	for i in range(max_replans):
		if(i%20==0):
			print(i)
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda action_seq: dynamics_model.get_expected_cost_for_cem(s_torch, action_seq, goal_seq, use_epsilon)
		action_seq,_ = cem(torch.zeros((action_seq_len,a_dim),device=device),torch.ones((action_seq_len,a_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
		action_seq = action_seq.unsqueeze(0)	
		action_seq = convert_epsilon_to_a(action_seq,s_torch[:1,:,:],goal_seq[:1,:,:],dynamics_model)
		#print(action_seq)
		for k in range(H):
			env.render()
			action_seq_np = action_seq.detach().cpu().numpy()
			state,_,_,_ = env.step(action_seq_np[k])
			dist_to_goal = np.sum((state[:2]-goal_state)**2)
			if(dist_to_goal <= 0.5):
				N_SUCCESS += 1
				success_flag = True
				print('Trial successful')
				break
		if(success_flag):
			break

	ci = proportion_confint(N_SUCCESS,trials+1)
	print('ci: ', ci)