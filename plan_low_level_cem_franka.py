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
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from utils import make_gif, make_video
from franka_reward_fn import FrankaRewardFn,franka_plan_ll_cost_fn
from cem import cem
from random import sample
#from wrappers import FrankaSliceWrapper

def convert_epsilon_to_a(epsilon,s0,model):

	s = s0
	a_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_a, sigma_a = model.prior(s)
		a_i = mu_a #+ sigma_a*action_seq[:,i:i+1,:]
		a_seq.append(a_i)
		s_mean,_ = model(s,a_i)
		s = s_mean

	return torch.cat(a_seq,dim=1)[0]


if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	# env = 'antmaze-medium-diverse-v0'
	env = 'kitchen-partial-v0'
	#env = 'kitchen-mixed-v0'
	#env = 'kitchen-complete-v0'
	env = gym.make(env)

	data = env.get_dataset()
	
	action_seq_len = 50#100
	H = 1#10
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 512
	batch_size = 100
	wd = .01
	beta = 1.0
	alpha = 1.0
	state_dependent_prior = True
	a_dist = 'normal'
	max_sig = None
	fixed_sig = 0.0
	state_dec_stop_grad = False # I don't think this really matters
	keep_frac = 0.1
	n_iters = 1
	n_additional_iters = 1
	cem_l2_pen = 0.000
	replan_iters = 500//H#200#500//H
	per_element_sigma = False
	render = True

	random_goals = False

	#PATH_DYNAMICS = 'checkpoints/ll_dynamics_kitchen-complete-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
	#PATH_PRIOR = 'checkpoints/ll_prior_kitchen-complete-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
	PATH_DYNAMICS = 'checkpoints/ll_dynamics_kitchen-partial-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
	PATH_PRIOR = 'checkpoints/ll_prior_kitchen-partial-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
	#PATH_DYNAMICS = 'checkpoints/ll_dynamics_kitchen-mixed-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
	#PATH_PRIOR = 'checkpoints/ll_prior_kitchen-mixed-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'

	dynamics_model = LowLevelDynamicsFF(state_dim,a_dim,h_dim,deterministic=False).cuda()
	checkpoint = torch.load(PATH_DYNAMICS)
	dynamics_model.load_state_dict(checkpoint['model_state_dict'])
	prior = Prior(state_dim,a_dim,h_dim,False).cuda()
	checkpoint = torch.load(PATH_PRIOR)
	prior.load_state_dict(checkpoint['model_state_dict'])
	dynamics_model.prior = prior

	ep_rewards = []
	tasks = ['bottom burner','top burner','light switch','slide cabinet','hinge cabinet','microwave','kettle']

	for j in range(200):
		if(random_goals):
			ep_tasks = sample(tasks,4)
			franka_reward_fn = FrankaRewardFn(ep_tasks)
			print('TASKS: ',ep_tasks)
		initial_state = env.reset()
		prev_states = np.expand_dims(initial_state,0)  # add dummy time dimension
		frames = []
		rewards = []
		state = initial_state
		action_seq = torch.zeros((1,action_seq_len,a_dim),device=device)

		for i in range(replan_iters):
			s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
			cost_fn = lambda action_seq: franka_plan_ll_cost_fn(prev_states,action_seq,dynamics_model,use_eps=True)
			if i == 0:
				action_seq,_ = cem(torch.zeros((action_seq_len,a_dim),device=device),torch.ones((action_seq_len,a_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
			else:
				init_sig =  torch.ones_like(action_seq)
				action_seq,_ = cem(action_seq,init_sig,cost_fn,batch_size,keep_frac,n_additional_iters,l2_pen=cem_l2_pen)
			
			action_seq = action_seq.unsqueeze(0)	
			action_seq = convert_epsilon_to_a(action_seq,s_torch[:1,:,:],dynamics_model)
			
			for k in range(H):
				#env.render()
				action_seq_np = action_seq.detach().cpu().numpy()
				state,r,_,_ = env.step(action_seq_np[k])
				if(random_goals):
					r,_ = franka_reward_fn.step(state)
				state_expand = np.expand_dims(state,axis=0)
				rewards.append(r)

			prev_states = np.concatenate([prev_states,state_expand],axis=0)
			if(i%50==0):
				print(i,': Episode reward: ', np.sum(rewards))
		ep_rewards.append(np.sum(rewards))
		print('EPISODE: ',j)
		print('MEAN REWARD: ',sum(ep_rewards)/(j+1))
		print('STD DEV: ',np.std(ep_rewards[1:],ddof=1))