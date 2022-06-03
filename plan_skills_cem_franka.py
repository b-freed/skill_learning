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
from utils import make_gif, make_video
from franka_reward_fn import FrankaRewardFn,franka_plan_cost_fn
from cem import cem
from wrappers import FrankaSliceWrapper

def run_skill(skill_model,s0,skill,env,H,render,use_epsilon=False):
	try:
		state = s0.flatten().detach().cpu().numpy()
	except:
		state = s0


	if use_epsilon:
		mu_z, sigma_z = skill_model.prior(torch.tensor(s0,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0))
		z = mu_z + sigma_z*skill
	else:
		z = skill
	
	states = [state]
	
	actions = []
	frames = []
	rewards = []
	rew_fn_rewards = []
	for j in range(H): #H-1 if H!=1
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,z)
		actions.append(action)
		state,r,_,_ = env.step(action)
		print('r: ', r)
		# r_fn = reward_fn.step(state)
		# print('r_fn: ', r_fn)
		
		states.append(state)
		rewards.append(r)

	return state,np.stack(states),np.stack(actions),np.sum(rewards),frames

def run_skill_with_disturbance(skill_model,s0,skill,env,H):
	state = s0.flatten().detach().cpu().numpy()
	states = []
	
	actions = []
	for j in range(H-1):
		action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
		
		if j == H//2:
			print('adding disturbance!!!!')
			action = action + 20
		actions.append(action)
		state,_,_,_ = env.step(action)
		
		states.append(state)
	  
	return np.stack(states),np.stack(actions)



if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	# env = 'antmaze-medium-diverse-v0'
	# env = 'kitchen-partial-v0'
	env_name = 'kitchen-complete-v0'
	env = gym.make(env_name)
	# env = FrankaSliceWrapper(gym.make(env))


	data = env.get_dataset()
	
	H = 10
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	batch_size = 100
	# episodes = 4
	wd = .01
	beta = 1.0
	alpha = 1.0
	state_dependent_prior = True
	a_dist = 'autoregressive' #'normal'
	max_sig = None
	fixed_sig = None
	state_dec_stop_grad = False # I don't think this really matters
	skill_seq_len = 50
	keep_frac = 0.5
	n_iters = 5
	n_additional_iters = 5
	cem_l2_pen = 0
	replan_iters = 500//H
	per_element_sigma = False
	render = False
	warm_start = False



	# if not state_dependent_prior:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
	# else:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'Franka_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_log_best.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best_sT.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_log_best_sT.pth'
	# filename = 'EM_model_kitchen-partial-v0_sliced_state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best_sT.pth'
	# filename = 'kitchen-partial-v0_per_el_sig_False_enc_type_state_action_sequencestate_dec_mlp_H_40_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best_sT.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.5_per_el_sig_False_log_best_sT.pth'
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_24_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best_sT.pth'
	# filename = 'EM_model_05_24_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_28_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_28_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	filename = 'EM_model_05_28_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_10_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'


	PATH = 'checkpoints/'+filename
	
	

	if not state_dependent_prior:
		skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	else:
		skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,per_element_sigma=per_element_sigma).cuda()
		
	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])
	
	
	# plan a sequence of skills
	# initial_state = env.reset()
	# prev_states = np.expand_dims(np.stack(batch_size*[initial_state]),1)  # add dummy time dimension
	# cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True)
	# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)

	# create experiment
	experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
	experiment.log_parameters({'env_name':env_name,
						   'filename':filename,
						   'H':H,
						   'fixed_sig':fixed_sig,
						   'max_sig':max_sig,
						   'cem_l2_pen':cem_l2_pen,
						   'n_iters':n_iters,
						   'n_additional_iters':n_additional_iters,
						   'skill_seq_len':skill_seq_len,
						   'warm_start':warm_start
						  })

	ep_rewards = []
	for j in range(1000):
		initial_state = env.reset()
		prev_states = np.expand_dims(initial_state,0)  # add dummy time dimension
		frames = []
		rewards = []
		t_since_last_save = 0
		state = initial_state
		for i in range(replan_iters):
			print('==============================================')
			print('i: ', i)
			# print('np.stack(np.stack(prev_states)).shape: ', np.stack(np.stack(prev_states)).shape)
			cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True)
			# run CEM on this cost fn
			if i == 0:
				skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
			else:
				if not warm_start: 
					skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
				else:
					skill_seq = torch.cat([skill_seq[1:,:],torch.zeros_like(skill_seq[:1,:])])
					init_sig =  torch.ones_like(skill_seq)
					init_sig[:-1,:] = .1*init_sig[:-1,:]

					skill_seq,_ = cem(skill_seq,init_sig,cost_fn,batch_size,keep_frac,n_additional_iters,l2_pen=cem_l2_pen)
			
			
			# run first skill
			

			z = skill_seq[:1,:]
			# skill_seq = skill_seq[1:,:]
			
			state,states_actual,actions,skill_rewards,skill_frames = run_skill(skill_model, state,z,env,H,render,use_epsilon=True)
			prev_states = np.concatenate([prev_states,states_actual[1:,:]],axis=0)
			print('states_actual.shape: ', states_actual.shape)
			frames += skill_frames
			rewards.append(skill_rewards)
			print('np.sum(rewards): ', np.sum(rewards))
			# make_video(frames,'franka_'+str(i))
			t_since_last_save += H
			if t_since_last_save >= 50 and render:
				make_video(frames,'franka_H_'+str(H)+filename)
				t_since_last_save = 0

		experiment.log_metric('reward',np.sum(rewards),step=j)
		ep_rewards.append(np.sum(rewards))
		experiment.log_metric('mean_reward',np.mean(ep_rewards),step=j)
	# actual_states = []
	# rewards = []
	# terminal_states = []
	# pred_states_sig = []
	# pred_states_mean = []
	# action_dist = []
	# frames = []
	# mses = []
	# state_lls = []
	# # collect an episode of data
	# initial_state = env.reset()
	# state = initial_state
	# render = True

	# for i in range(episodes):
	# 	print('i: ', i)

	# 	state = torch.reshape(torch.tensor(state,dtype=torch.float32).cuda(), (1,1,state_dim))
	# 	#actions = torch.tensor(actions,dtype=torch.float32).cuda()
		
	# 	if not state_dependent_prior:
	# 		z_mean = torch.zeros((1,1,z_dim), device=device)
	# 		z_sig = torch.ones((1,1,z_dim), device=device)
	# 	else:
	# 		z_mean,z_sig = skill_model.prior(state)
	# 		# print('z_mean: ', z_mean)

		
	# 	z = skill_model.reparameterize(z_mean,z_sig)
	# 	print('z: ', z)
	# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(state,z)

	# 	states_actual,actions,skill_rewards,skill_frames = run_skill(skill_model, state,z,env,H,reward_fn,render)
	# 	state = states_actual[-1,:]
	# 	terminal_states.append(state)
	# 	mses.append(np.mean((state - sT_mean.flatten().detach().cpu().numpy())**2))
	# 	state_dist = Normal.Normal(sT_mean, sT_sig )
	# 	state_ll = torch.mean(state_dist.log_prob(torch.tensor(state,dtype=torch.float32,device=device).reshape(1,1,-1)))
	# 	state_lls.append(state_ll.item())
	# 	frames += skill_frames
		
	# 	actual_states.append(states_actual)
	# 	action_dist.append(actions)
	# 	pred_states_mean.append(sT_mean[0,0,:].detach().cpu().numpy())
		
		
	# pred_states_mean = np.stack(pred_states_mean)
	# terminal_states = np.stack(terminal_states)
		
	# if render:
	# 	make_video(frames,'franka')

	# print('pred_states_mean.shape: ', pred_states_mean.shape)
	# print('terminal_states.shape: ', terminal_states.shape)
	# for i in range(terminal_states.shape[-1]):
	# 	plt.figure()
	# 	plt.plot(terminal_states[:,i])
	# 	plt.plot(pred_states_mean[:,i])
	# 	plt.savefig('states_'+str(i))

	# plt.figure()
	# plt.plot(mses)
	# plt.savefig('mses')

	# plt.figure()
	# plt.plot(state_lls)
	# plt.savefig('state_lls')

