from fileinput import filename
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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from utils import make_gif, make_video
from franka_reward_fn import FrankaRewardFn,franka_plan_cost_fn
from cem import cem
from wrappers import FrankaSliceWrapper
from utils import reparameterize,z_score

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
		# print('r: ', r)
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
	env = 'kitchen-partial-v0'
	# env = 'kitchen-complete-v0'
	env = gym.make(env)
	# env = FrankaSliceWrapper(gym.make(env))


	data = env.get_dataset()
	
	H = 5
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
	skill_seq_len = 100
	keep_frac = 0.5
	n_iters = 1000
	n_additional_iters = 20
	cem_l2_pen = 1.0
	replan_iters = 1000
	per_element_sigma = False
	render = True
	# initial_ind = 114075




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
	# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_09_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_24_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_24_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	# filename = 'EM_model_05_28_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'
	filename = 'EM_model_05_28_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best.pth'


	PATH = 'checkpoints/'+filename
	
	

	if not state_dependent_prior:
		skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	else:
		skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,per_element_sigma=per_element_sigma).cuda()
		
	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])
	
	data = env.get_dataset()
	states = data['observations']
	s_dim = states.shape[1]
	actions = data['actions']
	
	# plan a sequence of skills
	# initial_state = env.reset()
	# prev_states = np.expand_dims(np.stack(batch_size*[initial_state]),1)  # add dummy time dimension
	# cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True)
	# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)

	initial_state = env.reset()
	prev_states = np.expand_dims(initial_state,0)  # add dummy time dimension
	frames = []
	rewards = []
	state = initial_state
	# for i in range(replan_iters):
	# print('np.stack(np.stack(prev_states)).shape: ', np.stack(np.stack(prev_states)).shape)
	cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True)
	# run CEM on this cost fn
	# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
	# skill_seq = reparameterize(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device))
	# skill_seq = torch.zeros((skill_seq_len,z_dim),device=device)
	skill_seq = torch.normal(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device))


	ep_rewards = []
	for i in range(5):
		state = env.reset()
		# s0 = states[initial_ind,:]
		# # print('s0: ', s0)
		# # sim_state = env.sim.get_state()
		# # sim_state[:30] = s0[:30]
		# # env.sim.set_state(sim_state)
		# # env.sim.forward()
		frames = []
		states_actual = []
		actions_actual = []
		# state = s0
		
		rewards = []
		t_since_last_save = 0	
		for j in range(skill_seq_len):
			skill = skill_seq[j:j+1,:]


			# print(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0).shape)
			
			mu_z, sigma_z = skill_model.prior(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0))
			# z = reparameterize(mu_z,sigma_z)
			# z = mu_z + sigma_z*torch.normal(torch.zeros_like(mu_z),torch.ones_like(sigma_z))
			z = mu_z + skill*sigma_z
			# z = skill

			# print('z: ', z)
			# print('z scrore: ',z_score(z,mu_z,sigma_z))

			# assert False

			# state,states_actual,actions,skill_rewards,skill_frames = run_skill(skill_model, state,z,env,H,render,use_epsilon=True)
			sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0),z)

			last_state = state
			state,states_actual,actions,skill_rewards,skill_frames = run_skill(skill_model, state,z,env,H,render,use_epsilon=False)

				

			prev_states = np.concatenate([prev_states,states_actual[1:,:]],axis=0)
			frames += skill_frames
			rewards.append(skill_rewards)
			# print('np.sum(rewards): ', np.sum(rewards))
			# make_video(frames,'franka_'+str(i))
			t_since_last_save += H
			# if t_since_last_save >= 200 and render:
		
			make_video(frames,'debug_two_franka_H_'+str(H)+'_'+str(i)+'_'+filename)
			t_since_last_save = 0
			
			# plt.figure()
			# plt.plot(state - last_state)
			# plt.plot(sT_mean.squeeze().detach().cpu().numpy() - last_state)
			# ucb = sT_mean.squeeze().detach().cpu().numpy() - last_state + sT_sig.squeeze().detach().cpu().numpy()
			# lcb = sT_mean.squeeze().detach().cpu().numpy() - last_state - sT_sig.squeeze().detach().cpu().numpy()
			# plt.fill_between(np.arange(state.shape[-1]),ucb,lcb,alpha=.5)
			# plt.legend(['delta','pred_delta'])
			# plt.savefig('franka_states_'+str(j))

		ep_rewards.append(np.sum(rewards))

		print('ep_rewards: ', ep_rewards)


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

