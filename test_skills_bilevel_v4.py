import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, BilevelSkillModelV4, LowLevelDynamicsFF
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
from utils import make_gif, make_video, reparameterize


def run_skill(skill_model,s0,skill,env,H,render,pred_state):
	state = s0.flatten().detach().cpu().numpy()
	states = [state]
	
	actions = []
	frames = []
	for j in range(H): #H-1 if H!=1
	# for j in range(40):
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
		actions.append(action)
		state,_,_,_ = env.step(action)
		
		states.append(state)
		if np.sum((state[:2] - pred_state[:2])**2) < .1:
			break
	  
	return np.stack(states),np.stack(actions),frames

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
	
	# env = 'maze2d-large-v1'
	# env = 'antmaze-medium-diverse-v0'
	env = 'antmaze-large-diverse-v0'
	# env = 'kitchen-partial-v0'
	env = gym.make(env)
	data = env.get_dataset()
	
	# H = 10
	H = 40
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	batch_size = 1
	episodes = 1
	wd = .001
	state_dependent_prior = True
	state_dec_stop_grad = True
	beta = 1.0
	alpha = 1.0
	max_sig = None
	fixed_sig = None
	state_decoder_type = 'mlp'
	n_skills = 1
	colors = 3*['b','g','r','c','m','y','k']


	# if not state_dependent_prior:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
	# else:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'Franka_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'Antmaze_H20_l2reg_0.001_stopgrad_log_best.pth'
	# filename = 'AntMaze_H20_l2reg_0.001_a_10.0_b_1.0_log_best.pth'
	# filename = 'AntMaze_H20_l2reg_0.001_a_1.0_b_0.01_sg_True_log_best.pth'
	# filename = 'AntMaze_H20_l2reg_0.001_a_1.0_b_1.0_sg_True_max_sig_None_fixed_sig_0.1_log_best.pth'#'AntMaze_H20_l2reg_0.001_a_1.0_b_0.01_sg_False_log_best.pth'
	# filename = 'Noisy2_maze2d_H20_l2reg_0_log_best.pth'
	# filename = 'maze2d_H40_log_best.pth'
	# filename = 'AntMaze_H20_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
	# filename = 'bilevelv4_antmaze-large-diverse-v0state_dec_mlp_H_10_l2reg_0.001_a_1.0_b_1.0_log_best.pth'
	# filename = 'bilevelv4_antmaze-large-diverse-v0state_dec_mlp_H_10_l2reg_0.001_a_0.1_b_1.0_log_best.pth'
	# filename = 'bilevelv4_antmaze-large-diverse-v0state_dec_mlp_H_10_l2reg_0.001_a_0.1_b_1.0_log_best.pth'
	filename = 'bilevelv4_antmaze-large-diverse-v0state_dec_mlp_H_40_l2reg_0.001_a_0.1_b_1.0_log_best.pth'


	PATH = 'checkpoints/'+filename
	
	ll_dynamics_path = 'checkpoints/ll_dynamics_antmaze-large-diverse-v0_l2reg_0.001_lr_0.0001_log.pth'
	
	ll_dynamics = LowLevelDynamicsFF(state_dim,a_dim,h_dim).cuda()
	lld_checkpoint = torch.load(ll_dynamics_path)
	ll_dynamics.load_state_dict(lld_checkpoint['model_state_dict'])


	skill_model = BilevelSkillModelV4(state_dim,a_dim,z_dim,h_dim,alpha=alpha,beta=beta,state_decoder_type=state_decoder_type).cuda() 

	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])


	





	actual_states = []
	terminal_states = []
	pred_states_sig = []
	pred_states_mean = []
	action_dist = []
	frames = []
	mses = []
	state_lls = []
	# collect an episode of data

	render = False

	f1 = plt.figure(1)
	f2 = plt.figure(2)
	initial_state = env.reset()
	state = initial_state
	for j in range(n_skills):
		
		state = initial_state

		# initial_state = state

		state = torch.reshape(torch.tensor(state,dtype=torch.float32).cuda(), (1,1,state_dim))
		#actions = torch.tensor(actions,dtype=torch.float32).cuda()

		if not state_dependent_prior:
			z_mean = torch.zeros((1,1,z_dim), device=device)
			z_sig = torch.ones((1,1,z_dim), device=device)
		else:
			z_mean,z_sig = skill_model.prior(state)

		z = reparameterize(z_mean,z_sig)

		for i in range(episodes):
			# initial_state = env.reset()
			env.set_state(initial_state[:15],initial_state[15:])
			state = initial_state
			print('i: ', i)

			state = torch.reshape(torch.tensor(state,dtype=torch.float32).cuda(), (1,1,state_dim))
			#actions = torch.tensor(actions,dtype=torch.float32).cuda()
			
			# if not state_dependent_prior:
			# 	z_mean = torch.zeros((1,1,z_dim), device=device)
			# 	z_sig = torch.ones((1,1,z_dim), device=device)
			# else:
			# 	z_mean,z_sig = skill_model_sdp.prior(state)
			
			# z = skill_model_sdp.reparameterize(z_mean,z_sig)
			sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(state,z)
			#ipdb.set_trace()
			
			sT_samples = ll_dynamics.sample_from_sT_dist(torch.cat(10*[state]),torch.cat(10*[z]),skill_model.decoder.ll_policy,H)

		# 	# infer the skill
		# 	z_mean,z_sig = skill_model.encoder(states,actions)

		# 	z = skill_model.reparameterize(z_mean,z_sig)

		# 	# from the skill, predict the actions and terminal state
		# 	# sT_mean,sT_sig,a_mean,a_sig = skill_model.decoder(states,z)
		# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(states[:,0:1,:],z)
			

			states_actual,actions,skill_frames = run_skill(skill_model, state,z,env,H,render,sT_mean.flatten().detach().cpu().numpy())
			state = states_actual[-1,:]
			terminal_states.append(state)
			mses.append(np.mean((state - sT_mean.flatten().detach().cpu().numpy())**2))
			state_dist = Normal.Normal(sT_mean, sT_sig )
			state_ll = torch.mean(state_dist.log_prob(torch.tensor(state,dtype=torch.float32,device=device).reshape(1,1,-1)))
			state_lls.append(state_ll.item())
			frames += skill_frames
			# states_actual,actions = run_skill_with_disturbance(skill_model_sdp, states[:,0:1,:],z,env,H)
			
			# print('states_actual.shape: ', states_actual.shape)

			
			plt.figure(1)
			plt.scatter(states_actual[:,0],states_actual[:,1],c=colors[j])
			# plt.scatter(states_actual[0,0],states_actual[0,1],c=colors[j])
			# plt.scatter(states_actual[-1,0],states_actual[-1,1],c=colors[j])
			# plt.scatter(states_actual[0,0],states_actual[0,1],c=colors[j])
			plt.errorbar(sT_mean[0,0,0].detach().cpu().numpy(),sT_mean[0,0,1].detach().cpu().numpy(),xerr=sT_sig[0,0,0].detach().cpu().numpy(),yerr=sT_sig[0,0,1].detach().cpu().numpy(),c=colors[j])
			# plt.scatter(sT_mean[0,0,0].detach().cpu().numpy(),sT_mean[0,0,1].detach().cpu().numpy(),c=colors[j],marker='x')

			plt.figure(2)
			plt.scatter(sT_samples[0,0,:].detach().cpu().numpy(),sT_samples[1,0,:].detach().cpu().numpy())

		actual_states.append(states_actual)
		action_dist.append(actions)
		pred_states_mean.append(sT_mean[0,0,:].detach().cpu().numpy())
		

	plt.figure(1)
	plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
	plt.title('Skill Execution & Prediction (Skill-Dependent Prior) '+str(i))
	plt.axis('square')
	# plt.savefig('Skill_Prediction_H'+str(H)+'_'+str(i)+'.png')
	plt.savefig('Skill_Prediction_H'+str(H)+'_'+'.png')

	plt.figure(2)
	# plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
	# plt.title('Skill Execution & Prediction (Skill-Dependent Prior) '+str(i))
	plt.axis('square')
	# plt.savefig('Skill_Prediction_H'+str(H)+'_'+str(i)+'.png')
	plt.savefig('sT_samples'+str(H)+'_'+'.png')
		
		
		
		# pred_states_sig.append([sT_sig[0,-1,0].detach().cpu().numpy(),sT_sig[0,-1,1].detach().cpu().numpy()])
		
	pred_states_mean = np.stack(pred_states_mean)
	terminal_states = np.stack(terminal_states)
		
	# make_gif(frames,'franka')
	if render:
		make_video(frames,'franka')

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




