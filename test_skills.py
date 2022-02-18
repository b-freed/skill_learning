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


def run_skill(skill_model,s0,skill,env,H,render):
	state = s0.flatten().detach().cpu().numpy()
	states = [state]
	
	actions = []
	frames = []
	# for j in range(H): #H-1 if H!=1
	for j in range(40):
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
		actions.append(action)
		state,_,_,_ = env.step(action)
		
		states.append(state)
	  
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
	
	env = 'antmaze-medium-diverse-v0'
	# env = 'kitchen-partial-v0'
	env = gym.make(env)
	data = env.get_dataset()
	
	H = 20
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	batch_size = 1
	episodes = 20
	wd = .001
	state_dependent_prior = True
	n_skills = 3
	colors = ['r','g','b']


	# if not state_dependent_prior:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
	# else:
	# 	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'Franka_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	# filename = 'Antmaze_H20_l2reg_0.001_stopgrad_log_best.pth'
	filename = 'AntMaze_H20_l2reg_0.001_a_10.0_b_1.0_log_best.pth'
	PATH = 'checkpoints/'+filename
	
	

	if not state_dependent_prior:
		skill_model_sdp = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	else:
		skill_model_sdp = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()
		
	checkpoint = torch.load(PATH)
	skill_model_sdp.load_state_dict(checkpoint['model_state_dict'])
	# ll_policy = skill_model_sdp.decoder.ll_policy


	# filename = 'z_dim_4.pth' #'z_dim_4.pth'#'log.pth'
	# PATH = 'checkpoints/'+filename
	# skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	# checkpoint = torch.load(PATH)
	# skill_model.load_state_dict(checkpoint['model_state_dict'])
	# ll_policy = skill_model.decoder.ll_policy


	# # start out in s0
	# s0 = torch.zeros((batch_size,1,state_dim), device=device)
	# skill_seq = torch.randn((1,episodes,z_dim), device=device)

	# for i in range(episodes):
	# 	# sample a skill
	# 	skill = skill_seq[:,i:i+1,:]
	# 	# predict outcome of that skill
	# 	pred_next_s_mean, pred_next_s_sig = skill_model.decoder.abstract_dynamics(s0,skill)

	# 	# run skill in real world
	# 	state_seq = run_skill(s0,skill,env,H)
	# 	print('pred_next_s_sig: ', pred_next_s_sig)
	# 	# circle = plt.Circle((pred_next_s_sig[0,0,0].item(),pred_next_s_sig[0,0,1].item()), 0), 0.2, color='r')

	# 	plt.scatter(pred_next_s_mean[0,0,0].flatten().cpu().detach().numpy(),pred_next_s_mean[0,0,1].flatten().cpu().detach().numpy())
	# 	plt.scatter(state_seq[:,0],state_seq[:,1])
	# 	plt.show()





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

	plt.figure()
	for j in range(n_skills):
		initial_state = env.reset()
		state = initial_state

		state = torch.reshape(torch.tensor(state,dtype=torch.float32).cuda(), (1,1,state_dim))
		#actions = torch.tensor(actions,dtype=torch.float32).cuda()

		if not state_dependent_prior:
			z_mean = torch.zeros((1,1,z_dim), device=device)
			z_sig = torch.ones((1,1,z_dim), device=device)
		else:
			z_mean,z_sig = skill_model_sdp.prior(state)

		z = skill_model_sdp.reparameterize(z_mean,z_sig)

		for i in range(episodes):
			initial_state = env.reset()
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
			sT_mean,sT_sig = skill_model_sdp.decoder.abstract_dynamics(state,z)
			#ipdb.set_trace()
			

		# 	# infer the skill
		# 	z_mean,z_sig = skill_model.encoder(states,actions)

		# 	z = skill_model.reparameterize(z_mean,z_sig)

		# 	# from the skill, predict the actions and terminal state
		# 	# sT_mean,sT_sig,a_mean,a_sig = skill_model.decoder(states,z)
		# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(states[:,0:1,:],z)
			

			states_actual,actions,skill_frames = run_skill(skill_model_sdp, state,z,env,H,render)
			state = states_actual[-1,:]
			terminal_states.append(state)
			mses.append(np.mean((state - sT_mean.flatten().detach().cpu().numpy())**2))
			state_dist = Normal.Normal(sT_mean, sT_sig )
			state_ll = torch.mean(state_dist.log_prob(torch.tensor(state,dtype=torch.float32,device=device).reshape(1,1,-1)))
			state_lls.append(state_ll.item())
			frames += skill_frames
			# states_actual,actions = run_skill_with_disturbance(skill_model_sdp, states[:,0:1,:],z,env,H)
			
			
			plt.scatter(states_actual[:,0],states_actual[:,1],c=colors[j])
			plt.scatter(states_actual[0,0],states_actual[0,1],c=colors[j])
			plt.errorbar(sT_mean[0,0,0].detach().cpu().numpy(),sT_mean[0,0,1].detach().cpu().numpy(),xerr=sT_sig[0,0,0].detach().cpu().numpy(),yerr=sT_sig[0,0,1].detach().cpu().numpy(),c=colors[j])

		actual_states.append(states_actual)
		action_dist.append(actions)
		pred_states_mean.append(sT_mean[0,0,:].detach().cpu().numpy())
		
	
	plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
	plt.title('Skill Execution & Prediction (Skill-Dependent Prior) '+str(i))
	plt.axis('square')
	# plt.savefig('Skill_Prediction_H'+str(H)+'_'+str(i)+'.png')
	plt.savefig('Skill_Prediction_H'+str(H)+'_'+'.png')

		
		
		
		
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




