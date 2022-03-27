import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillPolicy
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


def run_policy(env,model,policy,goal_loc,n_skills,H,use_epsilon=True):

	goal_loc_np = goal_loc.squeeze().detach().cpu().numpy()
	s = torch.tensor(env.reset(),dtype=torch.float32,device=device)
	plt.figure()
	plt.scatter(goal_loc_np[0],goal_loc_np[1],s=300)
	for i in range(n_skills):
		skill = policy(s)
		if use_epsilon:
			skill_mean,skill_std = model.prior(s)# get the prior distribution over z for this state
			z = skill_mean + skill*skill_std
		
		s,skill_states,_,_ = run_skill(model,s,z,env,H)
		s = torch.tensor(s,dtype=torch.float32,device=device)

		plt.scatter(skill_states[:,0],skill_states[:,1])
		
		
		# plt.scatter(goals.flatten().detach().cpu().numpy()[0],goals.flatten().detach().cpu().numpy()[1])
		plt.axis('equal')
		plt.savefig('run_policy')

def run_policy_iterative_reopt(env,model,policy,goal_loc,n_skills,H,max_reopt_iters,batch_size,optimizer,n_iters,use_epsilon=True):

	
	goal_loc_np = goal_loc.squeeze().detach().cpu().numpy()
	s = torch.tensor(env.reset(),dtype=torch.float32,device=device)
	
	plt.scatter(goal_loc_np[0],goal_loc_np[1],s=300)
	for i in range(max_reopt_iters):
		skill = policy(s)
		if use_epsilon:
			skill_mean,skill_std = model.prior(s)# get the prior distribution over z for this state
			z = skill_mean + skill*skill_std
		
		s,skill_states,_,_ = run_skill(model,s,z,env,H,goal_loc)
		if np.sum((s[:2] - goal_loc_np)**2) < 1.0:
			return np.sum((s[:2] - goal_loc_np)**2)
		s = torch.tensor(s,dtype=torch.float32,device=device)

		plt.figure()
		plt.scatter(skill_states[:,0],skill_states[:,1])
		
		
		# plt.scatter(goals.flatten().detach().cpu().numpy()[0],goals.flatten().detach().cpu().numpy()[1])
		plt.axis('equal')
		plt.savefig('run_policy')

		# reoptimize policy for new starting loc
		# policy = SkillPolicy(state_dim,z_dim,h_dim).cuda()
		# optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=policy_l2_reg)

		policy = optimize_policy(torch.stack(batch_size * [s.reshape((1,-1))]),model,policy,optimizer,goal_loc,n_skills,n_iters,use_epsilon=True,temp=temp)

	return np.sum((s.detach().cpu().numpy()[:2] - goal_loc_np)**2)


def run_skill(skill_model,s0,skill,env,H,goal_loc,render=False):
	state = s0.flatten().detach().cpu().numpy()
	states = [state]
	goal_loc_np = goal_loc.squeeze().detach().cpu().numpy()
	actions = []
	frames = []
	for j in range(H): #H-1 if H!=1
	# for j in range(100):
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
		actions.append(action)
		state,_,_,_ = env.step(action)
		
		states.append(state)
		if np.sum((state[:2] - goal_loc_np)**2) < 1.0:
			return state, np.stack(states),np.stack(actions),frames
		
	  
	return state, np.stack(states),np.stack(actions),frames

def get_expected_cost(s0,model,policy,goal_loc,T,use_epsilon=True,plot=True,temp=1.0):

	s = s0
	costs = [torch.mean((s[:,:,:2] - goal_loc)**2,dim=-1)]
	states = [s]
	l2_costs = 0
	for i in range(T):
		# get high-level actions (skills)
		skill = policy(s)
		if use_epsilon:
			skill_mean,skill_std = model.prior(s)# get the prior distribution over z for this state
			z = skill_mean + skill*skill_std 

		else:
			z = skill

		l2_costs += torch.mean(skill**2)

		# sample state transitions
		s_next = model.decoder.abstract_dynamics.sample_from_sT_dist(s,z,temp=temp) 

		s = s_next
		states.append(s)

		cost_t = torch.mean((s_next[:,:,:2] - goal_loc)**2,dim=-1)  
		costs.append(cost_t) 
	

	costs = torch.stack(costs,dim=1)  # should be a batch_size x T 
	costs,_ = torch.min(costs,dim=1)

	if plot:
		plt.figure()
		plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
		plt.scatter(goal_loc[0,0,0].detach().cpu().numpy(),goal_loc[0,0,1].detach().cpu().numpy(),label='goal',s=300)
		# plt.xlim([0,25])
		# plt.ylim([0,25])
		states = torch.cat(states,1)
		
		plt.plot(states[:,:,0].T.detach().cpu().numpy(),states[:,:,1].T.detach().cpu().numpy())
			
		plt.savefig('pred_states_policy_opt')

	return torch.mean(costs),l2_costs 

def optimize_policy(s0,model,policy,optimizer,goal_loc,n_skills,n_iters,use_epsilon=True,temp=1.0):

	policy_losses = []
	for i in range(n_iters):
		print('i: ', i)
		# s0 = torch.stack(batch_size * [torch.tensor(env.reset(),dtype=torch.float32,device=device).reshape(1,-1)])
		policy_loss,l2_loss = get_expected_cost(s0,model,policy,goal_loc,n_skills,use_epsilon=use_epsilon,temp=temp)
		policy.zero_grad()
		loss = policy_loss + skill_l2_pen*l2_loss
		loss.backward()
		optimizer.step()

		policy_losses.append(policy_loss.item()) 
		print('policy loss: ', policy_loss.item())

	return policy


if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	# env = 'maze2d-large-v1'
	# env = 'antmaze-medium-diverse-v0'
	# env = 'kitchen-partial-v0'
	env = 'antmaze-large-diverse-v0'

	env = gym.make(env)
	data = env.get_dataset()
	
	H = 20
	# H = 40
	batch_size = 100
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	# batch_size = 1
	# episodes = 3
	state_dependent_prior = True
	state_dec_stop_grad = True
	encoder_type = 'state_action_sequence'
	state_decoder_type = 'autoregressive'
	beta = 0.1
	alpha = 1.0
	max_sig = None
	fixed_sig = 0.0
	a_dist = 'normal'
	ent_pen = 0
	colors = 3*['b','g','r','c','m','y','k']
	# n_skills = len(colors)
	n_skills = 50
	temp = 1.0
	n_iters = 20
	n_initial_iters = 20
	lr = 5e-5
	max_reopt_iters = 100
	policy_l2_reg = 10.0
	skill_l2_pen = 0.0


	filename = 'antmaze-large-diverse-v0_enc_type_state_action_sequencestate_dec_autoregressive_H_20_l2reg_0.0_a_1.0_b_0.1_sg_True_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
	PATH = 'checkpoints/'+filename

	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,state_decoder_type=state_decoder_type).cuda()

	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])


	policy = SkillPolicy(state_dim,z_dim,h_dim).cuda()

	optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=policy_l2_reg)

	goal_loc = torch.tensor(env.target_goal,dtype=torch.float32,device=device).reshape(1,1,-1)
	print('goal_loc.shape: ', goal_loc.shape)
	# policy_losses = []
	# for i in range(n_iters):
	# 	s0 = torch.stack(batch_size * [torch.tensor(env.reset(),dtype=torch.float32,device=device).reshape(1,-1)])
	# 	policy_loss = get_expected_cost(s0,model,policy,goal_loc,n_skills,use_epsilon=True,temp=temp)
	# 	policy.zero_grad()
	# 	policy_loss.backward()
	# 	optimizer.step()

	# 	policy_losses.append(policy_loss.item()) 
	# 	print('policy loss: ', policy_loss.item())

	

	# run_policy(env,model,policy,goal_loc,n_skills,H)
	dists_list = []
	for j in range(100):
		s0 = torch.stack(batch_size * [torch.tensor(env.reset(),dtype=torch.float32,device=device).reshape(1,-1)])
		policy = optimize_policy(s0,model,policy,optimizer,goal_loc,n_skills,n_initial_iters,use_epsilon=True,temp=temp)

		d = run_policy_iterative_reopt(env,model,policy,goal_loc,n_skills,H,max_reopt_iters,batch_size,optimizer,n_iters)
		dists_list.append(d)
		np.save('dists_policy_opt',dists_list)




