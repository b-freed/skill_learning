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
from utils import make_gif, make_video, reparameterize

def run_skill(skill_model,s0,skill,env,H,render=False):
	state = s0.flatten().detach().cpu().numpy()
	states = [state]
	
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
		# if np.sum((state[:2] - pred_state[:2])**2) < .1:
		# 	break
	  
	return state, np.stack(states),np.stack(actions),frames


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
	n_samples = 100
	n_env_samples = 100
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
	fixed_sig = None
	a_dist = 'normal'
	ent_pen = 0
	colors = 3*['b','g','r','c','m','y','k']
	n_skills = len(colors)
	temp = 0.1

	filename = 'antmaze-large-diverse-v0_enc_type_state_action_sequencestate_dec_autoregressive_H_20_l2reg_0.0_a_1.0_b_0.1_sg_True_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
	PATH = 'checkpoints/'+filename

	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,state_decoder_type=state_decoder_type).cuda()

	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])

	state_np = env.reset()
	for i in range(n_skills):
		state = torch.reshape(torch.tensor(state_np,dtype=torch.float32).cuda(), (1,1,state_dim))
		z_mean,z_sig = model.prior(state)
		z = reparameterize(z_mean,temp*z_sig)

		state_tiled = torch.cat(n_samples*[state],dim=0)
		z_tiled     = torch.cat(n_samples*[z],    dim=0)

		print('state_tiled.shape: ', state_tiled.shape)
		print('z_tiled.shape: ', z_tiled.shape)
		
		sT_samples = model.decoder.abstract_dynamics.sample_from_sT_dist(state_tiled,z_tiled)

		sT_samples_np = sT_samples.squeeze().detach().cpu().numpy()
		print('sT_samples_np.shape: ', sT_samples_np.shape)

		# plt.figure()
		plt.scatter(sT_samples_np[:,0],sT_samples_np[:,1],c=colors[i],marker='x')
		plt.scatter(state_np[0],state_np[1],c=colors[i],marker='*')


		term_states = []
		for j in range(n_env_samples):
			# reset env to state_np
			env.set_state(state_np[:15],state_np[15:])
			# run skill
			sT,_,_,_ = run_skill(model,state,z,env,H,render=False)
			term_states.append(sT)
		
		term_states = np.stack(term_states)
		print('term_states.shape: ', term_states.shape)
		print('i: ', i)
		state_np = sT

		plt.scatter(term_states[:,0],term_states[:,1],c=colors[i])
		plt.savefig('unroll_autoregressive')







