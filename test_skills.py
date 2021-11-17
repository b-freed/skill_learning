import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import matplotlib.pyplot as plt
import ipdb
from pointmass_env import PointmassEnv


def run_skill(s0,skill,env,H):
	state = env.reset(x0=s0.flatten().detach().cpu().numpy())
	states = []
	
	for j in range(H):
	    action = ll_policy.numpy_policy(state,skill)
	    state = env.step(action)
	    
	    states.append(state)
	  
	return np.stack(states)



if __name__ == '__main__':

	device = torch.device('cuda:0')
	H = 100
	state_dim = 4
	a_dim = 2
	h_dim = 128
	z_dim = 20
	batch_size = 1
	epochs = 100000
	episodes = 10
	filename = 'log.pth'

	PATH = 'checkpoints/'+filename

	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])
	ll_policy = skill_model.decoder.ll_policy

	env = PointmassEnv()


	# start out in s0
	s0 = torch.zeros((batch_size,1,state_dim), device=device)
	skill_seq = torch.randn((1,episodes,z_dim), device=device)

	for i in range(episodes):
		# sample a skill
		skill = skill_seq[:,i:i+1,:]
		# predict outcome of that skill
		pred_next_s_mean, pred_next_s_sig = skill_model.abstract_dynamics(s0,skill)

		# run skill in real world
		state_seq = run_skill(s0,skill,env,H)
		ipdb.set_trace()

		plt.scatter(pred_next_s_mean[0,0,0].flatten().cpu().detach().numpy(),pred_next_s_mean[0,0,1].flatten().cpu().detach().numpy())
		plt.scatter(state_seq[:,0],state_seq[:,1])
		plt.show()
