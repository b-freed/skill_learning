from tokenize import ContStr
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Normal
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
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi

device = torch.device('cuda:0')


def mppi_update(skill_seq_mean, s0, model, cost_fn, batch_size, T, z_dim, plot, goal_state, lam=0.1, eps=0.2):

	eps_dist = Normal(torch.zeros((batch_size, z_dim),device=device),torch.ones((batch_size, z_dim),device=device)*eps)

	with torch.no_grad():
		skill_seq_mean[:-1] = skill_seq_mean[1:].clone()
		skill_seq_mean[-1].zero_()

		
		#skill_seq_mean = skill_seq_mean.tile([batch_size,1,1])
		s = s0.repeat(batch_size, 1)

		sk, dz, log_prob, pred_states = [], [], [], [s]
		eta = None
		gamma = 0.5
		for t in range(T):
			eps = eps_dist.sample()
			eta = eps
			# if eta is None:
			#     eta = eps
			# else:
			#     eta = gamma*eta + ((1-gamma**2)**0.5) * eps
			v = skill_seq_mean[t].expand_as(eta) + eta
			s,_ = model.decoder.abstract_dynamics(s,v)
			s_i = s
			rew = cost_fn(s)
			log_prob.append(eps_dist.log_prob(eta).sum(1))
			dz.append(eta)
			sk.append(rew.squeeze())
			pred_states.append(s_i)

		sk = torch.stack(sk)
		sk = torch.cumsum(sk.flip(0), 0).flip(0)
		log_prob = torch.stack(log_prob)

		sk = sk + lam*log_prob
		sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
		w = torch.exp(sk.div(lam)) + 1e-5
		w.div_(torch.sum(w, dim=1, keepdim=True))
		for t in range(T):
			skill_seq_mean[t] = skill_seq_mean[t] + torch.mv(dz[t].T, w[t])

		goal_state = torch.cat(batch_size * [goal_state],dim=0)
		pred_states = torch.stack(pred_states, dim=1)
		if plot:
			plt.figure()
			plt.scatter(s0[0].detach().cpu().numpy(),s0[1].detach().cpu().numpy(), label='init state')
			plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
			# plt.xlim([0,25])
			# plt.ylim([0,25])
			#pred_states = torch.cat(pred_states,1)

			plt.plot(pred_states[:,:,0].T.detach().cpu().numpy(),pred_states[:,:,1].T.detach().cpu().numpy())
			plt.legend()
				
			plt.savefig('pred_states_mppi')

		return skill_seq_mean
	
			
			
    
