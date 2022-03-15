from tokenize import ContStr
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


def mppi_update(self, skill_seq_mean, s0, model, cost_fn, T, N, lam=0.1, eps=0.2):
	#a = torch.zeros(T, model.a_dim).to(device)
	eps = Normal(torch.zeros(N, model.z_dim).to(device),
                            (torch.ones(N, model.z_dim)*eps).to(device))

	with torch.no_grad():
		skill_seq_mean[:-1] = skill_seq_mean[1:].clone()
		skill_seq_mean[-1].zero_()

		s0 = torch.FloatTensor(s0).unsqueeze(0).to(device)
		s = s0.repeat(N, 1)

		sk, dz, log_prob = [], [], []
		eta = None
		gamma = 0.5
	    	for t in range(T):
			eps = eps.sample()
			eta = eps
			# if eta is None:
			#     eta = eps
			# else:
			#     eta = gamma*eta + ((1-gamma**2)**0.5) * eps
			v = skill_seq_mean[t].expand_as(eta) + eta
			rew, s = cost_fn(s,v)
			log_prob.append(eps.log_prob(eta).sum(1))
			dz.append(eta)
			sk.append(rew.squeeze())

		sk = torch.stack(sk)
		sk = torch.cumsum(sk.flip(0), 0).flip(0)
		log_prob = torch.stack(log_prob)

		sk = sk + lam*log_prob
		sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
		w = torch.exp(sk.div(lam)) + 1e-5
		w.div_(torch.sum(w, dim=1, keepdim=True))
	    	for t in range(T):
			skill_seq_mean[t] = skill_seq_mean[t] + torch.mv(dz[t].T, w[t])
	
		
		return skill_seq_mean[0].cpu().clone().numpy()
	
			
			
    
