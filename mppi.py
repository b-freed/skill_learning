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


def mppi_iter_variable_length(x,cost_fn,l2_pen,eps,samples,z):

		
		
		
		
		
		
		
		
	

def mppi(x_mean,x_std,cost_fn,pop_size,n_iters,samples=40,eps=0.2,max_ep=None):
	
	max_length = x_mean.shape[1]
	z = torch.zeros_like(x_mean).to(device)
	eps = Normal(torch.zeros(samples, x_mean.shape[0]).to(device),
                            (torch.ones(samples, x_mean.shape[0])*eps).to(device))

	
	with torch.no_grad():
		z[:-1] = z[1:].clone()
		z[-1].zero_()
		sk, da, log_prob = [], [], []
		eta = None
		gamma = 0.5
		for i in range(n_iters):
			eps = self.eps.sample()
			eta = eps
			v = z[t].expand_as(eta) + eta
			x = cost_fn(x_mean,v)
			log_prob.append(eps.log_prob(eta).sum(1))
			da.append(eta)
			
	
	return x_mean,x_std
	
	
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
		
    
