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

def mppi_call(x,cost_fn,frac_keep,l2_pen):
    '''
    INPUTS:
        x: N x _ tensor of initial solution candidates
        cost_fn: function that returns cost scores in the form of an N-dim tensor
    OUTPUTS:
        x_mean: _-dimensional tensor of mean of updated solution candidate population
        x_std:  _-dimensional tensor of stand dev of updated solution candidate population
    '''
    lamda = 0.00001
    costs,z_arr,delta_z_arr = cost_fn(x)
    prior_costs = 0*torch.sum(torch.sum(x*x,dim=2),dim=1)#10*lamda*torch.sum(torch.sum(z_arr*delta_z_arr,dim=2),dim=0)
    costs += prior_costs
    min_cost = torch.min(costs)
    #print(torch.mean(costs),min_cost)
    costs = costs - min_cost
    eta = torch.sum(torch.exp(-costs/lamda))
    weights = torch.exp(-costs/lamda)/eta
    for i in range(x.shape[0]):
        x[i] = weights[i]*x[i]
    x = torch.sum(x,dim=0)
    #final_cost,_,_ = cost_fn(x.unsqueeze(0))

    return x#,final_cost[0]

def mppi(x_mean,x_std,cost_fn,pop_size,frac_keep,n_iters,l2_pen,variance=0.9):
    with torch.no_grad():
        x_shape = [pop_size]+list(x_mean.shape)
        x = x_mean + x_std*torch.randn(x_shape,device=device)
        x = mppi_call(x,cost_fn,frac_keep,l2_pen)
        # print('i: ',i)
        #if(i%100==0):
        #print('cost: ', cost)

    return x