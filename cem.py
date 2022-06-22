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


def cem_iter(x,cost_fn,frac_keep,l2_pen):
    '''
    INPUTS:
        x: N x _ tensor of initial solution candidates
        cost_fn: function that returns cost scores in the form of an N-dim tensor
    OUTPUTS:
        x_mean: _-dimensional tensor of mean of updated solution candidate population
        x_std:  _-dimensional tensor of stand dev of updated solution candidate population
        cost_topk:  scalar mean cost of updated solution candidates
    '''
    N = x.shape[0]
    k = int(N*frac_keep) # k is for keep y'all
    
    # evaluate solution candidates, get sorted inds
    costs = cost_fn(x)
    #print(costs)
    l2_cost = l2_pen*torch.mean(torch.mean(x**2,dim=-1),dim=-1) 
    costs += l2_cost
    inds = torch.argsort(costs)
    # figure out which inds to keep
    inds_keep = inds[:k]
    # get best k solution candidates & their average cost
    x_topk = x[inds_keep,...]
    cost_topk = torch.mean(costs[inds_keep])
    # take mean and stand dev of new solution population
    x_mean = torch.mean(x_topk,dim=0)
    x_std  = torch.std( x_topk,dim=0)

    return x_mean,x_std,cost_topk

def cem(x_mean,x_std,cost_fn,pop_size,frac_keep,n_iters,l2_pen):
    count = 0
    for i in range(n_iters):
        count += 1
        x_shape = [pop_size]+list(x_mean.shape)
        x = x_mean + x_std*torch.randn(x_shape,device=device)
        x_mean,x_std,cost = cem_iter(x,cost_fn,frac_keep,l2_pen)

    return x_mean,x_std