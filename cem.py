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
    # ipdb.set_trace()

    return x_mean,x_std,cost_topk

def cem(x_mean,x_std,cost_fn,pop_size,frac_keep,n_iters,l2_pen):

    for i in range(n_iters):
        x_shape = [pop_size]+list(x_mean.shape)
        x = x_mean + x_std*torch.randn(x_shape,device=device)
        x_mean,x_std,cost = cem_iter(x,cost_fn,frac_keep,l2_pen)
        # print('i: ',i)
        # print('cost: ', cost)


    return x_mean,x_std, cost



def cem_iter_variable_length(x,lengths,cost_fn,frac_keep,l2_pen):
    '''
    INPUTS:
        x: N x _ tensor of initial solution candidates
        lengths: N-dimensional tensor of plan lengths (how long we'll execute the corresponding plan for)
        cost_fn: function that returns cost scores in the form of an N-dim tensor

    OUTPUTS:
    '''
    N = x.shape[0]
    k = int(N*frac_keep)
    max_length = x.shape[1]

     # evaluate solution candidates, get sorted inds
    costs = cost_fn(x,lengths)
    l2_cost = l2_pen*torch.mean(torch.mean(x**2,dim=-1),dim=-1) 
    # print('l2_cost.shape: ', l2_cost.shape)
    costs += l2_cost
    inds = torch.argsort(costs)
    # figure out which inds to keep
    inds_keep = inds[:k]
    # get best k solution candidates & their average cost
    x_topk = x[inds_keep,...]
    lengths_topk = lengths[inds_keep]
    cost_topk = torch.mean(costs[inds_keep])
    cost = torch.mean(costs)
    # take mean and stand dev of new solution population
    x_mean = torch.mean(x_topk,dim=0)
    x_std  = torch.std( x_topk,dim=0)
    # estimate categorical distributin associated with lengths of top k plans
    p_lengths = get_length_probs(lengths_topk,max_length)
    # p_lengths += .001
    # p_lengths = p_lengths/torch.sum(p_lengths)

    return x_mean,x_std,p_lengths,cost#cost_topk

def cem_variable_length(x_mean,x_std,p_lengths,cost_fn,pop_size,frac_keep,n_iters,max_ep=None,l2_pen=0):

    max_length = x_mean.shape[1]
    # ipdb.set_trace()

    for i in range(n_iters):
        x_shape = [pop_size]+list(x_mean.shape)
        epsilon = torch.randn(x_shape,device=device)
        if max_ep is not None:
            epsilon[epsilon >  max_ep]  =  max_ep
            epsilon[epsilon < -max_ep] = -max_ep
        x = x_mean + x_std*epsilon
        lengths = torch.multinomial(p_lengths, pop_size,replacement=True)# sample lengths from p_lengths
        x_mean,x_std,p_lengths,cost = cem_iter_variable_length(x,lengths,cost_fn,frac_keep,l2_pen)
        # ipdb.set_trace()

        # p_lengths += 1/len(p_lengths)
        # p_lengths = p_lengths/torch.sum(p_lengths)
       
        # print('i: ',i)
        # print('cost: ', cost)
    
    length = torch.argmax(p_lengths)
    
    
    # x_mean,x_std = x_mean[:length+1],x_std[:length+1]
    x_mean,x_std = x_mean[:length],x_std[:length]
    # ipdb.set_trace()
    return x_mean,x_std


def get_length_probs(lengths,max_length):
    counts = torch.stack([torch.sum(lengths == i) for i in range(max_length+1)])
    p_lengths = counts / torch.sum(counts)

    return p_lengths




    


