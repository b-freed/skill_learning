

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.categorical as Categorical
import torch.distributions.mixture_same_family as MixtureSameFamily
import torch.distributions.kl as KL
import ipdb
import matplotlib.pyplot as plt
from utils import reparameterize
from skill_model import LowLevelPolicy
from skill_model import AbstractDynamics
from skill_model import Prior


class CategoricalPrior(nn.Module):
    '''
    Prior for discrete skill model
    Maps s0 to prior probabilities for each possible skill
    '''
    def __init__(self,s_dim,z_dim,h_dim):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(s_dim,h_dim),nn.ReLU(),
                                    nn.Linear(h_dim,h_dim),nn.ReLU(),
                                    nn.Linear(h_dim,z_dim),nn.Softmax(dim=-1))

    def forward(self,s0):
        '''
        INPUTS:
            s0: batch_size x 1 x s_dim  
        OUTPUTS:
            z_prior: batch_size x 1 x z_dim indicating prior probability of each possible z conditioned on s0
        '''

        return self.layers(state)


class DiscreteSkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,alpha=1.0,beta=1.0):
        super().__init__()

        self.ll_policy = LowLevelPolicy()
        self.abs_dyn = AbstractDynamics()
        self.prior = CategoricalPrior(s_dim,z_dim,h_dim)

    def forward(self,states,actions):

        
        z = torch.stack(batch_size*[torch.eye(self.z_dim,device=states.device)]).unsqueeze(2) # add dummy time dimension
        z_tiled = torch.cat(T*[z],dim=2)
        # tile s_0, s_T, states, and action, so that each particular state/action in batch_dim is matched up with every skill
        s_0_tiled     = torch.stack(self.z_dim*[s_0],    dim=1)
        s_T_tiled     = torch.stack(self.z_dim*[s_T],    dim=1)
        states_tiled  = torch.stack(self.z_dim*[states], dim=1)
        actions_tiled = torch.stack(self.z_dim*[actions],dim=1)  # batch_size x z_dim x T x a_dim
        # pass sT_tiled and z thru abstract dynamics to get predicted terminal state distribution for each skill
        sT_means,sT_sigs = self.abstract_dynamics(s_0_tiled,z)
        # pass thru low level policy to get action distribution for each possible skill
        action_means,action_sigs = self.ll_policy(states_tiled,z_tiled) # should be size batch_size x z_dim x T x a_dim
        
        return action_means,action_sigs,sT_means,sT_sigs,prior_means,prior_sigs

    def get_loss(self,states,actions):

        # reshape prior means and sigs to size batch_size x z_dim x 1 x 1

        # Get distributions for everything
        policy_dist = 
        prior_dist = 
        sT_dist = 

        # get unnormalized "mixture weights"

        # get eta

        # get log P(sT|s0,z)

        # get log eta

        # compute entire loss
        (w/eta)*(log_prob_sT - log_eta)







