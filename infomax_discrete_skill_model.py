import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.categorical as Categorical
import torch.distributions.kl as KL
import ipdb
import matplotlib.pyplot as plt
from utils import reparameterize

from skill_model import AbstractDynamics
from utils import stable_weighted_log_sum_exp




class InfoMaxDiscreteSkillModel(nn.Module):
    def __init__(self,s_dim,a_dim,n_skills,h_dim):
        super().__init__()

        self.policies = # one low-level policy for each skill
        self.dynamics_models = # one abstract dynamics model for each skill

    def get_losses(self,states,actions):
        s0 = states[:,0:1,:]
        prior_logits = self.prior(s0)
        
        a_means,a_sigs = self.call_policies()  # should give us an n_skills x batch_size x T x a_dim tensor
        
        a_dist = Normal.Normal(a_means,a_sigs)
        a_loss_per_skill =   a_dist.log_prob(actions).sum(-1).mean(-1) # get losses for each skill.  Should be n_skills x batch_size

        # should get vector of losses that's batch_size, which we'll take mean of 
        # and should get vector of indecies that's batch_size, which indicates skill for each element in batch.

        # a_loss = torch.min(a_loss_per_skill)
        # skill_ind = torch.argmin(a_loss_per_skill)

        # get abstract dynamics and prior predictions
        sT_mean,sT_sig = self.call_dynamics_models(s0,skill_ind)
        sT_dist  = Normal.Normal(sT_mean,sT_sig)
        sT_loss = sT_dist.log_prob(sT).sum(-1).mean()

        self.prior_loss = -
        

