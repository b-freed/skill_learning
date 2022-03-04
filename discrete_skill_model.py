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
# from skill_model import LowLevelPolicy

class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist):

        super(LowLevelPolicy,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,a_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim),nn.Softplus())
        self.a_dist = a_dist
        self.a_dim = a_dim



    def forward(self,state,z):
        '''
        INPUTS:
            state: ... x state_dim tensor of states 
            z:     ... x z_dim tensor of states
        OUTPUTS:
            a_mean: ... x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  ... x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''

        # tile z along time axis so dimension matches state

        # Concat state and z_tiled
        state_z = torch.cat([state,z],dim=-1)
        # pass z and state through layers
        feats = self.layers(state_z)
        # get mean and stand dev of action distribution
        a_mean = self.mean_layer(feats)
        a_sig  = self.sig_layer(feats)

        return a_mean, a_sig

    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps

class CategoricalSkillDist(nn.Module):
    def __init__(self,s_dim,z_dim,h_dim):
        super(CategoricalSkillDist,self).__init__()

        self.layers = nn.Sequential(nn.Linear(s_dim,h_dim),nn.ReLU(),
                                    nn.Linear(h_dim,h_dim),nn.ReLU(),
                                    nn.Linear(h_dim,z_dim),nn.Softmax(dim=-1))

    def forward(self,state):
        return self.layers(state)

    # def sample_skill(state):
    #     p = self.forward(state)


class DiscreteSkillModel(nn.Module):
    def __init__(self,s_dim,a_dim,z_dim,h_dim,sT_weight):
        super(DiscreteSkillModel,self).__init__()

        self.skill_dist = CategoricalSkillDist(s_dim,z_dim,h_dim)
        self.abstract_dynamics = AbstractDynamics(s_dim,z_dim,h_dim)
        self.ll_policy = LowLevelPolicy(s_dim,a_dim,z_dim,h_dim, 'normal')
        self.z_dim = z_dim
        self.sT_weight = sT_weight
        self.state_dim = s_dim

    def get_losses(self,states,actions):
        
        batch_size,T,_ = states.shape
        s_0 = states[:,0:1,:]
        s_T = states[:,-1:,:]
        # Predict distribution over z.  This will be batch_size x z_dim.
        z_dist = self.skill_dist(s_0).squeeze()  
        # Create tensor of dim batch_size x z_dim x z_dim, with 1 hot vectors for each skill (so basically tile the identity matrix)
        z = torch.stack(batch_size*[torch.eye(self.z_dim,device=states.device)]).unsqueeze(2) # add dummy time dimension
        z_tiled = torch.cat(T*[z],dim=2)
        # tile s_0, s_T, states, and action, so that each particular state/action in batch_dim is matched up with every skill
        s_0_tiled     = torch.stack(self.z_dim*[s_0],    dim=1)
        s_T_tiled     = torch.stack(self.z_dim*[s_T],    dim=1)
        states_tiled  = torch.stack(self.z_dim*[states], dim=1)
        actions_tiled = torch.stack(self.z_dim*[actions],dim=1)  # batch_size x z_dim x T x a_dim
        # pass sT_tiled and z thru abstract dynamics to get predicted terminal state distribution for each skill
        s_T_means,s_T_sigs = self.abstract_dynamics(s_0_tiled,z)
        # pass thru low level policy to get action distribution for each possible skill
        action_means,action_sigs = self.ll_policy(states_tiled,z_tiled)
        # compute s_T_loss.  compute prob of sT given every possible skill, weight by skill probability, and sum.
        s_T_dist = Normal.Normal(s_T_means, s_T_sigs)
        s_T_probs = torch.clip(s_T_dist.log_prob(s_T_tiled).exp(),min=1e-15) # batch_size x z_dim x 1 x s_dim
        weighted_sum_s_T_probs = torch.sum(s_T_probs*z_dist.unsqueeze(-1).unsqueeze(-1),dim=1)  # add two dummy trailing dimensions.  Output should be batch_size x T x 1.
        s_T_loss = - torch.sum(torch.log(weighted_sum_s_T_probs))/(batch_size*T)
        #                                   x,w,sum_dim
        # s_T_logprobs = stable_weighted_log_sum_exp(s_T_dist.log_prob(s_T_tiled),z_dist.unsqueeze(-1).unsqueeze(-1),sum_dim=1)
        # s_T_loss = - torch.sum(s_T_logprobs)/(batch_size*T)
        # compute action loss. compute prob of actions given every possible skill, weight by skill probability, and sum.
        a_dist = Normal.Normal(action_means,action_sigs)
        a_probs = torch.clip(a_dist.log_prob(actions_tiled).exp(),min=1e-15)
        weighted_sum_a_probs = torch.sum(a_probs*z_dist.unsqueeze(-1).unsqueeze(-1),dim=1) # result should be batch_size x T x a_dim
        a_loss = - torch.sum(torch.log(weighted_sum_a_probs))/(batch_size*T)

        loss = a_loss + self.sT_weight*s_T_loss 

        return loss, s_T_loss, a_loss






        



