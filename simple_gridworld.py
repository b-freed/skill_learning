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

class GenerativeModel(nn.Module):
    def __init__(self,dyn,policy,prior):
        super().__init__()
        self.dyn = dyn
        self.policy = policy
        self.prior = prior

    def forward(self):
        pass


class GridworldSkillModel(nn.Module):

    def __init__(self):
        super().__init__()

        policy = nn.Parameter(.5*torch.ones((2,2)))
        prior  = nn.Parameter(.5*torch.ones((2,1)))
        post   = nn.Parameter(.5*torch.ones((2,2,2)))
        dyn    = nn.Parameter(.5*torch.ones((2,2)))\
        
        self.gen_model = GenerativeModel(dyn,policy,prior)
        # self.post = Posterior(post)


    # def get_post_loss(self,data):

    def get_E_loss(self,states,actions):
        '''
        states:  N x 1 array of binary variables representing states
        actions: N x 1 array of binary variables representing actions
        '''
        zeros = torch.zeros_like(states)
        ones = torch.ones_like(states)
        z = torch.cat([zeros,ones],dim=0)

        states  = torch.cat(2*[states],dim=0)
        actions = torch.cat(2*[states],dim=0)  

        
        return torch.mean(self.post[z,a,s]*(-torch.log(self.policy[a,z]) \
                                       - torch.log(self.prior[z]) \
                                       + torch.log(torch.post(z,a,s))))


    def get_M_loss(self,states,actions):
        '''
        states:  N x 1 array of binary variables representing states
        actions: N x 1 array of binary variables representing actions
        '''
        zeros = torch.zeros_like(states)
        ones = torch.ones_like(states)
        z = torch.cat([zeros,ones],dim=0)

        states  = torch.cat(2*[states],dim=0)
        actions = torch.cat(2*[states],dim=0)  

        return torch.mean(self.post[z,s,a]*(-torch.log(self.dyn[z,z]) \
                                            -torch.log(self.policy[a,z] \
                                            -torch.log(self.prior[z]))))



def collect_data(N):
    actions = []
    states = []
    # sample an action
    a = torch.bernoulli(torch.tensor(.5)).type(torch.int)
    # get dist over next states given action from T (this will be probability next state is a 1)
    p_s_next = T[torch.tensor([1]),a]
    # sample state according to that prob
    s = torch.bernoulli(p_s_next).type(torch.int)

    actions.append(a)
    states.append(s)

    return torch.stack(states),torch.stack(actions)

def train(model,states,actions,E_optimizer,M_optimizer):
    E_losses = []
    M_losses = []

    E_loss = model.get_E_loss(states,actions)
    model.zero_grad()
    E_loss.backward()
    E_optimizer.step()


    M_loss = model.get_M_loss(states,actions)
    model.zero_grad()
    M_loss.backward()
    M_optimizer.step()

    return E_loss.item(), M_loss.item()


if __name__ == '__main__':

    device = torch.device('cuda:0')

    # P(s'=i|s,a=j) = P(s'=i|a=j) = T[i,j]
    T = torch.tensor([[.72,.25],[.25,.75]],device=device)
    N = 1000
    iters = 1000
    lr = 1e-4

    states,actions = collect_data(N)

    model = GridworldSkillModel()
    E_optimizer = torch.optim.Adam(model.post, lr=lr)
    M_optimizer = torch.optim.Adam(model.gen_model.parameters(), lr=lr)

    for i in range(iters):
        train(model,states,actions,E_optimizer,M_optimizer)


