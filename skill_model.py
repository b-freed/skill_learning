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
import utils
import os
import matplotlib.pyplot as plt
from utils import reparameterize

class AbstractDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim):

        super(AbstractDynamics,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,state_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,state_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())

    def forward(self,s0,z):

        '''
        INPUTS:
            s0: batch_size x 1 x state_dim initial state (first state in execution of skill)
            z:  batch_size x 1 x z_dim "skill"/z
        OUTPUTS: 
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
        '''

        # concatenate s0 and z
        s0_z = torch.cat([s0,z],dim=-1)

        # pass s0_z through layers
        feats = self.layers(s0_z)

        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)

        return sT_mean,sT_sig

class AutoregressiveStateDecoder(nn.Module):
    def __init__(self,state_dim,z_dim,h_dim):
        super(AutoregressiveStateDecoder,self).__init__()

        self.state_dim = state_dim
        
        self.emb_layer  = nn.Sequential(nn.Linear(3*state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU()) 
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,1))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,1),nn.Softplus())


        

    def forward(self,sT,s0,z):
        '''
        returns the log probability of the terminal state sT given the initial state s0 and skill z

        INPUTS:
            sT: batch_size x 1 x state_dim terminal state
            s0: batch_size x 1 x state_dim initial state
            z:  batch_size x 1 x z_dim skill vector

        OUTPUTS:
            log_probs: batch_size x 1 x state_dim tensor of log probabilities for each element
        '''
        batch_size = s0.shape[0]

        # tile s0, z, and sT along 1st dimension, state_dim times
        s0_tiled = torch.cat(self.state_dim*[s0],dim=1) # should be batch_size x state_dim x state_dim
        z_tiled  = torch.cat(self.state_dim*[z], dim=1) # should be batch_size x state_dim x z_dim
        sT_tiled = torch.cat(self.state_dim*[sT],dim=1) # should be batch_size x state_dim x state_dim

        # Generate one hot vectors using identity 
        onehots = torch.stack(batch_size*[torch.eye(self.state_dim,device=os.environ.get('_DEVICE', 'cpu'))],dim=3) # batch_size x state_dim x state_dim

        # Mask out future elements of sT (so element-wise multiplication with a matrix with zeros on and below diagonal)
        mask = torch.tril(torch.ones((self.state_dim,self.state_dim),device=os.environ.get('_DEVICE', 'cpu')),diagonal=-1)

        sT_tiled_masked = sT_tiled * mask

        # Concatentate
        inp = torch.cat([s0_tiled,sT_tiled_masked,onehots,z_tiled],dim=-1)

        # pass thru layers to get mean and sig
        emb = self.emb_layer(inp)
        sT_mean, sT_sig = self.mean_layer(emb),self.sig_layer(emb) # should be batch_size x state_dim x 1
        
        # get rid of last singleton dimension, add one back in for the 1st dimension
        sT_mean = sT_mean.squeeze().unsqueeze(1)
        sT_sig  = sT_sig.squeeze().unsqueeze(1)

        return sT_mean, sT_sig

    def sample_from_sT_dist(self,s0,z,temp=1.0):
        '''
        Given an initial state s0 and skill z, sample from predicted terminal state distribution

        INPUTS:
            s0: batch_size x 1 x state_dim
            z:  batch_size x 1 x z_dim

        OUTPUTS:
            sT: batch_size x 1 x state_dim
        '''
        # if self.state_decoder_type is not 'autoregressive':
        #     raise NotImplementedError

        # tile s0, z, and sT along 1st dimension, state_dim times
        # s0_tiled = torch.cat(self.state_dim*[s0],dim=1) # should be batch_size x state_dim x state_dim
        # z_tiled  = torch.cat(self.state_dim*[z], dim=1) # should be batch_size x state_dim x z_dim
        # sT_tiled = torch.cat(self.state_dim*[sT],dim=1) # should be batch_size x state_dim x state_dim

        batch_size = s0.shape[0]
        sT_lessthan_i = torch.zeros((batch_size,1,self.state_dim),device=os.environ.get('_DEVICE', 'cpu'))
        for i in range(self.state_dim):
            onehot = self.get_onehot(i,batch_size)
            # concatenate s0, sT[<i], onehot, z
            inp = torch.cat([s0,sT_lessthan_i,onehot,z],dim=-1)
            emb = self.emb_layer(inp)
            sT_i_mean, sT_i_sig = self.mean_layer(emb),self.sig_layer(emb) # should be batch_size x state_dim x 1
            
            sT_i = reparameterize(sT_i_mean,temp*sT_i_sig)
            sT_lessthan_i[:,:,i:i+1] = sT_i

        return sT_lessthan_i

    def get_onehot(self,i,batch_size):

        onehot = torch.zeros((batch_size,1,self.state_dim),device=os.environ.get('_DEVICE', 'cpu'))
        onehot[:,:,i] = 1

        return onehot
    




# class DeadReckonStateDecoder(nn.Module):

#     def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):

#         self.state_dim = state_dim # state dimension
#         self.a_dim = a_dim # action dimension

#         self.s_emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
#         self.a_emb_layer  = nn.Sequential(nn.Linear(a_dim,    h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
#         self.rnn        = nn.GRU(2*h_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
#         #self.mean_layer = nn.Linear(h_dim,z_dim)
#         self.mean_layer = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
#         #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
#         self.sig_layer  = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())

#     def forward(self,s0,actions):

#         s0_emb = self.s_emb_layer(s0)
#         s0_emb_tiled = 
#         a_emb = self.a_emb_layer(actions)
#         emb = torch.cat([s0_emb_tiled,a_emb],dim=-1)
#         feats,_ = self.rnn(s_emb_a)
#         hn = feats[:,-1:,:]
#         # hn = hn.transpose(0,1) # use final hidden state, as this should be an encoding of all states and actions previously.
#         # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
#         sT_mean = self.mean_layer(hn)
#         sT_sig = self.sig_layer(hn)
        
#         return sT_mean, sT_sig





class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,max_sig=None,fixed_sig=None):

        super(LowLevelPolicy,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,a_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        self.a_dist = a_dist
        self.a_dim = a_dim
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig



    def forward(self,state,z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''

        # tile z along time axis so dimension matches state
        z_tiled = z.tile([1,state.shape[-2],1]) #not sure about this 

        # Concat state and z_tiled
        state_z = torch.cat([state,z_tiled],dim=-1)
        # pass z and state through layers
        feats = self.layers(state_z)
        # get mean and stand dev of action distribution
        a_mean = self.mean_layer(feats)
        if self.max_sig is None:
            a_sig  = nn.Softplus()(self.sig_layer(feats))
        else:
            a_sig = self.max_sig * nn.Sigmoid()(self.sig_layer(feats))

        if self.fixed_sig is not None:
            a_sig = self.fixed_sig*torch.ones_like(a_sig)

        return a_mean, a_sig
    
    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=os.environ.get('_DEVICE', 'cpu'),dtype=torch.float32),(1,1,-1))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps

class LowLevelPolicyWithTermination(nn.Module):
    r'''        
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    \beta(b_t|s_t,z) is our "skill termination model", it spits 0 and 1 corresponding to continuing and terminating a skill respectively.
    Low-level policy and terminal model share the same encoding model that converts s_t and z into a hidden state. 
    See Encoder and Decoder for more description
    '''
    def __init__(self, state_dim, a_dim, z_dim, h_dim, a_dist, max_sig=None, fixed_sig=None, termination_threshold=0.5):

        super(LowLevelPolicyWithTermination,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())

        self.mean_action_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        self.sig_action_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))

        self.mean_termination_layer = nn.Sequential(nn.Linear(h_dim,h_dim), nn.ReLU(), nn.Linear(h_dim, 1), nn.Sigmoid())
        self.sig_termination_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim, 1))

        self.a_dist = a_dist
        self.a_dim = a_dim
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig
        self.termination_threshold = termination_threshold

    def forward(self,state,z):
        '''
        INPUTS:
            state: batch_size x t x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x t x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x t x a_dim tensor of action standard devs for each t in {0.,,,.T}
            b_mean: batch_size x t x 1 tensor of termination means for each t in {0.,,,.t}
            b_sig:  batch_size x t x 1 tensor of termination standard devs for each t in {0.,,,.t}
        '''

        # tile z along time axis so dimension matches state
        z_tiled = z.tile([1,state.shape[-2],1]) #not sure about this 

        # Concat state and z_tiled
        state_z = torch.cat([state,z_tiled],dim=-1)
        
        # pass z and state through layers
        feats = self.layers(state_z)
        
        # get mean and stand dev of action distribution
        a_mean = self.mean_action_layer(feats)
        
        b_mean = (self.mean_termination_layer(feats) > self.termination_threshold)
        b_sig = self.mean_termination_layer(feats)
        
        if self.max_sig is None:
            a_sig  = nn.Softplus()(self.sig_action_layer(feats))
        else:
            a_sig = self.max_sig * nn.Sigmoid()(self.sig_action_layer(feats))

        if self.fixed_sig is not None:
            a_sig = self.fixed_sig*torch.ones_like(a_sig)

        return a_mean, a_sig, b_mean, b_sig
    
    def numpy_policy(self, state, z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=os.environ.get('_DEVICE', 'cpu'),dtype=torch.float32),(1,1,-1))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps

class LowLevelDynamicsFF(nn.Module):

    def __init__(self,s_dim,a_dim,h_dim):

        super(LowLevelDynamicsFF,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(s_dim+a_dim,h_dim),nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,s_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,s_dim),nn.Softplus())

       


    def forward(self,states,actions):
        '''
        INPUTS:
            states:  batch_size x T x state_dim tensor of states 
            actions: batch_size x T x z_dim tensor of actions
        OUTPUTS:
            s_next_mean: batch_size x T x s_dim tensor of means for predicted next states
            s_next_sig:  batch_size x T x a_dim tensor of standard devs for predicted next states
        '''

        state_actions = torch.cat([states,actions],dim=-1)
        feats = self.layer1(state_actions)
        delta_means = self.mean_layer(feats)
        s_next_mean = delta_means + states
        s_next_sig = self.sig_layer(feats)



        return s_next_mean, s_next_sig


class LowLevelDynamics(nn.Module):

    def __init__(self,s_dim,a_dim,h_dim):

        super(LowLevelDynamics,self).__init__()
        
        self.emb_layer = nn.Sequential(nn.Linear(s_dim+a_dim,h_dim),nn.ReLU())
        self.rnn = nn.GRU(input_size=h_dim,hidden_size=h_dim,batch_first=True)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,s_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,s_dim),nn.Softplus())

       


    def forward(self,states,actions,h=None):
        '''
        INPUTS:
            states:  batch_size x T x state_dim tensor of states 
            actions: batch_size x T x z_dim tensor of actions
        OUTPUTS:
            s_next_mean: batch_size x T x s_dim tensor of means for predicted next states
            s_next_sig:  batch_size x T x a_dim tensor of standard devs for predicted next states
        '''

        state_acitons = torch.cat([states,actions],dim=-1)
        sa_emb = self.emb_layer(state_acitons)
        if h is not None:
            feat,h = self.rnn(sa_emb,h)
        else:
            feat,h = self.rnn(sa_emb)

        delta_means = self.mean_layer(feat)
        s_next_mean = delta_means + states
        s_next_sig = self.sig_layer(feat)



        return s_next_mean, s_next_sig, h


class Encoder(nn.Module):
    '''
    Encoder module.
    We can try the following architecture initially:
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):
        super(Encoder, self).__init__()


        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.rnn        = nn.GRU(h_dim+a_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.sig_layer  = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())


    def forward(self,states,actions):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''

        
        s_emb = self.emb_layer(states)
        # through rnn
        s_emb_a = torch.cat([s_emb,actions],dim=-1)
        feats,_ = self.rnn(s_emb_a)
        hn = feats[:,-1:,:]
        # hn = hn.transpose(0,1) # use final hidden state, as this should be an encoding of all states and actions previously.
        # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
        z_mean = self.mean_layer(hn)
        z_sig = self.sig_layer(hn)
        
        return z_mean, z_sig


class EncoderAutoTermination(nn.Module):
    '''
    Encoder module.
    We can try the following architecture initially:
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):
        super(EncoderAutoTermination, self).__init__()


        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.rnn        = nn.GRU(h_dim+a_dim+z_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.z_mean_layer = nn.Sequential(nn.Linear(2*h_dim, h_dim),nn.ReLU(),nn.Linear(h_dim, z_dim))
        self.b_mean_layer = nn.Sequential(nn.Linear(2*h_dim, h_dim),nn.ReLU(),nn.Linear(h_dim, 2))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.z_sig_layer  = nn.Sequential(nn.Linear(2*h_dim, h_dim),nn.ReLU(),nn.Linear(h_dim, z_dim),nn.Softplus())
        self.b_sig_layer  = nn.Sequential(nn.Linear(2*h_dim, h_dim),nn.ReLU(),nn.Linear(h_dim, 2),nn.Softplus())


    def forward(self, states, actions, skills):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states:  batch_size x t x state_dim state sequence tensor
            actions: batch_size x t x a_dim action sequence tensor
            skills:  batch_size x t x skill_dim skill sequence tensor
        OUTPUTS:
            z_mean:  batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:   batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''

        
        s_emb = self.emb_layer(states)
        # through rnn
        s_emb_a = torch.cat([s_emb, actions, skills], dim=-1)
        feats, _ = self.rnn(s_emb_a)
        hn = feats[:, -1:, :]
        # hn = hn.transpose(0,1) # use final hidden state, as this should be an encoding of all states and actions previously.
        # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
        z_mean = self.z_mean_layer(hn)
        z_sig = self.z_sig_layer(hn)

        b_mean = self.b_mean_layer(hn)
        b_sig = self.b_sig_layer(hn)
        
        return z_mean, z_sig, b_mean, b_sig


class StateSeqEncoder(nn.Module):

    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):
        super(StateSeqEncoder, self).__init__()

        print('BUILDING STATE SEQ ENCODER!!!')
        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.rnn        = nn.GRU(h_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.sig_layer  = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())


    def forward(self,states,actions):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor (we don't use these, just to keep consistent with other encoders)
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''

        
        s_emb = self.emb_layer(states)
        # through rnn
        feats,_ = self.rnn(s_emb)
        hn = feats[:,-1:,:]
        # hn = hn.transpose(0,1) # use final hidden state, as this should be an encoding of all states and actions previously.
        # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
        z_mean = self.mean_layer(hn)
        z_sig = self.sig_layer(hn)
        
        return z_mean, z_sig


class S0STEncoder(nn.Module):
    '''
    Encoder module.
    Instead of taking entire trajectory, this encoder only takes s0 and sT.
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):
        super(S0STEncoder, self).__init__()


        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        # self.emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        # self.rnn        = nn.GRU(h_dim+a_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
        self.s0_emb_layer = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.sT_emb_layer = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.sig_layer  = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())


    def forward(self,states,actions):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''

        
        # s_emb = self.emb_layer(states)
        # # through rnn
        # s_emb_a = torch.cat([s_emb,actions],dim=-1)
        # feats,_ = self.rnn(s_emb_a)
        # hn = feats[:,-1:,:]
        # # hn = hn.transpose(0,1) # use final hidden state, as this should be an encoding of all states and actions previously.
        # # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
        # z_mean = self.mean_layer(hn)
        # z_sig = self.sig_layer(hn)


        s0 = states[:,:1,:]
        s0_emb = self.s0_emb_layer(s0)
        sT = states[:,-1:,:]
        sT_emb = self.sT_emb_layer(sT)
        s0_sT_emb = torch.cat([s0_emb,sT_emb],dim=-1)
        z_mean = self.mean_layer(s0_sT_emb)
        z_sig = self.sig_layer(s0_sT_emb)
        
        return z_mean, z_sig

        

class Decoder(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist,state_dec_stop_grad, max_sig, fixed_sig, state_decoder_type):

        super(Decoder,self).__init__()
        
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        if state_decoder_type == 'mlp':
            self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim)
        elif state_decoder_type == 'autoregressive':
            self.abstract_dynamics = AutoregressiveStateDecoder(state_dim,z_dim,h_dim)
        else:
            print('PICK VALID STATE DECODER TYPE!!!')
            assert False
        # self.ll_policy = LowLevelPolicyWithTermination(state_dim,a_dim,z_dim,h_dim, a_dist, max_sig = max_sig, fixed_sig=fixed_sig)
        self.ll_policy = LowLevelPolicy(state_dim,a_dim,z_dim,h_dim, a_dist, max_sig = max_sig, fixed_sig=fixed_sig)
        self.emb_layer  = nn.Linear(state_dim+z_dim,h_dim)
        self.fc = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())

        self.state_dec_stop_grad = state_dec_stop_grad

        self.state_decoder_type = state_decoder_type
        
    def forward(self,states,z):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            z:      batch_size x 1 x z_dim sampled z/skill variable
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        
        s_0 = states[:,0:1,:]
        s_T = states[:,-1:,:]
        a_mean,a_sig   = self.ll_policy(states,z)
        # a_mean, a_sig, b_mean, b_sig = self.ll_policy(states,z)

        if self.state_dec_stop_grad:
            z = z.detach()
        
        if self.state_decoder_type == 'autoregressive':
            z_detached = z.detach()
            sT_mean,sT_sig = self.abstract_dynamics(s_T,s_0,z)
        elif self.state_decoder_type == 'mlp':
            sT_mean,sT_sig = self.abstract_dynamics(s_0,z)
        else:
            print('PICK VALID STATE DECODER TYPE!!!')
            assert False
        
        return sT_mean, sT_sig, a_mean, a_sig


class Prior(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,z_dim,h_dim):

        super(Prior,self).__init__()
        
        self.state_dim = state_dim
        self.z_dim = z_dim

        self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())
        
    def forward(self,s0):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            
        OUTPUTS:
            z_mean: batch_size x 1 x state_dim tensor of z means
            z_sig:  batch_size x 1 x state_dim tensor of z standard devs
            
        '''
        feats = self.layers(s0)
        # get mean and stand dev of action distribution
        z_mean = self.mean_layer(feats)
        z_sig  = self.sig_layer(feats)

        return z_mean, z_sig

class TerminalStateDependentPrior(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,z_dim,h_dim):

        super(TerminalStateDependentPrior,self).__init__()

        print('INITING TERMINAL STATE DEP')
        
        self.state_dim = state_dim
        self.z_dim = z_dim

        self.s0_emb_layer = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU())
        self.sT_emb_layer = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU())

        self.layers = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())
        
    def forward(self,s0,sT):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            
        OUTPUTS:
            z_mean: batch_size x 1 x state_dim tensor of z means
            z_sig:  batch_size x 1 x state_dim tensor of z standard devs
            
        '''
        # feats = self.layers(s0)
        s0_emb = self.s0_emb_layer(s0)
        sT_emb = self.sT_emb_layer(s0)
        s0_sT_emb = torch.cat([s0_emb,sT_emb],dim=-1)
        feats = self.layers(s0_sT_emb)
        # get mean and stand dev of action distribution
        z_mean = self.mean_layer(feats)
        z_sig  = self.sig_layer(feats)

        return z_mean, z_sig
        


class SkillModelStateDependentPrior(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist='normal',state_dec_stop_grad=False,beta=1.0,alpha=1.0,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type='mlp'):
        super(SkillModelStateDependentPrior, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        
        if encoder_type == 'state_action_sequence':
            self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        elif encoder_type == 's0sT':
            self.encoder = S0STEncoder(state_dim,a_dim,z_dim,h_dim)
        elif encoder_type == 'state_sequence':
            self.encoder = StateSeqEncoder(state_dim,a_dim,z_dim,h_dim)
        else:
            print('INVALID ENCODER TYPE!!!!')
            assert False

        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim, a_dist, state_dec_stop_grad,max_sig=max_sig,fixed_sig=fixed_sig,state_decoder_type=state_decoder_type)
        self.prior   = Prior(state_dim,z_dim,h_dim)
        self.beta    = beta
        self.alpha   = alpha
        self.ent_pen = ent_pen

        if ent_pen != 0:
            assert not state_dec_stop_grad

    def forward(self,states,actions):
        
        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution 
        '''

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means,z_post_sigs = self.encoder(states,actions)
        # STEP 2. sample z from posterior 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        # STEP 3. Pass z_sampled and states through decoder 
        s_T_mean, s_T_sig, a_means, a_sigs = self.decoder(states,z_sampled)



        return s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs

    def get_losses(self,states,actions):
        '''
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)] 
          + E_q [sum_t=0^T P(a_t|s_t,z)] 
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
        Distributions we need:
        '''


        s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)

        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        if self.decoder.ll_policy.a_dist == 'normal':
            a_dist = Normal.Normal(a_means, a_sigs)
        elif self.decoder.ll_policy.a_dist == 'tanh_normal':
            base_dist = Normal.Normal(a_means, a_sigs)
            transform = torch.distributions.transforms.TanhTransform()
            a_dist = TransformedDistribution(base_dist, [transform])
        else:
            assert False
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        # z_prior_means = torch.zeros_like(z_post_means)
        # z_prior_sigs = torch.ones_like(z_post_sigs)
        z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:]) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        T = states.shape[1]
        s_T = states[:,-1:,:]  
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),   dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) 
        s_T_ent  = torch.mean(torch.sum(s_T_dist.entropy(),       dim=-1))/T
        # print('a_sigs: ', a_sigs)
        # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
        # loss term correpsonding ot kl loss between posterior and prior
        # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss + self.ent_pen*s_T_ent
        # loss_tot = s_T_loss + kl_loss

        return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent
    
    
        
    
    def get_expected_cost(self, s0, skill_seq, goal_states):
        '''
        s0 is initial state  # batch_size x 1 x s_dim
        skill sequence is a 1 x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_states = torch.cat(batch_size * [goal_states],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        for i in range(skill_seq_len):
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            mu_z, sigma_z = self.prior(s_i)
          

            z_i = mu_z + sigma_z*torch.cat(batch_size*[skill_seq[:,i:i+1,:]],dim=0)
            # converting z_i from 1x1xz_dim to batch_size x 1 x z_dim
            # z_i = torch.cat(batch_size*[z_i],dim=0) # feel free to change this to tile
            # use abstract dynamics model to predict mean and variance of state after executing z_i, conditioned on s_i
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled
            
            pred_states.append(s_i)
        
        #compute cost for sequence of states/skills
        # print('predicted final loc: ', s_i[:,:,:2])
        s_term = s_i
        cost = torch.mean((s_term[:,:,:2] - goal_states[:,:,:2])**2)
        
        
        return cost, torch.cat(pred_states,dim=1)

    def get_expected_cost_for_cem(self, s0, skill_seq, goal_state, use_epsilons=True, plot=True, length_cost=0):
        '''
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = [torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()]
        # costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        for i in range(skill_seq_len):
            
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i)
                
                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]
            
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze() + (i+1)*length_cost
            costs.append(cost_i)
            
            pred_states.append(s_i)
        
        costs = torch.stack(costs,dim=1)  # should be a batch_size x T or batch_size x T 
        costs,_ = torch.min(costs,dim=1)  # should be of size batch_size
        # print('costs: ', costs)
        # print('costs.shape: ', costs.shape)

        if plot:
            plt.figure()
            plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
            plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
            # plt.xlim([0,25])
            # plt.ylim([0,25])
            pred_states = torch.cat(pred_states,1)
            
            plt.plot(pred_states[:,:,0].T.detach().cpu().numpy(),pred_states[:,:,1].T.detach().cpu().numpy())
                
            plt.savefig('pred_states_cem')


            # plt.figure()
            # plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
            # plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
            # # plt.xlim([0,25])
            # # plt.ylim([0,25])
            # # pred_states = torch.cat(pred_states,1)
            # for i in range(batch_size):
            #     # ipdb.set_trace()
            #     plt.plot(pred_states[i,:,0].detach().cpu().numpy(),pred_states[i,:,1].detach().cpu().numpy())
                
            # plt.savefig('pred_states_cem_variable_length_FULL_SEQ')
        
        
        return costs
    
    def get_expected_cost_variable_length(self, s0, skill_seq, lengths, goal_state, use_epsilons=True, plot=False):
        '''
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        for i in range(skill_seq_len):
           
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i)
                
                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]
            
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = (lengths == i+1)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
            costs += cost_i
            
            pred_states.append(s_i)
        
        if plot:
            plt.figure()
            plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
            plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
            # plt.xlim([0,25])
            # plt.ylim([0,25])
            pred_states = torch.cat(pred_states,1)
            for i in range(batch_size):
                # ipdb.set_trace()
                plt.plot(pred_states[i,:lengths[i].item()+1,0].detach().cpu().numpy(),pred_states[i,:lengths[i].item()+1,1].detach().cpu().numpy())
                
            plt.savefig('pred_states_cem_variable_length')


            plt.figure()
            plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
            plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
            # plt.xlim([0,25])
            # plt.ylim([0,25])
            # pred_states = torch.cat(pred_states,1)
            for i in range(batch_size):
                # ipdb.set_trace()
                plt.plot(pred_states[i,:,0].detach().cpu().numpy(),pred_states[i,:,1].detach().cpu().numpy())
                
            plt.savefig('pred_states_cem_variable_length_FULL_SEQ')
        return costs
    
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps




class SkillModelStateDependentPriorAutoTermination(SkillModelStateDependentPrior):
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist='normal',state_dec_stop_grad=False,beta=1.0,alpha=1.0,gamma=1.0,temperature=1.0,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type='mlp', min_skill_len=None, max_skill_len=None, max_skills_per_seq=None):
        super(SkillModelStateDependentPriorAutoTermination, self).__init__(state_dim,a_dim,z_dim,h_dim,a_dist,state_dec_stop_grad,beta,alpha,max_sig,fixed_sig,ent_pen,encoder_type,state_decoder_type)
        if encoder_type == 'state_action_sequence':
            self.encoder = EncoderAutoTermination(state_dim,a_dim,z_dim,h_dim)
        else: 
            raise NotImplementedError
        self.gamma = gamma # skill length penalty
        self.temperature = temperature # gumbel-softmax temperature
        self.min_skill_len = min_skill_len
        self.max_skill_len = max_skill_len
        self.max_skills_per_seq = max_skills_per_seq

    def forward(self, states, actions, skills):
        
        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states:       batch_size x t x state_dim sequence tensor
            actions:      batch_size x t x a_dim action sequence tensor
            skills:       batch_size x t x skill_dim sequence tensor
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution 
        '''

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means, z_post_sigs, b_post_means, b_post_sigs = self.encoder(states, actions, skills)
        # STEP 2. sample z from posterior 
        z_sampled = self.reparameterize(z_post_means, z_post_sigs)
        # b_sampled = self.reparameterize(b_post_means, b_post_sigs)

        # STEP 3. Pass z_sampled and states through decoder 
        s_T_mean, s_T_sig, a_means, a_sigs = self.decoder(states, z_sampled)

        return s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs, b_post_means, b_post_sigs

    def compute_losses(self, s_T_mean, s_T_sig, a_means, a_sigs, 
                        z_post_means, z_post_sigs, s_0, time_steps, s_T_gt, a_gt):

        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        if self.decoder.ll_policy.a_dist == 'normal':
            a_dist = Normal.Normal(a_means, a_sigs)
        elif self.decoder.ll_policy.a_dist == 'tanh_normal':
            base_dist = Normal.Normal(a_means, a_sigs)
            transform = torch.distributions.transforms.TanhTransform()
            a_dist = TransformedDistribution(base_dist, [transform])
        else:
            assert False, f'{self.decoder.ll_policy.a_dist} not supported'

        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)

        z_prior_means,z_prior_sigs = self.prior(s_0) #self.prior(states[:,0:1,:]) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        # T = time_steps#states.shape[1]
        # s_T = s_T_gt#states[:,-1:,:]  
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T_gt), dim=-1))/time_steps # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(a_gt), dim=-1)) 
        s_T_ent  =  torch.mean(torch.sum(s_T_dist.entropy(), dim=-1))/time_steps
        # print('a_sigs: ', a_sigs)
        # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
        # loss term correpsonding ot kl loss between posterior and prior
        # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/time_steps # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss + self.ent_pen*s_T_ent

        return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent

    def update_skills(self, z_prev, z_updated, b_mean, b_sig, running_l, executed_skills, greedy=False):
        """
        Prior:
            p(b_t|z_{1:t-1}) = \begin{cases}
                0, & \text{if} \; n(z_t) \geq N_{max} \\
                1, & \text{elseif} \; l(z_t) \geq T_{max} \\
                \sigma(f_{\theta}(s_{1:t}, z_{1:t-1})), & \text{otherwise}
            \end{cases}
        Copy -> keep prev skill
        Read -> update skill
        """
        # Assume you already get a batch of sampled skills

        # sample b from b_dist
        if greedy:
            b_probabilities = b_mean
        else:
            b_probabilities = self.reparameterize(b_mean, b_sig)

        b, _ = utils.boundary_sampler(b_probabilities.squeeze(), temperature=self.temperature)

        # Now, use inductive bias / prior
        over_max_len_idxs = running_l >  self.max_skill_len
        over_max_n_skills_idxs = executed_skills >= self.max_skills_per_seq

        # overwrite b with prior
        b[over_max_len_idxs, 0] = 0.0
        b[over_max_len_idxs, 1] = 1.0 # if skill_len > max_skill_len, update
        b[over_max_n_skills_idxs, 0] = 1.0 # if total executed skills > N max skills allowed, don't update
        b[over_max_n_skills_idxs, 1] = 0.0

        copy, read = b[:, 0].unsqueeze(dim=-1).unsqueeze(dim=-1), b[:, 1].unsqueeze(dim=-1).unsqueeze(dim=-1)

        z = (copy * z_prev) + (read.clone() * z_updated.clone()) # TODO: will (most likely) make it slow
        # only update last entry for skill and b
        return z, b


class SkillModelTerminalStateDependentPrior(SkillModelStateDependentPrior):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=False,beta=1.0,alpha=1.0,fixed_sig=None):
        super(SkillModelTerminalStateDependentPrior, self).__init__(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig)

        self.prior = TerminalStateDependentPrior(state_dim,z_dim,h_dim)
        # self.prior = Prior(state_dim,z_dim,h_dim)

    def get_losses(self,states,actions):
        # get state output distributions, 
        s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)
        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        a_dist = Normal.Normal(a_means, a_sigs)
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        
        z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:],states[:,-1:,:]) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)



        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        T = states.shape[1]
        s_T = states[:,-1:,:]  
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),   dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) 
        s_T_ent  = torch.mean(torch.sum(s_T_dist.entropy(),       dim=-1))/T
        # print('a_sigs: ', a_sigs)
        # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
        # loss term correpsonding ot kl loss between posterior and prior
        # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        # z_post_dist_detached = Normal.Normal(z_post_means.detach(),z_post_sigs.detach())
        # z_prior_means_non_term_state_dep = self.prior(state_dim)

        loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss
        # loss_tot = s_T_loss + kl_loss

        return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent

    
    def get_expected_cost_for_cem(self, s0, skill_seq, sT_seq, goal_state, use_epsilons=True, plot=False):
        '''
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = [torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()]
        # costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        for i in range(skill_seq_len):
            
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i,sT_seq[:,i:i+1,:])
                
                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]
            
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
            costs.append(cost_i)
            
            pred_states.append(s_i)
        
        costs = torch.stack(costs,dim=1)  # should be a batch_size x T or batch_size x T 
        costs,_ = torch.min(costs,dim=1)  # should be of size batch_size
        # print('costs: ', costs)
        # print('costs.shape: ', costs.shape)

        if plot:
            plt.figure()
            plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
            plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
            # plt.xlim([0,25])
            # plt.ylim([0,25])
            pred_states = torch.cat(pred_states,1)
            
            plt.plot(pred_states[:,:,0].T.detach().cpu().numpy(),pred_states[:,:,1].T.detach().cpu().numpy())
                
            plt.savefig('pred_states_cem')
        
        return costs

class SkillPolicy(nn.Module):
    def __init__(self,state_dim,z_dim,h_dim):
        super(SkillPolicy,self).__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))

    def forward(self,state):

        return self.layers(state)

class DecoderWithLLDynamics(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,ff=False):

        super(DecoderWithLLDynamics,self).__init__()
        
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.ff = ff

        if ff:
            self.ll_dynamics = LowLevelDynamicsFF(state_dim,a_dim,h_dim)
        else:
            self.ll_dynamics = LowLevelDynamics(state_dim,a_dim,h_dim)
        self.ll_policy = LowLevelPolicy(state_dim,a_dim,z_dim,h_dim, 'normal')
        
    def forward(self,states,actions,z):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            z:      batch_size x 1 x z_dim sampled z/skill variable
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        
        # s_0 = states[:,0:1,:]
        # z_detached = z.detach()
        s_next_mean,s_next_sig = self.ll_dynamics(states[:,:-1,:],actions[:,:-1,:])
        a_mean,a_sig   = self.ll_policy(states,z)


        return s_next_mean,s_next_sig,a_mean,a_sig

    # def get_expected_sT_dist(self,s0,z,H,n_samples=10):

    #     # tile s0 and z n_samples number of times
    #     batch_size,T,state_dim = s0.shape
    #     _,_,z_dim = z.shape
    #     s0_tiled = torch.stack(n_samples*[s0])
    #     z_tiled = torch.stack(n_samples*[z])
    #     s0_tiled_reshaped = s0_tiled.reshape(n_samples*batch_size,T,state_dim)
    #     z_tiled_reshaped  =  z_tiled.reshape(n_samples*batch_size,1,z_dim)
    #     h = None
    #     state = s0_tiled_reshaped
    #     for t in range(H):
    #         a_mean,a_sig = self.ll_policy(state,z_tiled_reshaped)
    #         action = reparameterize(a_mean,a_sig)
    #         s_next_mean,s_next_sig,h = self.ll_dynamics(state,action,h)
    #         s_next = reparameterize(s_next_mean,s_next_sig)
    #         state = s_next

    #     sT_dist_means = s_next_mean.reshape(s0_tiled.shape)
    #     sT_dist_sigs  = s_next_sig.reshape( s0_tiled.shape)
    #     # mix = Categorical.Categorical(torch.ones(n_samples,))
    #     # comp = Normal.Normal(sT_dist_means,sT_dist_sigs)
    #     # sT_dist = MixtureSameFamily.MixtureSameFamily(mix, comp)
    #     sT_dist = GaussianMixtureDist(sT_dist_means,sT_dist_sigs)
    #     print('sT_dist_means.shape: ',sT_dist_means.shape)
    #     print('sT_dist_means[:,0,0,:2]: ', sT_dist_means[:,0,0,:2])
    #     print('sT_dist_sigs[:,0,0,:2]: ', sT_dist_sigs[:,0,0,:2])

    #     return sT_dist

    # def get_expected_next_state_dist(self,states,z):
    #     assert self.ff
    #     batch_size,T,state_dim = s0.shape
    #     _,_,z_dim = z.shape
    #     states_tiled = torch.stack(n_samples*[states],dim=0)
    #     z_tiled      = torch.stack(n_samples*[z],dim=0)
    #     # state_tiled_reshaped = state_tiled.reshape(n_samples*batch_size,T,state_dim)
    #     # z_tiled_reshaped     =  z_tiled.reshape(n_samples*batch_size,1,z_dim)

    #     a_means,a_sigs = self.ll_policy(states,z)
    #     a_means_tiled = torch.stack(n_samples*[a_means],dim=0) 
    #     a_sigs_tiled = torch.stack(n_samples*[a_sigs],dim=0)
    #     actions = reparameterize(a_means,a_sigs)  # this should be n_samples distinct action sample for every (state,z)

    #     next_s_means,next_s_sigs = self.ll_dynamics(states_tiled[:,:,:-1,:],actions[:,:,:-1,:])
    #     s_next_dist = GaussianMixtureDist(s_next_means,s_next_sigs)

    #     return s_next_dist

    def get_expected_sT_dist(self,s0,z,H,n_samples):
        '''
        s0: batch_size x 1 x state_dim tensor of initial states
        z:  batch_size x 1 x z_dim tensor of skill vectors
        H:  how long to run skills for 
        n_samples: number of samples to use to get approximate expectated sT dist
        '''

        batch_size,_,s_dim = s0.shape
        z_dim = z.shape[-1]

        # Now we're going to simulate rollouts from s0 using z, n_sample times for every s0/z combo
        # tile s0 and z along new 0th dimension
        s0_tiled = torch.stack(n_samples*[s0]) # n_samples x batch_size x 1 x s_dim
        z_tiled = torch.stack(n_samples*[z])   # n_samples x batch_size x 1 x z_dim
        state = s0_tiled
        for i in range(H):
            # get actions
            a_means,a_sigs = self.ll_policy(state,z_tiled)
            action = reparameterize(a_means,a_sigs)
            s_next_means,s_next_sigs = self.ll_dynamics(state,action)
            state = reparameterize(s_next_means,s_next_sigs)

        # now we should have samples from P(sT|s0,z).  We can approximately compute the entropy of this distribution.
        # estimate the mean and covariance
        sT_means = torch.mean(state,dim=0) # back down to batch_size x 1 x s_dim
        assert sT_means.shape == (batch_size,1,s_dim)
        sT_vars  = torch.var(state, dim=0) # back down to batch_size x 1 x s_dim
        assert sT_vars.shape == (batch_size,1,s_dim)

        return sT_means,sT_vars

    def get_sT_entropy(self,s0,z,H,n_samples=10):

        sT_means,sT_vars = self.get_expected_sT_dist(s0,z,H,n_samples)

        sT_dist = Normal.Normal(sT_means,sT_vars)

        H = sT_dist.entropy()

        return H










class GaussianMixtureDist(nn.Module):
    def __init__(self,means,sigs):
        super(GaussianMixtureDist, self).__init__()
        self.n_mix_comps = means.shape[0]
        self.mix_comps = [Normal.Normal(means[i,...],sigs[i,...]) for i in range(self.n_mix_comps)]
        self.weight = 1/self.n_mix_comps

    def log_prob(self,x):

        log_prob_comps = torch.stack([self.mix_comps[i].log_prob(x) for i in range(self.n_mix_comps)])
        
        log_prob = torch.logsumexp(log_prob_comps,dim=0) + np.log(self.weight)

        return log_prob
    
    def sample(self,n_samples):
        samples = torch.cat([self.mix_comps[i].sample((n_samples,)) for i in range(self.n_mix_comps)])
        return samples

class BilevelSkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim):
        super(BilevelSkillModel, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.decoder = DecoderWithLLDynamics(state_dim,a_dim,z_dim,h_dim)
        self.prior   = Prior(state_dim,z_dim,h_dim)

    def forward(self):
        pass

    def get_losses(self,states,actions):

        '''
        s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)

        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        if self.decoder.ll_policy.a_dist == 'normal':
            a_dist = Normal.Normal(a_means, a_sigs)
        elif self.decoder.ll_policy.a_dist == 'tanh_normal':
            base_dist = Normal.Normal(a_means, a_sigs)
            transform = torch.distributions.transforms.TanhTransform()
            a_dist = TransformedDistribution(base_dist, [transform])
        else:
            assert False
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        # z_prior_means = torch.zeros_like(z_post_means)
        # z_prior_sigs = torch.ones_like(z_post_sigs)
        z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:]) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        T = states.shape[1]
        s_T = states[:,-1:,:]  
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions),dim=-1)) 
        # print('a_sigs: ', a_sigs)
        # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
        # loss term correpsonding ot kl loss between posterior and prior
        # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = s_T_loss + a_loss + self.beta*kl_loss
        # loss_tot = s_T_loss + kl_loss

        return  loss_tot, s_T_loss, a_loss, kl_loss
        '''

        batch_size, T, _ = states.shape

        s_next = states[:,1:,:]
        s0 = states[:,:1,:]
        sT = states[:,-1:,:]

        z_post_means,z_post_sigs = self.encoder(states,actions)
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs) 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        
        z_prior_means,z_prior_sigs = self.prior(s0) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        s_next_means,s_next_sigs,a_means,a_sigs = self.decoder(states,actions,z_sampled)

        a_dist = Normal.Normal(a_means, a_sigs)
        s_next_dist = Normal.Normal(s_next_means,s_next_sigs)
        
        exp_next_s_dist = self.decoder.get_expected_sT_dist(s0,z_sampled,T)

        s_T_loss = - torch.sum(sT_dist.log_prob(sT))/(batch_size*T)
        s_loss = - torch.sum(s_next_dist.log_prob(s_next))/(batch_size*T)
        a_loss = - torch.sum(a_dist.log_prob(actions))/(batch_size*T)

        kl_loss = torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist))/(batch_size*T) # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss = s_loss + a_loss + s_T_loss + kl_loss

        return loss, s_loss, a_loss, s_T_loss, kl_loss



    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps



        
class BilevelSkillModelV2(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim):
        super(BilevelSkillModelV2, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.decoder = DecoderWithLLDynamics(state_dim,a_dim,z_dim,h_dim,ff=True)
        self.prior   = Prior(state_dim,z_dim,h_dim)

    def forward(self):
        pass

    def get_losses(self,states,actions):


        batch_size, T, _ = states.shape

        s_next = states[:,1:,:]
        s0 = states[:,:1,:]
        sT = states[:,-1:,:]

        z_post_means,z_post_sigs = self.encoder(states,actions)
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs) 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        
        z_prior_means,z_prior_sigs = self.prior(s0) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        s_next_means,s_next_sigs,a_means,a_sigs = self.decoder(states,actions,z_sampled)

        a_dist = Normal.Normal(a_means, a_sigs)
        s_next_dist = Normal.Normal(s_next_means,s_next_sigs)
        
        exp_s_next_dist = self.decoder.get_expected_next_state_dist(states,z_sampled,T)

        s_next_loss = - torch.sum(exp_s_next_dist.log_prob(sT))/(batch_size*T)
        s_loss = - torch.sum(s_next_dist.log_prob(s_next))/(batch_size*T)
        a_loss = - torch.sum(a_dist.log_prob(actions))/(batch_size*T)

        kl_loss = torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist))/(batch_size*T) # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss = s_loss + a_loss + s_next_loss + kl_loss

        return loss, s_loss, a_loss, s_next_loss, kl_loss



    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps


class BilevelSkillModelV3(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,ent_pen):
        super(BilevelSkillModelV3, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.ent_pen = ent_pen

        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.decoder = DecoderWithLLDynamics(state_dim,a_dim,z_dim,h_dim,ff=True)
        self.prior   = Prior(state_dim,z_dim,h_dim)

    def forward(self):
        pass

    def get_losses(self,states,actions):


        batch_size, T, _ = states.shape

        s_next = states[:,1:,:]
        s0 = states[:,:1,:]
        sT = states[:,-1:,:]

        z_post_means,z_post_sigs = self.encoder(states,actions)
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs) 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        
        z_prior_means,z_prior_sigs = self.prior(s0) 
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        s_next_means,s_next_sigs,a_means,a_sigs = self.decoder(states,actions,z_sampled)

        a_dist = Normal.Normal(a_means, a_sigs)
        s_next_dist = Normal.Normal(s_next_means,s_next_sigs)
        
        # exp_s_next_dist = self.decoder.get_expected_next_state_dist(states,z_sampled,T)

        s_loss = - torch.sum(s_next_dist.log_prob(s_next))/(batch_size*T)
        a_loss = - torch.sum(a_dist.log_prob(actions))/(batch_size*T)

        sT_ent =   torch.sum(self.decoder.get_sT_entropy(s0,z_sampled,T))/(batch_size*T)

        kl_loss = torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist))/(batch_size*T) 
        loss = s_loss + a_loss + kl_loss + self.ent_pen*sT_ent

        return loss, s_loss, a_loss, kl_loss, sT_ent



    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps
    

class SkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist='normal'):
        super(SkillModel, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim, a_dist)  # TODO


    def forward(self,states,actions):
        
        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution 
        '''

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means,z_post_sigs = self.encoder(states,actions)
        # STEP 2. sample z from posterior 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        # STEP 3. Pass z_sampled and states through decoder 
        s_T_mean, s_T_sig, a_means, a_sigs = self.decoder(states,z_sampled)



        return s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs

        

    def get_losses(self,states,actions):
        '''
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)] 
          + E_q [sum_t=0^T P(a_t|s_t,z)] 
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
        Distributions we need:
        '''

        s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)

        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        a_dist = Normal.Normal(a_means, a_sigs)
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        z_prior_means = torch.zeros_like(z_post_means)
        z_prior_sigs = torch.ones_like(z_post_sigs)
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        T = states.shape[1]
        s_T = states[:,-1:,:]  
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions),dim=-1)) 
        # print('a_sigs: ', a_sigs)
        # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
        # loss term correpsonding ot kl loss between posterior and prior
        # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = s_T_loss + a_loss + kl_loss
        # loss_tot = s_T_loss + kl_loss

        return  loss_tot, s_T_loss, a_loss, kl_loss
    
    def get_expected_cost(self, s0, skill_seq, goal_states):
        '''
        s0 is initial state  # batch_size x 1 x s_dim
        skill sequence is a 1 x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_states = torch.cat(batch_size * [goal_states],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        #pred_states = []
        for i in range(skill_seq_len):
            z_i = skill_seq[:,i:i+1,:] # might need to reshape
            # converting z_i from 1x1xz_dim to batch_size x 1 x z_dim
            z_i = torch.cat(batch_size*[z_i],dim=0) # feel free to change this to tile
            # use abstract dynamics model to predict mean and variance of state after executing z_i, conditioned on s_i
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled
            
            #pred_states.append(s_sampled)
        
        #compute cost for sequence of states/skills
        pred_states = s_i
        cost = torch.mean((pred_states[:,:,:2] - goal_states[:,:,:2])**2)
        
        return cost
    
    
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps


