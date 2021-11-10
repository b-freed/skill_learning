

'''
Hi Anirudh.

Here's a high-level description of the skeleton code to follow.
We're using variational inference (VI) to learn skills from an offline dataset.
That means we'll need an "encoder" and a "decoder" (basically same as the encoder and a decoder in a VAE). 
As in a VAE, we'll have latent variables (z), in this case representing skills.
The reason we want to model skills as latent variables is that they are essentially unobserved variables 
(you do not "observe" the skill someone is executing, only the results)
that are useful to represent the distribution of behaviors that were demonstrated in the offline dataset.
The role of the encoder is to infer the skill (the same way an encoder in a vae infers the latent variable).
The role of the decoder is to output the prediction about terminal states and actions given the latent variable.

Also like a VAE, the objective we will optimize is a variational lower bound, the derivaiton of which is in the notes I sent you.


The code will be structured similar to a VAE, so that the conceptual bridge becomes a bit easier.
I'll write the skeleton, and let you have a crack at filling it in. 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal


class AbstractDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)

    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim):

        super(AbstractDynamics,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.mean_layer = nn.Linear(h_dim,state_dim)
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,state_dim),nn.Softplus())

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


class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.

    See Encoder and Decoder for more description


    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim):

        super(LowLevelPolicy,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.mean_layer = nn.Linear(h_dim,a_dim)
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())



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
        z_tiled = z.tile([1,state.shape[1],1]) #not sure about this 

        # Concat state and z_tiled
        state_z = torch.cat([state,z_tiled],dim=-1)
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
        state = torch.tensor(state,device=torch.device('cuda:0'))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)
        
        return action.detach().cpu().numpy()
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps

        
        
        
        

class Encoder(nn.Module):
    '''
    Encoder module.
    We can try the following architecture initially:
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim):
        super(Encoder, self).__init__()


        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.emb_layer  = nn.Linear(state_dim+a_dim,h_dim)
        self.rnn        = nn.GRU(h_dim,h_dim,batch_first=True,bidirectional=True)
        self.mean_layer = nn.Linear(h_dim*2,z_dim)
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim*2,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive


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

        # concatenate states and actions
        s_a = torch.cat([states,actions],dim=-1)
        # embedd states and actions
        embed = self.emb_layer(s_a)
        # through rnn
        feats,hn = self.rnn(embed)
        # get z_mean and z_sig by passing rnn output through mean_layer and sig_layer
        z_mean = self.mean_layer(feats[:,-1:,:])
        z_sig = self.sig_layer(feats[:,-1:,:])
        
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
    def __init__(self,state_dim,a_dim,z_dim,h_dim):

        super(Decoder,self).__init__()
        
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim) # TODO
        self.ll_policy = LowLevelPolicy(state_dim,a_dim,z_dim,h_dim) # TODO
        self.emb_layer  = nn.Linear(state_dim+z_dim,h_dim)
        self.fc = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        
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
        sT_mean,sT_sig = self.abstract_dynamics(s_0,z)
        a_mean,a_sig   = self.ll_policy(states,z)


        return sT_mean,sT_sig,a_mean,a_sig


class SkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim):
        super(SkillModel, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim)  # TODO
        self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim)


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
        s_T = states[:,-1:,:]  # <-probably oging to be of size batch x state_dim.. We want batch x 1 x state_dim
        s_T = s_T.unsqueeze(1) # adds extra dimension along time axis
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions),dim=-1))
        # loss term correpsonding ot kl loss between posterior and prior
        kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist.loc, z_prior_dist.loc),dim=-1))

        loss_tot = s_T_loss + a_loss + kl_loss

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
        pred_states = []
        for i in range(skill_seq_len):
            z_i = skill_seq[:,i:i+1,:] # might need to reshape
            # converting z_i from 1x1xz_dim to batch_size x 1 x z_dim
            z_i = torch.cat(batch_size*[z_i],dim=0) # feel free to change this to tile
            # use abstract dynamics model to predict mean and variance of state after executing z_i, conditioned on s_i
            s_mean, s_sig = self.abstract_dynamics(s_i,z_i)
            
            # sample s_i+1 using reparameterize
            s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled
            
            pred_states.append(s_sampled)
        
        #compute cost for sequence of states/skills
        pred_states = torch.stack(pred_states,dim = 1)
        cost = torch.mean((pred_states - goal_states)**2)
        
        return cost
    
    
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps
