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





class SkillPolicyModel(nn.Module):

    def __init__(self,state_dim,a_dim,z_dim,h_dim,beta,a_dist,fixed_sig=None):
    	super().__init__()
		self.ll_policy = LowLevelPolicy(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,max_sig=None,fixed_sig=fixed_sig)
        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.prior = Prior(state_dim,z_dim,h_dim)
        self.beta = beta

    def get_losses(self,states,actions):
        # STEP 1. Encode states and actions to get posterior over z
		z_post_means,z_post_sigs = self.encoder(states,actions)
      	z_post_dist = Normal.Normal(z_post_means, z_post_sigs)

		# STEP 2. sample z from posterior 
		z_sampled = reparameterize(z_post_means,z_post_sigs)

		# STEP 3. Pass z_sampled and states through policy
        if self.a_dist != 'autoregressive': 
		    a_means, a_sigs = self.ll_policy(states,z_sampled)
        else:
		    a_means, a_sigs = self.ll_policy(states,actions,z_sampled)
        a_dist = Normal.Normal(a_means, a_sigs)

        # STEP 4. Get prior.
        z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:]) 
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

    	# STEP 5. Get losses.
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) 
		kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        loss = a_loss + self.beta*kl_loss

		return loss, a_loss, kl_loss 

class DynamicsModel(nn.Module):
    def __init__(self,state_dim,z_dim,h_dim,beta,per_element_sigma):
        super().__init__()
        self.dynamics_model = AbstractDynamics(state_dim,z_dim,h_dim,init_state_dependent=True,per_element_sigma=per_element_sigma)
        self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        self.prior = Prior(state_dim,z_dim,h_dim)
        self.beta = beta


    def get_losses(self,states,actions):
        s_0 = states[:,0:1,:]
		s_T = states[:,-1:,:]
        # STEP 1. Encode states and actions to get posterior over z
		z_post_means,z_post_sigs = self.encoder(states,actions)
		# STEP 2. sample z from posterior 
		z_sampled = self.reparameterize(z_post_means,z_post_sigs)
		# STEP 3. Pass z_sampled and states through decoder 
		s_T_mean, s_T_sig = self.dynamics_model(s0,z_sampled)
        # Get prior.
		z_prior_means,z_prior_sigs = self.prior(s0) 


        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 


		s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),   dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
		kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T


		loss = s_T_loss + self.beta*kl_loss

		return  loss, s_T_loss, kl_loss


    
    





class TwoStageSkillModel(nn.Module):

    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,stage,path,skill_beta=1.0,model_beta=1.0,fixed_sig=None,per_element_sigma=True,lr=5e-5,wd=0.0):
        super().__init__()
        assert stage in ['train_skills','train_dynamics']
        self.stage = stage
        self.path = path
        self.model_beta = model_beta
        self.skill_beta = skill_beta
        self.skill_model = SkillPolicyModel(state_dim,a_dim,z_dim,h_dim,skill_beta,a_dist,fixed_sig=None)
        self.dymamics_model = DynamicsModel(state_dim,z_dim,h_dim,model_beta,per_element_sigma)

        if stage = 'train_skills':
            self.optimizer = torch.optim.Adam(self.skill_model.parameters(), lr=lr, weight_decay=wd)
        elif stage == 'train_dynamics':
            self.E_optimizer = torch.optim.Adam(self.dynamics_model.encoder.parameters(),   lr=lr, weight_decay=wd)
            self.M_optimizer = torch.optim.Adam(self.dynamics_model.gen_model.parameters(), lr=lr, weight_decay=wd)

        else:
            assert False


        

        if stage == 'train_dynamics':
            print('LOADING SKILL MODEL FROM FILE')
            # load skill model from file
            path = os.path.join(path,'log_best.pth')  # create skill path
            # load skill model from skill path 
            checkpoint = torch.load(path)
            self.skill_model.load_state_dict(checkpoint['skill_model_state_dict'])


    def train(self,data):

        data = data.cuda()
		states = data[:,:,:model.state_dim]
		actions = data[:,:,model.state_dim:]

        if stage == 'train_skills':
            # get action losses
            loss,a_loss,kl_loss = self.skill_model.get_losses(states,actions)

            # perform training update
            self.skill_model.zero_grad()
		    loss.backward()
		    optimizer.step()

            return loss,a_loss,kl_loss


        elif stage == 'train_dynamics':
            ########### E STEP ###########
            E_loss = self.get_E_loss(states,actions)
            self.dynamics_model.zero_grad()
            E_loss.backward()
            self.E_optimizer.step()

            ########### M STEP ###########
            M_loss = self.get_M_loss(states,actions)
            self.dynamics_model.zero_grad()
            M_loss.backward()
            M_optimizer.step()

            return E_loss,M_loss

        else:
            assert False
    
    def validate(self,data):
        with torch.no_grad():
            data = data.cuda()
            states = data[:,:,:model.state_dim]
            actions = data[:,:,model.state_dim:]

            # if we're in the skill learning portion, get the "normal" skill learning losses
            if self.stage == 'skill_learning':
                
                # get action losses
                loss,a_loss,kl_loss = self.skill_model.get_losses(states,actions)

                return loss,a_loss,kl_loss

            else:
                
                loss,sT_loss,kl_loss = self.dynamics_model.get_losses(states,actions):
                return loss, sT_loss,kl_loss

        
    def get_E_loss(self,states,actions):

        assert self.encoder_type == 'state_action_sequence'
		assert not self.state_dec_stop_grad

        assert self.model_beta == 1.0

		batch_size,T,_ = states.shape
		denom = T*batch_size
		# get KL divergence between approximate and true posterior
		z_post_means,z_post_sigs = self.dynamics_model.encoder(states,actions)

		z_sampled = self.reparameterize(z_post_means,z_post_sigs)

		z_prior_means,z_prior_sigs = self.dynammics_model.prior(states[:,0:1,:]) 
		a_means,a_sigs = self.skill_model.ll_policy(states,z_sampled)

		post_dist = Normal.Normal(z_post_means,z_post_sigs)
		a_dist    = Normal.Normal(a_means,a_sigs)
		prior_dist = Normal.Normal(z_prior_means,z_prior_sigs)

		log_pi = torch.sum(a_dist.log_prob(actions)) / denom
		log_prior = torch.sum(prior_dist.log_prob(z_sampled)) / denom
		log_post  = torch.sum(post_dist.log_prob(z_sampled)) / denom
		

		return -log_pi + -self.model_beta*log_prior + self.model_beta*log_post

    # def get_M_loss(self):
    #     # TODO
    #     pass

    def get_M_loss(self,states,actions):

        batch_size,T,_ = states.shape
		denom = T*batch_size
		
		z_post_means,z_post_sigs = self.dymamics_model.encoder(states,actions)

		z_sampled = self.reparameterize(z_post_means,z_post_sigs)

		z_prior_means,z_prior_sigs = self.dynamics_model.prior(states[:,0:1,:]) 
		sT_mean, sT_sig = self.dynamics_model.dynamics_model(states,z_sampled)

		sT_dist  = Normal.Normal(sT_mean,sT_sig)
		prior_dist = Normal.Normal(z_prior_means,z_prior_sigs)

		sT = states[:,-1:,:]
		sT_loss = -torch.sum(sT_dist.log_prob(sT)) / denom
		prior_loss = -torch.sum(prior_dist.log_prob(z_sampled)) / denom

		return sT_loss + self.beta*prior_loss


    def save_checkpoint(self,filename):
        # TODO
        path = os.path.join(self.path,filename)
        if self.stage == 'train_skills':
            torch.save({
                'skill_model_state_dict': self.skill_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, self.path)
        else:
            torch.save({
                'skill_model_state_dict': self.skill_model.state_dict(),
                'dynamics_model_state_dict':self.
                'optimizer_state_dict': self.optimizer.state_dict(),
                'E_optimizer_state_dict': self.E_optimizer.state_dict(),
                'M_optimizer_state_dict': self.M_optimizer.state_dict()
                }, self.path)


    def load_from_checkpoint(self,filename):
        # TODO
        pass





