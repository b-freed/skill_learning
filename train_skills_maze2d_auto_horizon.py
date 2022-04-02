from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillModelTerminalStateDependentPrior, SkillModelStateDependentPriorAutoTermination
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
import utils
import os

def train(model, model_optimizer, initial_skill):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	model.train()

	for batch_id, data_ in enumerate(train_loader):
		if data_.shape[0] != hp.batch_size:
			continue
		data_ = data_.to(hp.device)
		loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent = 0, 0, 0, 0, 0

		# sample initial skill prior, start with zero initial skill(s)
		data = data_[:, :hp.H_min, ...]
		states = data[:,:,:model.state_dim] # first state_dim elements are the state
		actions = data[:,:,model.state_dim:] # rest are actions
		z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, initial_skill)

		z_t = model.reparameterize(z_post_means, z_post_sigs)
		z_t_history = z_t.repeat(1, hp.H_min + 1, 1)
		# z_t = torch.cat([z_t, z_t[:, -1:, :]], dim=1)
		b_t = model.decide_termination(b_post_means, b_post_sigs)

		for H in range(hp.H_min+1, hp.H_max):
			# if b_t == 1:
			# 	# sample new z_t
			data = data_[:, :H, ...]
			states = data[:, :, :model.state_dim] # first state_dim elements are the state
			actions = data[:, :, model.state_dim:] # rest are actions
			s_0 = states[:,0:1,:]
			s_T_gt = states[:,-1:,:]  

			s_T_mean, s_T_sig, a_means, a_sigs = model.decoder(states, z_t)

			# loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.get_losses(states, actions)
			loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.compute_losses(s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs, s_0, H, s_T_gt, actions) 

			loss_tot += loss_tot_
			s_T_loss += s_T_loss_
			a_loss += a_loss_
			kl_loss += kl_loss_
			s_T_ent += s_T_ent_

			z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, z_t_history)
			z_t_ = model.reparameterize(z_post_means, z_post_sigs)
			b_t_ = model.decide_termination(b_post_means, b_post_sigs)
			z_t_history = torch.cat([z_t_history, z_t_history[:, -1:, :]], dim=1)
			if (b_t == True).any():
				new_indices = torch.where(b_t == True)[0]
				# update skills
				z_t[new_indices] = z_t_[new_indices]
				z_t_history[new_indices] = z_t_[new_indices]
			b_t = b_t_
	
		model_optimizer.zero_grad()
		loss_tot.backward()
		model_optimizer.step()

		# log losses
		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())
		s_T_ents.append(s_T_ent.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents)

def test(model, initial_skill):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	model.train()

	for batch_id, data_ in enumerate(train_loader):
		if data_.shape[0] != hp.batch_size:
			continue
		data_ = data_.to(hp.device)
		loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent = 0, 0, 0, 0, 0

		# sample initial skill prior, start with zero initial skill(s)
		data = data_[:, :hp.H_min, ...]
		states = data[:,:,:model.state_dim] # first state_dim elements are the state
		actions = data[:,:,model.state_dim:] # rest are actions
		z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, initial_skill)

		z_t = model.reparameterize(z_post_means, z_post_sigs)
		z_t_history = z_t.repeat(1, hp.H_min + 1, 1)
		# z_t = torch.cat([z_t, z_t[:, -1:, :]], dim=1)
		b_t = model.decide_termination(b_post_means, b_post_sigs)

		for H in range(hp.H_min+1, hp.H_max):
			# if b_t == 1:
			# 	# sample new z_t
			data = data_[:, :H, ...]
			states = data[:, :, :model.state_dim] # first state_dim elements are the state
			actions = data[:, :, model.state_dim:] # rest are actions
			s_0 = states[:,0:1,:]
			s_T_gt = states[:,-1:,:]  

			s_T_mean, s_T_sig, a_means, a_sigs = model.decoder(states, z_t)

			# loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.get_losses(states, actions)
			loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.compute_losses(s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs, s_0, H, s_T_gt, actions) 

			loss_tot += loss_tot_
			s_T_loss += s_T_loss_
			a_loss += a_loss_
			kl_loss += kl_loss_
			s_T_ent += s_T_ent_

			z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, z_t_history)
			z_t_ = model.reparameterize(z_post_means, z_post_sigs)
			b_t_ = model.decide_termination(b_post_means, b_post_sigs)
			z_t_history = torch.cat([z_t_history, z_t_history[:, -1:, :]], dim=1)
			if (b_t == True).any():
				new_indices = torch.where(b_t == True)[0]
				# update skills
				z_t[new_indices] = z_t_[new_indices]
				z_t_history[new_indices] = z_t_[new_indices]
			b_t = b_t_
	
		# log losses
		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())
		s_T_ents.append(s_T_ent.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents)


class HyperParams:
    def __init__(self):
        self.batch_size = 256
        self.h_dim = 256
        self.z_dim = 256
        self.lr = 5e-5
        self.wd = 0.001
        self.state_dependent_prior = True
        self.term_state_dependent_prior = False
        self.state_dec_stop_grad = True
        self.beta = 0.1
        self.alpha = 1.0
        self.ent_pen = 0.0
        self.max_sig = None
        self.fixed_sig = None
        self.H_min = 5
        self.H_max = 30
        self.stride = 1
        self.n_epochs = 50000
        self.test_split = .2
        self.a_dist = 'normal' # 'tanh_normal' or 'normal'
        self.encoder_type = 'state_action_sequence' #'state_sequence'
        self.state_decoder_type = 'mlp'
        self.env_name = 'antmaze-large-diverse-v0'
        self.device_id = 3
        self.device = f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
        if self.term_state_dependent_prior:
            self.filename = f'{self.env_name}_tsdp_H{self.H_max}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_log'
        else:
            self.filename = f'{self.env_name}_enc_type_{self.encoder_type}_state_dec_{self.state_decoder_type}_H_{self.H_max}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_ent_pen_{self.ent_pen}_log'
        self.msg = "variable_length_skills_2_30"


hp = HyperParams()

# set environment variable for device
os.environ["_DEVICE"] = hp.device

# dataset = utils.create_dataset_padded(utils.create_dataset_raw, hp.env_name)
dataset_ = utils.create_dataset_raw(hp.env_name)

states = dataset_['observations']
actions = dataset_['actions']
goals = dataset_['infos/goal']

N_episodes = len(states)
state_dim = states[0].shape[-1]
a_dim = actions[0].shape[-1]

N_train = int((1 - hp.test_split) * N_episodes)
N_test = N_episodes - N_train

states_train  = states[:N_train]
actions_train = actions[:N_train]
goals_train = goals[:N_train]

states_test  = states[N_train:]
actions_test = actions[N_train:]
goals_test   = goals[N_train:]

# obs_chunks_train, action_chunks_train, targets_train = chunks(states_train, actions_train, goals_train, H, stride)
# obs_chunks_test,  action_chunks_test,  targets_test  = chunks(states_test,  actions_test,  goals_test,  H, stride)

experiment = Experiment(api_key = 'Wlh5wstMNYkxV0yWRxN7JXZRu', project_name = 'dump')
experiment.set_name(hp.msg)

# First, instantiate a skill model
if hp.term_state_dependent_prior:
	model = SkillModelTerminalStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, state_dec_stop_grad=hp.state_dec_stop_grad,beta=hp.beta,alpha=hp.alpha,fixed_sig=hp.fixed_sig).to(hp.device)
elif hp.state_dependent_prior:
	model = SkillModelStateDependentPriorAutoTermination(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist,state_dec_stop_grad=hp.state_dec_stop_grad,beta=hp.beta,alpha=hp.alpha,max_sig=hp.max_sig,fixed_sig=hp.fixed_sig,ent_pen=hp.ent_pen,encoder_type=hp.encoder_type,state_decoder_type=hp.state_decoder_type).to(hp.device)
else:
	model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
	
model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

experiment.log_parameters(hp.__dict__)

# add chunks of data to a pytorch dataloader
inputs_train_ = utils.create_dataset_auto_horizon(states_train, actions_train, hp.H_max)
inputs_test_ = utils.create_dataset_auto_horizon(states_test, actions_test, hp.H_max)
inputs_train = torch.from_numpy(inputs_train_) # dim: (N_train, H, state_dim)
inputs_test  = torch.from_numpy(inputs_test_) # dim: (N_test, H, state_dim)

train_loader = DataLoader(
	inputs_train,
	batch_size=hp.batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=hp.batch_size,
	num_workers=0)

min_test_loss = 10**10

# TODO: Decide on something solid. rn doing skill learning for [H_min, H_max] time steps.
initial_skill = torch.zeros(hp.batch_size, hp.H_min, hp.z_dim).to(hp.device) # TODO: probably not the right thing to do. sample from prior/learable initial skill?

for i in range(hp.n_epochs):
	loss, s_T_loss, a_loss, kl_loss, s_T_ent = train(model, model_optimizer, initial_skill)
	
	print("--------TRAIN---------")
	
	print('loss: ', loss)
	print('s_T_loss: ', s_T_loss)
	print('a_loss: ', a_loss)
	print('kl_loss: ', kl_loss)
	print('s_T_ent: ', s_T_ent)
	print(i)
	experiment.log_metric("loss", loss, step=i)
	experiment.log_metric("s_T_loss", s_T_loss, step=i)
	experiment.log_metric("a_loss", a_loss, step=i)
	experiment.log_metric("kl_loss", kl_loss, step=i)
	experiment.log_metric("s_T_ent", s_T_ent, step=i)

	test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent = test(model, initial_skill)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	print('test_s_T_loss: ', test_s_T_loss)
	print('test_a_loss: ', test_a_loss)
	print('test_kl_loss: ', test_kl_loss)
	print('test_s_T_ent: ', test_s_T_ent)

	print(i)
	experiment.log_metric("test_loss", test_loss, step=i)
	experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
	experiment.log_metric("test_a_loss", test_a_loss, step=i)
	experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
	experiment.log_metric("test_s_T_ent", test_s_T_ent, step=i)

	if i % 10 == 0:
		checkpoint_path = 'checkpoints/'+ hp.filename + '.pth'
		torch.save({'model_state_dict': model.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)

	if test_loss < min_test_loss:
		min_test_loss = test_loss
		checkpoint_path = 'checkpoints/'+ hp.filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
			        'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)
