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
import time
import os
from configs import HyperParams


def run_iteration(model, initial_skill, model_optimizer=None, train_mode=False):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	sl_losses = []
	s_T_ents = []

	model.train() if train_mode else model.eval()

	for batch_id, data_ in enumerate(train_loader):
		if data_.shape[0] != hp.batch_size:
			continue
		data_ = data_.to(hp.device)
		loss_tot, s_T_loss, a_loss, kl_loss, sl_loss, s_T_ent = 0, 0, 0, 0, 0, 0

		# sample initial skill prior, start with zero initial skill(s)
		data = data_[:, :hp.H_min, ...]
		states = data[:,:,:model.state_dim] # first state_dim elements are the state
		actions = data[:,:,model.state_dim:] # rest are actions
		z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, initial_skill)

		z_t = model.reparameterize(z_post_means, z_post_sigs)
		z_t_history = z_t.repeat(1, hp.H_min, 1)
		z_t_p = z_t_history[:, -1:, :].detach().clone()

		z_t, b_t = model.update_skills(z_t_p, z_t, b_post_means, b_post_sigs)
		z_t_history = torch.cat([z_t_history, z_t], dim=1)
		sl_tracker = b_t[:, 0] # loss for sticking w/ skills 

		for H in range(hp.H_min+1, hp.H_max):
			# if b_t == 1:
			# 	# sample new z_t
			data = data_[:, :H, ...]
			states = data[:, :, :model.state_dim] # first state_dim elements are the state
			actions = data[:, :, model.state_dim:] # rest are actions
			s_0 = states[:,0:1,:]
			s_T_gt = states[:,-1:,:]  

			s_T_mean, s_T_sig, a_means, a_sigs = model.decoder(states, z_t)

			loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.compute_losses(s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs, s_0, H, s_T_gt, actions) 

			sl_loss_ = model.gamma * sl_tracker.sum()
			loss_tot_ += sl_loss_

			loss_tot += loss_tot_
			s_T_loss += s_T_loss_
			a_loss += a_loss_
			kl_loss += kl_loss_
			sl_loss += sl_loss_
			s_T_ent += s_T_ent_

			z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(states, actions, z_t_history)
			z_t = model.reparameterize(z_post_means, z_post_sigs)
			z_t_p = z_t_history[:, -1:, :].detach().clone()

			z_t, b_t = model.update_skills(z_t_p, z_t, b_post_means, b_post_sigs)
			z_t_history = torch.cat([z_t_history, z_t], dim=1)
			sl_tracker += b_t[:, 0] # loss for sticking w/ skills 

		if train_mode:	
			model_optimizer.zero_grad()
			loss_tot.backward()
			model_optimizer.step()

		# log losses
		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())
		sl_losses.append(sl_loss.item())
		s_T_ents.append(s_T_ent.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents), np.mean(sl_losses)

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
experiment.set_name(hp.exp_name)

# First, instantiate a skill model
if hp.term_state_dependent_prior:
	model = SkillModelTerminalStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, \
				state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
				fixed_sig=hp.fixed_sig).to(hp.device)
elif hp.state_dependent_prior:
	model = SkillModelStateDependentPriorAutoTermination(state_dim, a_dim, hp.z_dim, hp.h_dim, \
				a_dist=hp.a_dist, state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
				temperature=hp.temperature, gamma=hp.gamma, max_sig=hp.max_sig, fixed_sig=hp.fixed_sig, \
				ent_pen=hp.ent_pen, encoder_type=hp.encoder_type, state_decoder_type=hp.state_decoder_type).to(hp.device)
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
	loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss = run_iteration(model, initial_skill, model_optimizer=model_optimizer, train_mode=True)
	
	print(f'Iteration: {i}')
	print("--------TRAIN---------")
	
	print('loss: ', loss)
	print('s_T_loss: ', s_T_loss)
	print('a_loss: ', a_loss)
	print('kl_loss: ', kl_loss)
	print('s_T_ent: ', s_T_ent)
	print('s_T_ent: ', s_T_ent)
	print('sl_loss: ', sl_loss)
	print('')

	experiment.log_metric("loss", loss, step=i)
	experiment.log_metric("s_T_loss", s_T_loss, step=i)
	experiment.log_metric("a_loss", a_loss, step=i)
	experiment.log_metric("kl_loss", kl_loss, step=i)
	experiment.log_metric("s_T_ent", s_T_ent, step=i)
	experiment.log_metric("sl_loss", sl_loss, step=i)

	with torch.no_grad():
		test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss = run_iteration(model, initial_skill, train_mode=False)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	print('test_s_T_loss: ', test_s_T_loss)
	print('test_a_loss: ', test_a_loss)
	print('test_kl_loss: ', test_kl_loss)
	print('test_s_T_ent: ', test_s_T_ent)
	print('sl_loss: ', sl_loss)
	print('')

	experiment.log_metric("test_loss", test_loss, step=i)
	experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
	experiment.log_metric("test_a_loss", test_a_loss, step=i)
	experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
	experiment.log_metric("test_s_T_ent", test_s_T_ent, step=i)
	experiment.log_metric("test_sl_loss", sl_loss, step=i)

	if i % 10 == 0:
		checkpoint_path = os.path.join(hp.log_dir, 'latest.pth')
		torch.save({'model_state_dict': model.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)

	if test_loss < min_test_loss:
		min_test_loss = test_loss
		checkpoint_path = os.path.join(hp.log_dir, 'best.pth')
		torch.save({'model_state_dict': model.state_dict(),
			        'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)