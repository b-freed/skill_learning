import os
from comet_ml import Experiment

from sklearn.metrics import accuracy_score 
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
from logger import Logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from skill_model import ContinuousSkillModel as SkillModel
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
import utils
import time
import tqdm
import os
from configs import Configs
import torch.distributions.kl as KL


def run_iteration(model, data_loader, cfgs, model_optimizer=None, train_mode=False):
	loss_tracker = utils.DataTracker(verbose=cfgs['verbose'])

	device = model.device

	model.train() if train_mode else model.eval()

	get_masked_loss = lambda raw_loss, data_lens_tensor, mask, bs: (1/bs)*torch.sum(torch.sum(torch.sum(raw_loss, dim=-1) * mask, dim=-1)/data_lens_tensor)

	for batch_id, (data_, data_lens) in tqdm.tqdm(enumerate(data_loader), desc=f"Progress"):
		data_ = data_.to(device) # (batch_size, 1000, state_dim + action_dim)
		states, actions = data_[:, :, :state_dim], data_[:, :, state_dim:]

		# Infer skills for the data
		z_post_means, z_post_sigs = model.encoder(states, actions, unequal_subtraj_lens=True, data_lens=data_lens)

		# sample z_ts from the posterior
		z_t = model.reparameterize(z_post_means, z_post_sigs)

		# # Extract s0 and sT ground truths from the data sequence
		# s0 = states[:, :1, :] # (batch_size, 1, state_dim)
		# sT_gt = states[torch.arange(batch_size, device=device), data_lens_tensor-1, :].unsqueeze(dim=1) # (batch_size, 1, state_dim)

		# Use skills and states to generate terminal state and action sequences
		sT_mean, sT_sig, a_means, a_sigs, b_ = model.decoder(states, z_t, s0)

		b = b_.squeeze(2) if b_.ndim == 3 else b_

		# Compute skill and termination prior
		z_prior_means, z_prior_sigs = model.prior(s0)

		# Construct required distributions
		sT_dist = Normal.Normal(sT_mean, sT_sig)
		z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)


		# Increase gt terminal state likelihood.
		sT_loss = -sT_dist.log_probs(sT_gt)
		# sT_loss = -get_masked_loss(sT_dist.log_prob(sT_gt), data_lens_tensor, loss_mask, batch_size)

		# (?) Entropy in 
		# s_T_ent = torch.mean(torch.sum(s_T_dist.entropy(), dim=-1))/time_steps
		s_T_ent = 0

		# Increase gt action likelihood.
		a_loss = -get_masked_loss(a_dist.log_prob(actions), data_lens_tensor, loss_mask, batch_size)

		# Tighten the KL bounds.
		# kl_loss = get_masked_loss(KL.kl_divergence(z_post_dist, z_prior_dist), data_lens_tensor, loss_mask, batch_size)
		kl_loss = -KL.kl_divergence(z_post_dist, z_prior_dist)

		# Incentivize longer skill lengths.
		sl_loss = 0

		tot_loss = cfgs['a_loss_coeff'] * a_loss + cfgs['sT_loss_coeff'] * sT_loss + \
					cfgs['kl_loss_coeff'] * kl_loss + cfgs['sl_loss_coeff'] * sl_loss

		if train_mode:
			model_optimizer.zero_grad()
			tot_loss.backward()
			model.clip_gradients()
			model_optimizer.step()

		loss_tracker.update(**{
			'tot_loss': tot_loss,
			'a_loss': a_loss,
			'sT_loss': sT_loss,
			'kl_loss': kl_loss,
			'sl_loss': sl_loss,
		})

	return loss_tracker.to_dict(mean=True)


if __name__ == '__main__':
	hp = Configs()

	# set environment variable for device
	os.environ["_DEVICE"] = hp.device

	logger = Logger(hp)

	# TODO: might want to get rid of this
	env = gym.make(hp.env_name)
	state_dim = env.observation_space.shape[0]
	a_dim = env.action_space.shape[0]

	# Load the dataset files
	raw_data = np.load(os.path.join(hp.data_dir, f'{hp.env_name}.npz'))
	inputs_train = torch.from_numpy(raw_data['inputs_train'])
	inputs_test  = torch.from_numpy(raw_data['inputs_test'])

	# Instantiate a skill model
	model = SkillModel(hp.dict)
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	# Create dataloaders
	train_loader = DataLoader(inputs_train, batch_size=hp.batch_size, num_workers=0)
	test_loader = DataLoader(inputs_test, batch_size=hp.batch_size, num_workers=0)

	# It's time to roll baby!
	for epoch in range(hp.n_epochs):
		# Run training loop
		losses = run_iteration(model, train_loader, hp, model_optimizer=model_optimizer, train_mode=True)

		# Update logger
		logger.update(epoch, losses)
		
		# Run evaluation loop
		with torch.no_grad():
			test_losses = run_iteration(model, test_loader, hp, train_mode=False)

		# Update logger
		logger.update(epoch, test_losses)
		
		# Periodically save training
		if epoch % 10 == 0:
			logger.save_training_state(epoch, model, model_optimizer, 'latest.pth')

		# Save on hitting new best
		if test_losses['tot_loss'] < logger.min_test_loss:
			logger.min_test_loss = test_losses['tot_loss']
			logger.save_training_state(epoch, model, model_optimizer, 'best.pth')