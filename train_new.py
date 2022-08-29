import os
from comet_ml import Experiment

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
	skill_lens = []
	loss_tracker = utils.DataTracker(verbose=cfgs['verbose'])

	device = cfgs.device

	model.train() if train_mode else model.eval()

	for data_raw in tqdm.tqdm(data_loader, desc=f"Progress"):
		data_raw = data_raw.to(device) # (batch_size, 1000, state_dim + action_dim)
		states, actions = data_raw[:, :, :state_dim], data_raw[:, :, state_dim:]

		# Extract s0 and sT ground truths from the data sequence
		s0 = states[:, :1, :] # (batch_size, 1, state_dim)
		# sT_gt = 0 # TODO states[torch.arange(batch_size, device=device), data_lens_tensor-1, :].unsqueeze(dim=1) # (batch_size, 1, state_dim)

		# Infer skills for the data
		# Encode states and actions to get posterior over z
		z_means, z_stds, z_len_means, z_len_stds = model.encoder(states)

		z_post_dist = Normal.Normal(z_means, z_stds)
		z_len_post_dist = model.encoder.get_skill_len_dist(z_len_means, z_len_stds)

		z_lens = z_len_post_dist.rsample()
		z_t = z_post_dist.rsample()

		z, skill_lens_batch = model.encoder.get_executable_skills(z_t, z_lens) # Get skill to be actually executed at each timestep.

		# Pass z_sampled and states through decoder
		sT_mean, sT_sig, a_means, a_sigs = model.decoder(states, z, s0)

		a_dist = Normal.Normal(a_means, a_sigs)

		# Compute skill and termination prior
		z_prior_means, z_prior_sigs, z_lens_prior_means, z_lens_prior_sigs = model.prior(s0)

		# Construct required distributions
		# sT_dist = Normal.Normal(sT_mean, sT_sig)
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)
		z_len_prior_dist = Normal.Normal(z_lens_prior_means, z_lens_prior_sigs)

		# Increase gt terminal state likelihood.
		# sT_loss = -sT_dist.log_probs(sT_gt)

		# (?) Entropy in sT prediction
		# sT_ent = torch.mean(sT_dist.entropy)

		# Increase gt action likelihood.
		a_loss = a_dist.log_prob(actions).mean()

		# Tighten the KL bounds.
		# kl_loss = get_masked_loss(KL.kl_divergence(z_post_dist, z_prior_dist), data_lens_tensor, loss_mask, batch_size)
		kl_loss = -KL.kl_divergence(z_post_dist, z_prior_dist).mean()# + KL.kl_divergence(z_len_post_dist, z_len_prior_dist))

		# Incentivize longer skill lengths.
		sl_loss = -torch.mean(z_lens)

		tot_loss = cfgs['a_loss_coeff'] * a_loss + cfgs['kl_loss_coeff'] * kl_loss + cfgs['sl_loss_coeff'] * sl_loss # + \
				 # cfgs['sT_loss_coeff'] * sT_loss + cfgs['sT_ent_coeff'] * sT_ent + \

		if train_mode:
			model_optimizer.zero_grad()
			tot_loss.backward()
			model.clip_gradients()
			model_optimizer.step()

		loss_tracker.update(**{
			'tot_loss': tot_loss,
			'a_loss': a_loss,
			'kl_loss': kl_loss,
			'sl_loss': sl_loss,
			# 'sT_loss': sT_loss,
			# 'sT_ent': sT_ent,
		})

		skill_lens.extend(skill_lens_batch)

	return loss_tracker.to_dict(mean=True), skill_lens


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
	model = SkillModel(hp.dict).to(hp.device)
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	# Create dataloaders
	train_loader = DataLoader(inputs_train, batch_size=hp.batch_size, num_workers=0)
	test_loader = DataLoader(inputs_test, batch_size=hp.batch_size, num_workers=0)

	# It's time to roll baby!
	for epoch in range(hp.n_epochs):
		# Run training loop
		losses, skill_lens = run_iteration(model, train_loader, hp, model_optimizer=model_optimizer, train_mode=True)

		# Update logger
		logger.update(epoch, losses)
		logger.add_length_histogram(epoch, skill_lens)

		# Run evaluation loop
		with torch.no_grad():
			test_losses, skill_lens = run_iteration(model, test_loader, hp, train_mode=False)

		# Update logger
		logger.update(epoch, test_losses, mode='test')
		logger.add_length_histogram(epoch, skill_lens, mode='test')

		# Periodically save training
		if epoch % 10 == 0:
			logger.save_training_state(epoch, model, model_optimizer, 'latest.pth')

		# Save on hitting new best
		if test_losses['tot_loss'] < logger.min_test_loss:
			logger.min_test_loss = test_losses['tot_loss']
			logger.save_training_state(epoch, model, model_optimizer, 'best.pth')