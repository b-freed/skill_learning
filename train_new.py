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
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillModelTerminalStateDependentPrior, SkillModelStateDependentPriorAutoTermination
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
from configs import HyperParams
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.distributions.kl as KL


def get_train_phase(i):
	"""
	For initial n iteration, don't train termination model. Afterwards, train both the modules. 
	"""
	phase_zero_iterations = 10000
	if i < phase_zero_iterations:
		return 0
	else: 
		return 1


def get_confusion_matrix(prediction, gt):
	"""
	Finds the confusion matrix for a given prediction and ground truth.
	"""
	confusion_matrix = torch.zeros((2, 2))
	TP = torch.sum(prediction[gt == 1])
	FP = torch.sum(prediction[gt == 0])
	TN = torch.sum(torch.logical_not(prediction[gt == 0]))
	FN = torch.sum(torch.logical_not(prediction[gt == 1]))
	confusion_matrix = (1/gt.numel()) * torch.tensor([[TP, FP], [FN, TN]])
	return confusion_matrix


def run_iteration(i, model, data_loader, model_optimizer=None, train_mode=False):
	losses = []
	s_T_losses = []
	a_losses = []
	z_kl_losses = []
	b_losses = []
	s_T_ents = []

	device = model.device
	horizon_length = model.max_skill_len

	model.train() if train_mode else model.eval()
	train_phase = get_train_phase(i)

	b_criterion = torch.nn.BCELoss(reduction='none')
	get_masked_loss = lambda raw_loss, data_lens_tensor, mask, bs: (1/bs)*torch.sum(torch.sum(torch.sum(raw_loss, dim=-1) * mask, dim=-1)/data_lens_tensor)

	for batch_id, (data_, data_lens) in tqdm.tqdm(enumerate(data_loader), desc=f"Progress"):
		batch_size = len(data_)
		data_lens_tensor = torch.tensor(data_lens, device=device)

		loss_mask = torch.arange(horizon_length, device=device).repeat(batch_size, 1)
		loss_mask = loss_mask < data_lens_tensor.unsqueeze(dim=-1)

		b_gt = torch.zeros(batch_size, horizon_length).to(device).float()
		b_gt[torch.arange(batch_size, device=device), data_lens_tensor-1] = 1
		b_loss_weight = torch.zeros_like(b_gt)

		# b_loss_weight[:, :] = 0.5 / (data_lens_tensor - 1)
		b_loss_weight = 0.5 / (data_lens_tensor.unsqueeze(dim=-1).repeat(1, horizon_length) - 1)
		b_loss_weight[torch.arange(batch_size, device=device), data_lens_tensor-1] = 0.5

		data_ = data_.to(device) # (batch_size, 1000, state_dim + action_dim)
		states, actions = data_[:, :, :state_dim], data_[:, :, state_dim:]

		# Infer skills for the data
		z_post_means, z_post_sigs = model.encoder(states, actions, unequal_subtraj_lens=True, data_lens=data_lens)

		# sample z_ts from the posterior
		z_t = model.reparameterize(z_post_means, z_post_sigs)

		# Extract s0 and sT ground truths from the data sequence
		s0 = states[:, :1, :] # (batch_size, 1, state_dim)
		sT_gt = states[torch.arange(batch_size, device=device), data_lens_tensor-1, :].unsqueeze(dim=1) # (batch_size, 1, state_dim)

		# Use skills and states to generate terminal state and action sequences
		sT_mean, sT_sig, a_means, a_sigs, b_ = model.decoder(states, z_t, s0)

		b = b_.squeeze(2) if b_.ndim == 3 else b_

		# Compute skill and termination prior
		z_prior_means, z_prior_sigs = model.prior(s0)

		# Construct required distributions
		sT_dist = Normal.Normal(sT_mean, sT_sig)
		z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)

		if model.decoder.ll_policy.a_dist == 'normal':
			a_dist = Normal.Normal(a_means, a_sigs)
		elif model.decoder.ll_policy.a_dist == 'tanh_normal':
			base_dist = Normal.Normal(a_means, a_sigs)
			transform = torch.distributions.transforms.TanhTransform()
			a_dist = TransformedDistribution(base_dist, [transform])
		else:
			assert False, f'{model.decoder.ll_policy.a_dist} not supported'

		# Loss for predicting terminal state # mean over time steps and batch
		s_T_loss = -get_masked_loss(sT_dist.log_prob(sT_gt), data_lens_tensor, loss_mask, batch_size)

		# Entropy corresponding to terminal state
		# s_T_ent = torch.mean(torch.sum(s_T_dist.entropy(), dim=-1))/time_steps
		s_T_ent = 0

		# Los for predicting actions # mean over time steps and batch 
		# a_loss = -(1/batch_size)*torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1))
		a_loss = -get_masked_loss(a_dist.log_prob(actions), data_lens_tensor, loss_mask, batch_size)

		# KL divergence between skill prior and posterior # mean over time steps and batch
		# z_kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1)) # mean over time steps and batch
		z_kl_loss = get_masked_loss(KL.kl_divergence(z_post_dist, z_prior_dist), data_lens_tensor, loss_mask, batch_size)

		# Loss for prediction termination condition
		# Termination encoder should learn to terminate on reaching sT_gt when running the planned skill		
		b_loss = (1/batch_size)*torch.sum(torch.sum(b_criterion(b, b_gt) * b_loss_weight, dim=-1)/data_lens_tensor, dim=-1)

		loss_tot = model.alpha * s_T_loss + a_loss + model.beta * z_kl_loss + model.gamma * b_loss

		if train_mode:
			model_optimizer.zero_grad()
			loss_tot.backward()
			model.clip_gradients()
			model_optimizer.step()

		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		# s_T_ents.append(s_T_ent.item())
		s_T_ents.append(0)
		a_losses.append(a_loss.item())
		z_kl_losses.append(z_kl_loss.item())
		b_losses.append(b_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(b_losses), np.mean(z_kl_losses), np.mean(s_T_ents)


if __name__ == '__main__':
	hp = HyperParams()

	# set environment variable for device
	os.environ["_DEVICE"] = hp.device

	logger = Logger(hp)

	# TODO: might want to get rid of this
	env = gym.make(hp.env_name)
	state_dim = env.observation_space.shape[0]
	a_dim = env.action_space.shape[0]

	# Load the dataset files
	# raw_data = np.load(os.path.join(hp.data_dir, f'{hp.env_name}.npz'))
	# inputs_train = torch.from_numpy(raw_data['inputs_train']).reshape(-1, 20, 37)
	# inputs_test  = torch.from_numpy(raw_data['inputs_test']).reshape(-1, 20, 37)
	data_path = os.path.join(hp.data_dir, f'{hp.env_name}.npz')
	inputs_train = utils.UniformRandomSubTrajectory(data_path, train=True, min_len=hp.min_skill_len, max_len=hp.max_skill_len)
	inputs_test = utils.UniformRandomSubTrajectory(data_path, train=False, min_len=hp.min_skill_len, max_len=hp.max_skill_len)

	# Instantiate a skill model
	if hp.term_state_dependent_prior:
		model = SkillModelTerminalStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, \
					state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
					fixed_sig=hp.fixed_sig).to(hp.device)
	elif hp.state_dependent_prior:
		model = SkillModelStateDependentPriorAutoTermination(state_dim, a_dim, hp.z_dim, hp.h_dim, \
					a_dist=hp.a_dist, state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
					temperature=hp.temperature, gamma=hp.gamma, max_sig=hp.max_sig, fixed_sig=hp.fixed_sig, \
					ent_pen=hp.ent_pen, encoder_type=hp.encoder_type, state_decoder_type=hp.state_decoder_type, \
					min_skill_len=hp.min_skill_len, grad_clip_threshold=hp.grad_clip_threshold, \
					max_skill_len=hp.max_skill_len, max_skills_per_seq=hp.max_skills_per_seq, device=hp.device).to(hp.device)
	else:
		model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
		
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	# It's time to roll baby!
	for i in range(hp.n_epochs):
		# Resample subtrajectories
		inputs_train.sample_random_len_sequences()
		inputs_test.sample_random_len_sequences()

		train_loader = DataLoader(inputs_train,
									batch_size=hp.batch_size,
									collate_fn=utils.pad_collate_custom,
									num_workers=0)

		test_loader = DataLoader(inputs_test,
								batch_size=hp.batch_size,
								collate_fn=utils.pad_collate_custom,
								num_workers=0)

		# Run training loop
		loss, s_T_loss, a_loss, b_loss, z_kl_loss, s_T_ent = run_iteration(i, model, train_loader, model_optimizer=model_optimizer, train_mode=True)

		# Update logger
		logger.update_train(i, loss, s_T_loss, a_loss, b_loss, z_kl_loss, s_T_ent)
		
		# Run evaluation loop
		with torch.no_grad():
			test_loss, test_s_T_loss, test_a_loss, test_b_loss, test_z_kl_loss, test_s_T_ent = run_iteration(i, model, test_loader, train_mode=False)

		# Update logger
		logger.update_test(i, test_loss, test_s_T_loss, test_a_loss, test_b_loss, test_z_kl_loss, test_s_T_ent)
		
		# Periodically save training
		if i % 10 == 0:
			logger.save_training_state(i, model, model_optimizer, 'latest.pth')

		# Save on hitting new best
		if test_loss < logger.min_test_loss:
			logger.min_test_loss = test_loss
			logger.save_training_state(i, model, model_optimizer, 'best.pth')

		# Anneal temperature for relaxed distributions
		if hp.temperature_anneal:
			model.temperature = (hp.max_temperature - hp.min_temperature) * 0.999 ** (i / hp.temperature_anneal) + hp.min_temperature
		else:
			model.temperature = hp.max_temperature
