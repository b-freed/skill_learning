import os 
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


def run_iteration(i, model, data_loader, model_optimizer=None, train_mode=False):
	losses = []
	s_T_losses = []
	a_losses = []
	b_kl_losses = []
	z_kl_losses = []
	sl_losses = []
	s_T_ents = []

	sl_means = []
	sl_stds = []
	sl_mins = []
	sl_maxs = []

	ns_means = []
	ns_stds = []
	ns_mins = []
	ns_maxs = []

	model.train() if train_mode else model.eval()
	train_phase = get_train_phase(i)

	for batch_id, data_ in tqdm.tqdm(enumerate(data_loader), desc=f"Progress"):
		data_ = data_.to(model.device) # (batch_size, 1000, state_dim + action_dim)
		states, actions = data_[:, :, :state_dim], data_[:, :, state_dim:]

		# Infer skills and terminations for the data
		z_post_means, z_post_sigs, b_post = model.encoder(states, actions)

		# Sequentially update z_t using termination events
		z_t, s0, sT_gt, read, b_post_sampled_logits, _termination_loss, _sl_loss, n_executed_skills, skill_lens_data, \
			n_executed_skills_data = model.update_skillsv2(z_post_means, z_post_sigs, b_post, states, train_phase=train_phase)

		# Use skills and states to generate action and terminal state sequence
		sT_mean, sT_sig, a_means, a_sigs = model.decoder(states, z_t, s0)

		# Compute skill and termination prior
		z_prior_means, z_prior_sigs = model.prior(s0)
		if train_phase == 0:
			b_prior = b_post.clone()
		elif train_phase == 1:
			b_prior = model.termination_prior(states, actions) # TODO: these inputs for termination prior?

		# Regularize termination prior (if train phase is not zero)
		if train_phase == 1: b_prior, b_prior_sampled_logits = model.regularize_termination(b_prior)
		read_prior = b_prior[..., -1:]

		# Construct required distributions
		sT_dist = Normal.Normal(sT_mean, sT_sig)
		z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
		z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)
		# b_post_dist = torch.distributions.Bernoulli(read) # TODO: issue with finding KL only over its support set
		# b_prior_dist = torch.distributions.Bernoulli(read_prior)

		if model.decoder.ll_policy.a_dist == 'normal':
			a_dist = Normal.Normal(a_means, a_sigs)
		elif model.decoder.ll_policy.a_dist == 'tanh_normal':
			base_dist = Normal.Normal(a_means, a_sigs)
			transform = torch.distributions.transforms.TanhTransform()
			a_dist = TransformedDistribution(base_dist, [transform])
		else:
			assert False, f'{model.decoder.ll_policy.a_dist} not supported'

		# Loss for predicting terminal state
		s_T_loss = -torch.mean(torch.sum(sT_dist.log_prob(sT_gt), dim=-1) * read.squeeze()) # mean over time steps and batch

		# Entropy corresponding to terminal state
		# s_T_ent = torch.mean(torch.sum(s_T_dist.entropy(), dim=-1))/time_steps
		s_T_ent = 0

		# Los for predicting actions
		a_loss = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) # mean over time steps and batch

		# KL divergence between skill prior and posterior
		z_kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1) * read.detach().squeeze()) # mean over time steps and batch

		# KL divergence between termiation prior and posterior
		# TODO: decide a method for computing bernoulli kl divergence
		# b_kl_loss = torch.mean(torch.sum(KL.kl_divergence(b_post_dist, b_prior_dist), dim=-1)) # mean over time steps and batch
		# b_post_log_density = model.log_density_concrete(b_prior, b_post_sampled_logits)
		# b_prior_log_density = model.log_density_concrete(b_post, b_post_sampled_logits)
		# b_kl_loss = torch.mean(b_post_log_density - b_prior_log_density) # mean over time steps and batch
		b_kl_loss = torch.mean(utils.kl_divergence_bernoulli(b_post, b_prior)) if train_phase == 1 else torch.zeros(1, device=model.device) # mean over time steps and batch

		# Loss for number of terminations
		termination_loss = torch.mean(_termination_loss) # mean over batch

		# Loss for sticking with the same skills
		skill_len_loss = torch.mean(_sl_loss) # mean over batch

		loss_tot = model.alpha * s_T_loss + a_loss + model.beta * (z_kl_loss + b_kl_loss) + model.gamma * skill_len_loss

		if train_mode:
			model_optimizer.zero_grad()
			loss_tot.backward()
			model.clip_gradients()
			model_optimizer.step()

		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		# s_T_ents.append(s_T_ent.item())
		s_T_ents.append(0)
		sl_losses.append(skill_len_loss.item())
		a_losses.append(a_loss.item())
		b_kl_losses.append(b_kl_loss.item())
		z_kl_losses.append(z_kl_loss.item())
		sl_means.append(skill_lens_data['mean'])
		sl_stds.append(skill_lens_data['std'])
		sl_mins.append(skill_lens_data['min'])
		sl_maxs.append(skill_lens_data['max'])
		ns_means.append(n_executed_skills_data['mean'])
		ns_stds.append(n_executed_skills_data['std'])
		ns_mins.append(n_executed_skills_data['min'])
		ns_maxs.append(n_executed_skills_data['max'])

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(b_kl_losses), np.mean(z_kl_losses), np.mean(s_T_ents), np.mean(sl_losses), np.mean(sl_means), np.mean(sl_stds), np.mean(sl_mins), np.mean(sl_maxs), np.mean(ns_means), np.mean(ns_stds), np.mean(ns_mins), np.mean(ns_maxs)


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
	raw_data = np.load(os.path.join(hp.data_dir, f'{hp.env_name}.npz'))
	inputs_train = torch.from_numpy(raw_data['inputs_train']).reshape(-1, 20, 37)
	inputs_test  = torch.from_numpy(raw_data['inputs_test']).reshape(-1, 20, 37)

	train_loader = DataLoader(
		inputs_train,
		batch_size=hp.batch_size,
		num_workers=0)  # not really sure about num_workers...

	test_loader = DataLoader(
		inputs_test,
		batch_size=hp.batch_size,
		num_workers=0)

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
		# Run training loop
		loss, s_T_loss, a_loss, b_kl_loss, z_kl_loss, s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, \
			ns_mean, ns_std, ns_min, ns_max = run_iteration(i, model, train_loader, model_optimizer=model_optimizer, train_mode=True)

		# Update logger
		logger.update_train(i, loss, s_T_loss, a_loss, b_kl_loss, z_kl_loss, s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max)
		
		# Run evaluation loop
		with torch.no_grad():
			test_loss, test_s_T_loss, test_a_loss, test_b_kl_loss, test_z_kl_loss, test_s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, \
			ns_mean, ns_std, ns_min, ns_max = run_iteration(i, model, test_loader, train_mode=False)

		# Update logger
		logger.update_test(i, test_loss, test_s_T_loss, test_a_loss, test_b_kl_loss, test_z_kl_loss, test_s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max)
		
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
