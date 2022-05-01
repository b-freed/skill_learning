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


def run_iteration(model, data_loader, model_optimizer=None, train_mode=False):
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
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

	model.train()

	for batch_id, data_ in tqdm.tqdm(enumerate(data_loader), desc=f"Progress"):
		data_ = data_.to(model.device) # (batch_size, 1000, state_dim + action_dim)
		states, actions = data_[:, :, :state_dim], data_[:, :, state_dim:]

		time_steps = states.shape[1]

		z_post_means_update, z_post_sigs_update, b_post_means, b_post_sigs = model.encoder(states, actions)

		# # (differentiably) sample skills 
		# z_t_ = model.reparameterize(z_post_means, z_post_sigs)

		# Actually find the skills to be executed, i.e., use the termination events
		# z_t, _, _ = model.update_skills(z_t_, b_post_means, b_post_sigs, greedy=False)
		z_t, z_post_means, z_post_sigs, s0, _termination_loss, _, skill_steps, skill_lens_data, n_executed_skills_data = model.update_skills(z_post_means_update, z_post_sigs_update, b_post_means, b_post_sigs, states, greedy=False)
		skill_steps = skill_steps.detach()
		# # z_t, b_t = model.update_skills(z_history, z_t, b_post_means, b_post_sigs, running_l, executed_skills, greedy=(not train_mode))

		# Pass through the action and terminal state generators (decoders)
		a_means, a_sigs = model.decoder(states, z_t)

		# Compute losses
		# # compute skill prior
		# Assemble s0 and z_executed
		z_prior_means, z_prior_sigs = model.prior(s0)

		# # # Construct required distributions
		# s_T_dist = Normal.Normal(s_T_mean, s_T_sig)
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

		# Loss for predicting terminal state
		# s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T_gt), dim=-1)/skill_steps) # divide each skill by its length and take mean across batch
		s_T_loss = 0
		# Entropy corresponding to terminal state
		# s_T_ent = torch.mean(torch.sum(s_T_dist.entropy(), dim=-1))/time_steps
		s_T_ent = 0
		# Los for predicting actions
		a_loss = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) 
		# Compute KL Divergence between prior and posterior
		kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1)/skill_steps) # divide each skill by its length and take mean across batch
		# Incentive for sticking with skills
		termination_loss = torch.mean(_termination_loss)
		loss_tot = model.alpha*s_T_loss + a_loss + model.beta*kl_loss + model.ent_pen*s_T_ent + model.gamma * termination_loss

		# return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent
		if train_mode:
			model_optimizer.zero_grad()
			loss_tot.backward()
			# TODO: clip gradients
			model.clip_gradients()
			model_optimizer.step()

		losses.append(loss_tot.item())
		# s_T_losses.append(s_T_loss.item())
		s_T_losses.append(0)
		# s_T_ents.append(s_T_ent.item())
		s_T_ents.append(0)
		sl_losses.append(termination_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())
		sl_means.append(skill_lens_data['mean'])
		sl_stds.append(skill_lens_data['std'])
		sl_mins.append(skill_lens_data['min'])
		sl_maxs.append(skill_lens_data['max'])
		ns_means.append(n_executed_skills_data['mean'])
		ns_stds.append(n_executed_skills_data['std'])
		ns_mins.append(n_executed_skills_data['min'])
		ns_maxs.append(n_executed_skills_data['max'])


	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents), np.mean(sl_losses), np.mean(sl_means), np.mean(sl_stds), np.mean(sl_mins), np.mean(sl_maxs), np.mean(ns_means), np.mean(ns_stds), np.mean(ns_mins), np.mean(ns_maxs)


# Seems correct # TODO remove this comment
if __name__ == '__main__':
	hp = HyperParams()

	# set environment variable for device
	os.environ["_DEVICE"] = hp.device

	logger = Logger(hp)

	# might want to get rid of this
	env = gym.make(hp.env_name)
	state_dim = env.observation_space.shape[0]
	a_dim = env.action_space.shape[0]

	# Now, only load the npz files
	raw_data = np.load(os.path.join(hp.data_dir, f'{hp.env_name}.npz'))
	inputs_train = torch.from_numpy(raw_data['inputs_train'])
	inputs_test  = torch.from_numpy(raw_data['inputs_test'])

	train_loader = DataLoader(
		inputs_train,
		batch_size=hp.batch_size,
		num_workers=0)  # not really sure about num_workers...

	test_loader = DataLoader(
		inputs_test,
		batch_size=hp.batch_size,
		num_workers=0)

	# First, instantiate a skill model
	if hp.term_state_dependent_prior:
		model = SkillModelTerminalStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, \
					state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
					fixed_sig=hp.fixed_sig).to(hp.device)
	elif hp.state_dependent_prior:
		model = SkillModelStateDependentPriorAutoTermination(state_dim, a_dim, hp.z_dim, hp.h_dim, \
					a_dist=hp.a_dist, state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha, \
					temperature=hp.temperature, gamma=hp.gamma, max_sig=hp.max_sig, fixed_sig=hp.fixed_sig, \
					ent_pen=hp.ent_pen, encoder_type=hp.encoder_type, state_decoder_type=hp.state_decoder_type, \
					min_skill_len=hp.min_skill_len, max_skill_len=hp.max_skill_len, \
                    max_skills_per_seq=hp.max_skills_per_seq, device=hp.device).to(hp.device)
	else:
		model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
		
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	for i in range(hp.n_epochs):
		loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, \
			ns_mean, ns_std, ns_min, ns_max = run_iteration(model, train_loader, model_optimizer=model_optimizer, train_mode=True)

		logger.update_train(i, loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max)
		
		with torch.no_grad():
			test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, \
			ns_mean, ns_std, ns_min, ns_max = run_iteration(model, test_loader, train_mode=False)

		logger.update_test(i, test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max)
		
		if i % 10 == 0:
			logger.save_training_state(i, model, model_optimizer, 'latest.pth')

		if test_loss < logger.min_test_loss:
			logger.min_test_loss = test_loss
			logger.save_training_state(i, model, model_optimizer, 'best.pth')