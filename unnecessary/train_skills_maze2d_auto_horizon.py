import os 
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
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
import tqdm
import os
from configs import HyperParams
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def prepare_data(data, start_idxs, end_idxs, s_dim, a_dim=None):
	"""
	Slice entire data tensor -> list of tensors w/ unequal lengths -> (pad -> packed seq) data that can be parallely 
	processed by the model (LSTM/GRU cells).
	"""
	# Slice the data according to the start and end indices
	total_dim = data.shape[-1]

	seq_lens = (end_idxs - start_idxs).tolist()

	if a_dim is None: a_dim = total_dim - s_dim

	max_l = torch.max(end_idxs - start_idxs)
	
	# Create buffer_tensors
	# End idx is same across the batch
	end_idx = end_idxs[0]
	padded_states = data[:, end_idx - max_l:end_idx, :state_dim].clone() # w/o clone, making might destroy the og data
	padded_actions = data[:, end_idx - max_l:end_idx, state_dim:].clone()

	# create mask ()
	# data_dim = batch_size, seq_len, state_dim + action_dim
	idx_mask = torch.arange(end_idx - max_l, end_idx).unsqueeze(dim=0).expand(data.shape[0], -1).to(hp.device) < end_idxs.unsqueeze(dim=1)
	mask = torch.logical_not(idx_mask) # To mask idxs

	padded_states[mask, :] = 0
	padded_actions[mask, :] = 0

	return padded_states, padded_actions, seq_lens, mask


def run_iteration2(model, data_loader, model_optimizer=None, train_mode=False):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	sl_losses = []
	s_T_ents = []

	model.train() if train_mode else model.eval()

	for batch_id, data_ in tqdm.tqdm(enumerate(data_loader), desc=f"Progress"):
		# Now, we have a batch of trajectories, i.e., dim(data_) = (batch_size, 1000, state_dim + action_dim)
		data_ = data_.to(hp.device)

		# Reset required running variables
		z_history = []
		running_l = torch.zeros(hp.batch_size)
		executed_skills = torch.zeros(hp.batch_size)
		start_idxs = torch.zeros(hp.batch_size, dtype=torch.int).to(hp.device)
		loss_tot, s_T_loss, a_loss, kl_loss, sl_loss, s_T_ent = 0, 0, 0, 0, 0, 0

		for i in range(1, data_.shape[1]):
			'''
			i is the end index of all trajectories in the sampled batch. We keep track of the start index, 
			updating it on termination events. Now, all trajectories in a batch will have different lengths.
			'''
			end_idxs = i * torch.ones(hp.batch_size, dtype=torch.int).to(hp.device)
			padded_states, padded_actions, seq_lens, mask = prepare_data(data_, start_idxs, end_idxs, model.state_dim)

			H = (end_idxs - start_idxs).float()

			# states, actions = pack_padded_seq[:, :, :model.state_dim], pack_padded_seq[:, :, model.state_dim:] # first state_dim elements are the state, rest are actions
			s_0 = data_[torch.arange(data_.shape[0]), start_idxs.long(), :model.state_dim].unsqueeze(dim=1)
			# s_0 = states[:, 0:1, :]
			s_T_gt = data_[torch.arange(data_.shape[0]), end_idxs.long(), :model.state_dim].unsqueeze(dim=1) # TODO: This isn't the right way to do it, but it's fine for now. don't IGNORE.

			# Infer skills from the sequence 
			z_post_means, z_post_sigs, b_post_means, b_post_sigs = model.encoder(padded_states, padded_actions, seq_lens, variable_length=True)
			# Sample a skill from the posterior
			z_t = model.reparameterize(z_post_means, z_post_sigs)
			# Choose if we want to update the skill
			z_t, b_t = model.update_skills(z_history, z_t, b_post_means, b_post_sigs, running_l, executed_skills, greedy=(not train_mode))

			z_history.append(z_t)

			# TODO: We want to train s_T only against the true terminal state. Rn, ignore and just train (since there's no grad propogating from abstract dynamics model to the skill model).
			s_T_mean, s_T_sig, a_means, a_sigs = model.decoder(padded_states, z_t)

			# Now we want to mass the stuff from !mask

			# Compute the losses
			loss_tot_, s_T_loss_, a_loss_, kl_loss_, s_T_ent_ = model.compute_losses(s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs, s_0, H, s_T_gt, padded_actions, mask) 
			sl_tracker = b_t[:, 0] # loss for sticking w/ skills 
			sl_loss_ = model.gamma * sl_tracker.sum()
			loss_tot_ += sl_loss_

			# Update running loss
			loss_tot += loss_tot_
			s_T_loss += s_T_loss_
			a_loss += a_loss_
			kl_loss += kl_loss_
			sl_loss += sl_loss_
			s_T_ent += s_T_ent_

			# Update length running trackers
			update = b_t[:, 1].bool()
			executed_skills[update] += 1
			running_l[update] = 1 
			running_l[~update] += 1

			# Update start index
			start_idxs[update] = i

		if train_mode:
			model_optimizer.zero_grad()
			loss_tot.backward()
			# TODO: clip gradients
			model.clip_gradients()
			model_optimizer.step()

		# log losses
		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())
		s_T_ents.append(s_T_ent.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents), np.mean(sl_losses)


# Seems correct # TODO remove this comment
if __name__ == '__main__':
	hp = HyperParams()

	# set environment variable for device
	os.environ["_DEVICE"] = hp.device

	experiment = Experiment(api_key = 'Wlh5wstMNYkxV0yWRxN7JXZRu', project_name = 'dump')
	experiment.set_name(hp.exp_name)

	experiment.log_parameters(hp.__dict__)
	# save hyperparams locally
	os.makedirs(hp.log_dir, exist_ok=True) # safely create log dir
	os.system(f'cp configs.py {hp.log_dir}')

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
					min_skill_len=hp.min_skill_len, max_skill_len=hp.max_skill_len, max_skills_per_seq=hp.max_skills_per_seq).to(hp.device)
	else:
		model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
		
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	min_test_loss = 10**10

	for i in range(hp.n_epochs):
		loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss = run_iteration2(model, train_loader, model_optimizer=model_optimizer, train_mode=True)
		
		print(f'Exp: {hp.exp_name} | Iter: {i}')
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
			test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss = run_iteration2(model, test_loader, train_mode=False)
		
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