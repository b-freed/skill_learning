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


def run_iteration(model, data_loader, experiment, model_optimizer=None, train_mode=False):
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	sl_losses = []
	s_T_ents = []

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
		z_t, z_post_means, z_post_sigs, s0, _, _, skill_steps = model.update_skills(z_post_means_update, z_post_sigs_update, b_post_means, b_post_sigs, states, greedy=False)
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

		loss_tot = model.alpha*s_T_loss + a_loss + model.beta*kl_loss + model.ent_pen*s_T_ent

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
		# sl_losses.append(sl_loss.item())
		sl_losses.append(0)
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())


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
					min_skill_len=hp.min_skill_len, max_skill_len=hp.max_skill_len, \
                    max_skills_per_seq=hp.max_skills_per_seq, device=hp.device).to(hp.device)
	else:
		model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
		
	model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

	min_test_loss = 10**10

	for i in range(hp.n_epochs):
		loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss = run_iteration(model, train_loader, experiment, model_optimizer=model_optimizer, train_mode=True)
		
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
			test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss = run_iteration(model, test_loader, experiment, train_mode=False)
		
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