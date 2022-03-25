from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillModelTerminalStateDependentPrior
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
import utils


def train(model, model_optimizer):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]  # first state_dim elements are the state
		actions = data[:,:,model.state_dim:]	 # rest are actions

		loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent = model.get_losses(states, actions)

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

def test(model):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	with torch.no_grad():
		for batch_id, data in enumerate(test_loader):
			data = data.cuda()
			states = data[:,:,:model.state_dim]  # first state_dim elements are the state
			actions = data[:,:,model.state_dim:]	 # rest are actions

			loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent  = model.get_losses(states, actions)

			# log losses
			losses.append(loss_tot.item())
			s_T_losses.append(s_T_loss.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())
			s_T_ents.append(s_T_ent.item())


	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents)


class HyperParams:
    def __init__(self):
        self.batch_size = 100
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
        self.H = 20
        self.stride = 1
        self.n_epochs = 50000
        self.test_split = .2
        self.a_dist = 'normal' # 'tanh_normal' or 'normal'
        self.encoder_type = 'state_action_sequence' #'state_sequence'
        self.state_decoder_type = 'autoregressive'
        self.env_name = 'antmaze-large-diverse-v0'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.term_state_dependent_prior:
            self.filename = f'{self.env_name}_tsdp_H{self.H}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_log'
        else:
            self.filename = f'{self.env_name}_enc_type_{self.encoder_type}_state_dec_{self.state_decoder_type}_H_{self.H}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_ent_pen_{self.ent_pen}_log'


hp = HyperParams()

dataset = utils.create_dataset_padded(utils.create_dataset_raw, hp.env_name)

states = dataset['observations']
actions = dataset['actions']
goals = dataset['infos/goal']

N_episodes = states.shape[0]
state_dim = states.shape[-1]
a_dim = actions.shape[-1]

N_train = int((1 - hp.test_split) * N_episodes)
N_test = N_episodes - N_train

states_train  = states[:N_train, ...]
actions_train = actions[:N_train, ...]
goals_train = goals[:N_train, ...]

states_test  = states[N_train:,:]
actions_test = actions[N_train:,:]
goals_test   = goals[N_train:,:]

assert states_train.shape[0] == actions_train.shape[0] == goals_train.shape[0] == N_train
assert states_test.shape[0] == actions_test.shape[0] == goals_test.shape[0] == N_test

# obs_chunks_train, action_chunks_train, targets_train = chunks(states_train, actions_train, goals_train, H, stride)
# obs_chunks_test,  action_chunks_test,  targets_test  = chunks(states_test,  actions_test,  goals_test,  H, stride)

experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
# experiment.add_tag('noisy2')

# First, instantiate a skill model
if hp.term_state_dependent_prior:
	model = SkillModelTerminalStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, state_dec_stop_grad=hp.state_dec_stop_grad,beta=hp.beta,alpha=hp.alpha,fixed_sig=hp.fixed_sig).to(hp.device)
elif hp.state_dependent_prior:
	model = SkillModelStateDependentPrior(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist,state_dec_stop_grad=hp.state_dec_stop_grad,beta=hp.beta,alpha=hp.alpha,max_sig=hp.max_sig,fixed_sig=hp.fixed_sig,ent_pen=hp.ent_pen,encoder_type=hp.encoder_type,state_decoder_type=hp.state_decoder_type).to(hp.device)
else:
	model = SkillModel(state_dim, a_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist).to(hp.device)
	
model_optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

experiment.log_parameters(hp.__dict__)

# add chunks of data to a pytorch dataloader
inputs_train = torch.tensor(np.concatenate([obs_chunks_train, action_chunks_train],axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim
inputs_test  = torch.tensor(np.concatenate([obs_chunks_test,  action_chunks_test], axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim


train_loader = DataLoader(
	inputs_train,
	batch_size=hp.batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=hp.batch_size,
	num_workers=0)

min_test_loss = 10**10
for i in range(hp.n_epochs):
	loss, s_T_loss, a_loss, kl_loss, s_T_ent = train(model,model_optimizer)
	
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


	test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent = test(model)
	
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
		torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
	if test_loss < min_test_loss:
		min_test_loss = test_loss

		
			
		checkpoint_path = 'checkpoints/'+ hp.filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
			    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)
