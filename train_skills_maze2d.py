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
from utils import chunks
import os
import config

def train(model,model_optimizer):
	
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


# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
# env_name = 'antmaze-medium-diverse-v0'  
# env_name = 'maze2d-large-v1'
env_name = 'antmaze-large-diverse-v0'

dataset_file = config.dataset_file
#dataset_file = "datasets/maze2d-umaze-v1.hdf5"
# dataset_file = 'datasets/maze2d-large-v1-noisy-2.hdf5'

if dataset_file is None:
	env = gym.make(env_name)
	dataset = d4rl.qlearning_dataset(env) #env.get_dataset()
else:
	print('LOADING DATASET FROM FILE!!!')
	data_path = os.path.join(config.data_dir,dataset_file)
	if '.npz' in dataset_file:
		# load numpy file
		pass # TODO
	elif '.hdf5' in dataset_file:
		env = gym.make(env_name)
		dataset = d4rl.qlearning_dataset(env,h5py.File(dataset_file, "r"))  # Not sure if this will work

	else:
		print('Unrecognized data format!!!')
		assert False

batch_size = 100
h_dim = 256
z_dim = 256
lr = 5e-5
wd = 0.0
state_dependent_prior = True
term_state_dependent_prior = False
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0.0
max_sig = None
fixed_sig = None
H = 40
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'normal' # 'tanh_normal' or 'normal'
encoder_type = 'state_action_sequence' #'state_sequence'
state_decoder_type = 'mlp' #'autoregressive'
			



states = dataset['observations']
next_states = dataset['next_observations']
actions = dataset['actions']

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

states_train  = states[:N_train,:]
next_states_train = next_states[:N_train,:]
actions_train = actions[:N_train,:]


states_test  = states[N_train:,:]
next_states_test = next_states[N_train:,:]
actions_test = actions[N_train:,:]

assert states_train.shape[0] == N_train
assert states_test.shape[0] == N_test
										            #  obs,next_obs,actions,H,stride
obs_chunks_train, action_chunks_train = chunks(states_train, next_states_train, actions_train, H, stride)

print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')

obs_chunks_test,  action_chunks_test  = chunks(states_test,  next_states_test,  actions_test,  H, stride)


experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
# experiment.add_tag('noisy2')


# First, instantiate a skill model

if term_state_dependent_prior:
	model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,state_decoder_type=state_decoder_type).cuda()

else:
	model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
	
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

filename = env_name+'_enc_type_'+str(encoder_type)+'state_dec_'+str(state_decoder_type)+'_H_'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_sg_'+str(state_dec_stop_grad)+'_max_sig_'+str(max_sig)+'_fixed_sig_'+str(fixed_sig)+'_ent_pen_'+str(ent_pen)+'_log'



if term_state_dependent_prior:
	filename = env_name+'_tsdp'+'_H'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_sg_'+str(state_dec_stop_grad)+'_max_sig_'+str(max_sig)+'_fixed_sig_'+str(fixed_sig)+'_log'


experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'state_dependent_prior':state_dependent_prior,
							'term_state_dependent_prior':term_state_dependent_prior,
							'z_dim':z_dim,
							'H':H,
							'a_dist':a_dist,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'state_dec_stop_grad':state_dec_stop_grad,
							'beta':beta,
							'alpha':alpha,
							'max_sig':max_sig,
							'fixed_sig':fixed_sig,
							'ent_pen':ent_pen,
							'env_name':env_name,
							'filename':filename,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type})
experiment.add_tag('fixed chunks')

# add chunks of data to a pytorch dataloader
# inputs_train = torch.tensor(np.concatenate([obs_chunks_train, action_chunks_train],axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim
# inputs_test  = torch.tensor(np.concatenate([obs_chunks_test,  action_chunks_test], axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim
inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

# dataset_size = len(inputs)
# train_data, test_data = torch.utils.data.random_split(inputs, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])
# train_targets, test_targets = torch.utils.data.random_split(targets, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])



# train_data = TensorDataset(torch.tensor(inputs_train, dtype=torch.float32))
# test_data  = TensorDataset(torch.tensor(inputs_test,  dtype=torch.float32))

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
for i in range(n_epochs):
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
		
			
		checkpoint_path = 'checkpoints/'+ filename + '.pth'
		torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
	if test_loss < min_test_loss:
		min_test_loss = test_loss

		
			
		checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
			    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)

	
	if test_s_T_loss < min_test_s_T_loss:
		min_test_s_T_loss = test_s_T_loss

		checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best_sT.pth')
		torch.save({'model_state_dict': model.state_dict(),
			    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)