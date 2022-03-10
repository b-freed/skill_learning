from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py

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
env_name = 'antmaze-medium-diverse-v0'  # maze whatever
# env_name = 'maze2d-large-v1'
# env_name = 'antmaze-large-diverse-v0'
env = gym.make(env_name)
# dataset = env.get_dataset()  # dictionary, with 'observations', 'rewards', 'actions', 'infos/goal'
dataset_file = None
#dataset_file = "datasets/maze2d-umaze-v1.hdf5"
# dataset_file = 'datasets/maze2d-large-v1-noisy-2.hdf5'

if dataset_file is None:
	dataset = env.get_dataset()
else:
	dataset = h5py.File(dataset_file, "r")


batch_size = 100



h_dim = 256
z_dim = 256
lr = 5e-5
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = False
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

# splitting up the dataset into subsequences in which we're going to a particular goal.  Every time the goal changes we make a new subsequence.
# chunks might not all be same length, might have to split long chunks down into sub-chunks, discarding leftover chunks that are shorter than our chunck length.
# so if I have a chunk that's 125 long, I can split into 6 x 20 sub chunks, discard last 5
			
def chunks(obs,actions,goals,H,stride):
	'''
	obs is a N x 4 array
	goals is a N x 2 array
	H is length of chunck
	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
	'''
	
	obs_chunks = []
	action_chunks = []
	targets = []
	N = obs.shape[0]
	for i in range(N//stride - H):
		start_ind = i*stride
		end_ind = start_ind + H
		# If end_ind = 4000000, it goes out of bounds
		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
		# if end_ind == N:
		# 	end_ind = N-1
		
		obs_chunk = obs[start_ind:end_ind,:]

		action_chunk = actions[start_ind:end_ind,:]
		
		targets.append(goals[start_ind:end_ind,:])
		obs_chunks.append(obs_chunk)
		action_chunks.append(action_chunk)
			
	
	return np.stack(obs_chunks),np.stack(action_chunks),np.stack(targets)


states = dataset['observations']
actions = dataset['actions']
goals = dataset['infos/goal']

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

states_train  = states[:N_train,:]
actions_train = actions[:N_train,:]
goals_train = goals[:N_train,:]


states_test  = states[N_train:,:]
actions_test = actions[N_train:,:]
goals_test   = goals[N_train:,:]

assert states_train.shape[0] == N_train
assert states_test.shape[0] == N_test

obs_chunks_train, action_chunks_train, targets_train = chunks(states_train, actions_train, goals_train, H, stride)
print('obs_chunks: ', obs_chunks_train)
obs_chunks_test,  action_chunks_test,  targets_test  = chunks(states_test,  actions_test,  goals_test,  H, stride)


experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
# experiment.add_tag('noisy2')


# First, instantiate a skill model
if not state_dependent_prior:
	model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
else:
	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen).cuda()

model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

filename = env_name+'_'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_sg_'+str(state_dec_stop_grad)+'_max_sig_'+str(max_sig)+'_fixed_sig_'+str(fixed_sig)+'_ent_pen_'+str(ent_pen)+'_log'

experiment.log_parameters({'lr':lr,
							   'h_dim':h_dim,
							   'state_dependent_prior':state_dependent_prior,
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
							   'filename':filename})

# add chunks of data to a pytorch dataloader
inputs_train = torch.tensor(np.concatenate([obs_chunks_train, action_chunks_train],axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim
inputs_test  = torch.tensor(np.concatenate([obs_chunks_test,  action_chunks_test], axis=-1),dtype=torch.float32) # array that is dataset_size x T x state_dim+action_dim


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
