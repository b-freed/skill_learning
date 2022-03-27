from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import LowLevelDynamicsFF
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py

device = torch.device('cuda:0')

def train(model,model_optimizer):
	
	losses = []
	
	for batch_id, (state,action,next_state) in enumerate(train_loader):
		
		state = state.cuda()
		action = action.cuda()
		next_state = next_state.cuda()
		loss = model.get_loss(state, action, next_state)

		model_optimizer.zero_grad()
		loss.backward()
		model_optimizer.step()

		# log losses

		losses.append(loss.item())
		
	return np.mean(losses)
	
def test(model):
	
	losses = []
	
	with torch.no_grad():
		for batch_id, (state,action,next_state) in enumerate(test_loader):
			state = state.cuda()
			action = action.cuda()
			next_state = next_state.cuda()
			loss = model.get_loss(state, action, next_state)

			# log losses
			losses.append(loss.item())

	return np.mean(losses)


# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
# env_name = 'antmaze-medium-diverse-v0'  
# env_name = 'maze2d-large-v1'
env_name = 'antmaze-large-diverse-v0'
env = gym.make(env_name)

dataset_file = None
#dataset_file = "datasets/maze2d-umaze-v1.hdf5"
# dataset_file = 'datasets/maze2d-large-v1-noisy-2.hdf5'

if dataset_file is None:
	dataset = d4rl.qlearning_dataset(env) #env.get_dataset()
else:
	dataset = d4rl.qlearning_dataset(env,h5py.File(dataset_file, "r"))  # Not sure if this will work


batch_size = 100

h_dim = 256
lr = 1e-4
wd = 0.001
n_epochs = 50000
test_split = .2
			
# def chunks(obs,actions,goals,H,stride):
# 	'''
# 	obs is a N x 4 array
# 	goals is a N x 2 array
# 	H is length of chunck
# 	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
# 	'''
	
# 	obs_chunks = []
# 	action_chunks = []
# 	targets = []
# 	N = obs.shape[0]
# 	for i in range(N//stride - H):
# 		start_ind = i*stride
# 		end_ind = start_ind + H
# 		# If end_ind = 4000000, it goes out of bounds
# 		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
# 		# if end_ind == N:
# 		# 	end_ind = N-1
		
# 		obs_chunk = obs[start_ind:end_ind,:]

# 		action_chunk = actions[start_ind:end_ind,:]
		
# 		targets.append(goals[start_ind:end_ind,:])
# 		obs_chunks.append(obs_chunk)
# 		action_chunks.append(action_chunk)
			
	
# 	return np.stack(obs_chunks),np.stack(action_chunks),np.stack(targets)


# states = torch.tensor(dataset['observations'][:-1,:],dtype=torch.float32,device=device)
# next_states = torch.tensor(dataset['observations'][1:,:],dtype=torch.float32,device=device)
# actions = torch.tensor(dataset['actions'][:-1,:],dtype=torch.float32,device=device)

states = torch.tensor(dataset['observations'],dtype=torch.float32,device=device)
next_states = torch.tensor(dataset['next_observations'],dtype=torch.float32,device=device)
actions = torch.tensor(dataset['actions'],dtype=torch.float32,device=device)

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train



experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
# experiment.add_tag('noisy2')



filename = 'll_dynamics_'+str(env_name)+'_l2reg_'+str(wd)+'_lr_'+str(lr)+'_log'


# TODO delete irrelevant items
experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'env_name':env_name,
							'filename':filename})

# define model
model = LowLevelDynamicsFF(state_dim,a_dim,h_dim).cuda()
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


dataset = torch.utils.data.TensorDataset(states,actions,next_states)
# train_data, test_data = torch.utils.data.random_split(dataset, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])
train_data, test_data = torch.utils.data.random_split(dataset, [N_train, N_test])


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=True)


min_test_loss = 10**10
for i in range(n_epochs):
	loss = train(model,model_optimizer)
	
	print("--------TRAIN---------")
	
	print('loss: ', loss)
	print(i)
	experiment.log_metric("loss", loss, step=i)
	

	test_loss = test(model)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	
	experiment.log_metric("test_loss", test_loss, step=i)
	
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
