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

	for batch_id, (data, targets) in enumerate(train_loader):
		data, targets = data.cuda(), targets.cuda()
		states = data[:,:,:model.state_dim]  # first state_dim elements are the state
		actions = data[:,:,model.state_dim:]	 # rest are actions

		loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses(states, actions)

		model_optimizer.zero_grad()
		loss_tot.backward()
		model_optimizer.step()

		# log losses

		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)

def test(model):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []

	with torch.no_grad():
		for batch_id, (data, targets) in enumerate(test_loader):
			data, targets = data.cuda(), targets.cuda()
			states = data[:,:,:model.state_dim]  # first state_dim elements are the state
			actions = data[:,:,model.state_dim:]	 # rest are actions

			loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses(states, actions)

			# log losses

			losses.append(loss_tot.item())
			s_T_losses.append(s_T_loss.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)


# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
env = 'antmaze-medium-diverse-v0'  # maze whatever
env = gym.make(env)
dataset = env.get_dataset()  # dictionary, with 'observations', 'rewards', 'actions', 'infos/goal'
#dataset_file = "datasets/maze2d-umaze-v1.hdf5"
#dataset = h5py.File(dataset_file, "r")

batch_size = 100

states = dataset['observations']
N = states.shape[0]

state_dim = states.shape[1]
actions = dataset['actions']
a_dim = actions.shape[1]

h_dim = 256
z_dim = 256
lr = 5e-5
state_dependent_prior = True


goals = dataset['infos/goal']
H = 40
stride = 40
n_epochs = 50000
a_dist = 'normal' # 'tanh_normal' or 'normal'
print(a_dim)
print(state_dim)

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
	for i in range((N-1)//stride):
		start_ind = i*stride
		end_ind = start_ind + H
		# If end_ind = 4000000, it goes out of bounds
		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
		if end_ind == N:
			end_ind = N-1
		
		start_goal = goals[start_ind,:]
		end_goal = goals[end_ind,:]
		
		if start_goal[0] == end_goal[0] and start_goal[1] == end_goal[1]:
		
			obs_chunk = obs[start_ind:end_ind,:]
			action_chunk = actions[start_ind:end_ind,:]
			
			targets.append(goals[start_ind:end_ind,:])
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
			
	return np.stack(obs_chunks),np.stack(action_chunks),np.stack(targets)

obs_chunks, action_chunks, targets = chunks(states, actions, goals, H, stride)

experiment = Experiment(api_key = 'yQQo8E8TOCWYiVSruS7nxHaB5', project_name = 'skill-learning', workspace = 'anirudh-27')
experiment.add_tag('AntMaze H_'+str(H)+' model')

# First, instantiate a skill model
if not state_dependent_prior:
	model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()

model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

experiment.log_parameters({'lr':lr,
							   'h_dim':h_dim,
							   'state_dependent_prior':state_dependent_prior,
							   'z_dim':z_dim,
			  				   'H':H,
			   				   'a_dist':a_dist})

# add chunks of data to a pytorch dataloader
inputs = np.concatenate([obs_chunks, action_chunks],axis=-1) # array that is dataset_size x T x state_dim+action_dim

dataset_size = len(inputs)
train_data, test_data = torch.utils.data.random_split(inputs, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])
train_targets, test_targets = torch.utils.data.random_split(targets, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])

train_data = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32))

train_loader = DataLoader(
	train_data,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	test_data,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10
for i in range(n_epochs):
	loss, s_T_loss, a_loss, kl_loss = train(model,model_optimizer)
	
	print("--------TRAIN---------")
	
	print('loss: ', loss)
	print('s_T_loss: ', s_T_loss)
	print('a_loss: ', a_loss)
	print('kl_loss: ', kl_loss)
	print(i)
	experiment.log_metric("loss", loss, step=i)
	experiment.log_metric("s_T_loss", s_T_loss, step=i)
	experiment.log_metric("a_loss", a_loss, step=i)
	experiment.log_metric("kl_loss", kl_loss, step=i)
	
	test_loss, test_s_T_loss, test_a_loss, test_kl_loss = test(model)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	print('test_s_T_loss: ', test_s_T_loss)
	print('test_a_loss: ', test_a_loss)
	print('test_kl_loss: ', test_kl_loss)
	print(i)
	experiment.log_metric("test_loss", test_loss, step=i)
	experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
	experiment.log_metric("test_a_loss", test_a_loss, step=i)
	experiment.log_metric("test_kl_loss", test_kl_loss, step=i)

	if i % 10 == 0:
		filename = 'AntMaze_H'+str(H)+'_log.pth'
		checkpoint_path = 'checkpoints/'+ filename
		torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
	if test_loss < min_test_loss:
		min_test_loss = test_loss
		filename = 'AntMaze_H'+str(H)+'_log_best.pth'
		checkpoint_path = 'checkpoints/'+ filename
		torch.save({'model_state_dict': model.state_dict(),
			    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)
