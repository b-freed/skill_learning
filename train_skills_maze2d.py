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
import d4rl
import ipdb

def train(model,model_optimizer):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []

	for batch_id, (data, target) in enumerate(train_loader):
		data, target = data.cuda(), target.cuda()
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


# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
env = 'maze2d-large-v1'  # maze whatever
env = gym.make(env)
data = env.get_dataset()  # dictionary, with 'observations', 'rewards', 'actions', 'infos/goal'

batch_size = 100

states = data['observations']
state_dim = states.shape[1]
actions = data['actions']
a_dim = actions.shape[1]
h_dim = 128
z_dim = 20
lr = 1e-4
state_dependent_prior = True

N = states.shape[0]
goals = data['infos/goal']
H = 20
stride = 20
n_epochs = 10000

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
experiment.add_tag('New model on d4rl envs')

# First, instantiate a skill model
if not state_dependent_prior:
	model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()

model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

experiment.log_parameters({'lr':lr,
							   'h_dim':h_dim,
							   'state_dependent_prior':state_dependent_prior,
							   'z_dim':z_dim})

# add chunks of data to a pytorch dataloader
inputs = np.concatenate([obs_chunks, action_chunks],axis=-1) # array that is dataset_size x T x state_dim+action_dim 
train_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32) ,torch.tensor(targets,dtype=torch.float32))

train_loader = DataLoader(
	train_data,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

for i in range(n_epochs):
	loss, s_T_loss, a_loss, kl_loss = train(model,model_optimizer)
	print('loss: ', loss)
	print('s_T_loss: ', s_T_loss)
	print('a_loss: ', a_loss)
	print('kl_loss: ', kl_loss)
	print(i)
	experiment.log_metric("loss", loss, step=i)
	experiment.log_metric("s_T_loss", s_T_loss, step=i)
	experiment.log_metric("a_loss", a_loss, step=i)
	experiment.log_metric("kl_loss", kl_loss, step=i)

	if i % 10 == 0:
		if not state_dependent_prior:
			filename = 'maze2d_log.pth'
		else:
			filename = 'maze2d_state_dep_prior_log.pth'
		checkpoint_path = 'checkpoints/'+ filename
		torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
