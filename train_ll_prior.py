from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import Prior
import gym
from mujoco_py import GlfwContext
import clean_antmaze_dataset
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py

device = torch.device('cuda:0')

def train(model,model_optimizer):
	
	losses = []
	
	for batch_id, (state,goal,action) in enumerate(train_loader):
		
		state = state.cuda()
		action = action.cuda()
		loss = model.get_loss(state, action, goal)

		model_optimizer.zero_grad()
		loss.backward()
		model_optimizer.step()

		# log losses

		losses.append(loss.item())
		
	return np.mean(losses)
	
def test(model):
	
	losses = []
	
	with torch.no_grad():
		for batch_id, (state,goal,action) in enumerate(test_loader):
			state = state.cuda()
			action = action.cuda()
			loss = model.get_loss(state, action, goal)

			# log losses
			losses.append(loss.item())

	return np.mean(losses)


# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
# env_name = 'antmaze-medium-diverse-v0'  
# env_name = 'maze2d-large-v1'
env_name = 'antmaze-medium-diverse-v0'
env = gym.make(env_name)

dataset_file = None
#dataset_file = "datasets/maze2d-umaze-v1.hdf5"
# dataset_file = 'datasets/maze2d-large-v1-noisy-2.hdf5'

if dataset_file is None:
	dataset = env.get_dataset()#d4rl.qlearning_dataset(env) #env.get_dataset()
else:
	dataset = d4rl.qlearning_dataset(env,h5py.File(dataset_file, "r"))  # Not sure if this will work


batch_size = 100

h_dim = 512
lr = 1e-4
wd = 0.001
n_epochs = 50000
test_split = .2
decay = True
lr_decay = 0.1
lr_decay_epochs_interval = 100
goal_conditioned = True
		
states = torch.tensor(dataset['observations'],dtype=torch.float32,device=device)
#next_states = torch.tensor(dataset['next_observations'],dtype=torch.float32,device=device)
actions = torch.tensor(dataset['actions'],dtype=torch.float32,device=device)
goals = torch.tensor(dataset['infos/goal'],dtype=torch.float32,device=device)
#states, next_states, actions = clean_antmaze_dataset.clean_data(states, next_states, actions)

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]
goal_dim = goals.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train



experiment = Experiment(api_key = 'wb7Q8i6WX2nuxXfHJdBSs9PUV', project_name = 'skill-learning')
# experiment.add_tag('noisy2')



filename = 'll_prior_'+str(env_name)+'_l2reg_'+str(wd)+'_lr_'+str(lr)+'_log'


# TODO delete irrelevant items
experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'env_name':env_name,
							'filename':filename})

# define model
model = Prior(state_dim,a_dim,h_dim,goal_conditioned).cuda()
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
if decay == True:
	lr_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, 1, lr_decay)

if(goal_conditioned):
	dataset = torch.utils.data.TensorDataset(states,goals,actions)
else:
	dataset = torch.utils.data.TensorDataset(states,actions)
# train_data, test_data = torch.utils.data.random_split(dataset, [int(0.8*dataset_size), int(dataset_size-int(0.8*dataset_size))])
train_data, test_data = torch.utils.data.random_split(dataset, [N_train, N_test])


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=True)


min_test_loss = 10**10
for i in range(n_epochs):
	print('EPOCH: ',i)
	if (i+1)%lr_decay_epochs_interval == 0 and decay == True:
		lr_scheduler.step()

	train_loss = train(model,model_optimizer)
	
	print("--------TRAIN---------")
	
	print('train_loss: ', train_loss)

	experiment.log_metric("train_loss", train_loss, step=i)
	

	test_loss = test(model)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	
	experiment.log_metric("test_loss", test_loss, step=i)
	
	if i % 10 == 0:
		
			
		checkpoint_path = 'checkpoints/'+ filename + '_hdim_512_GoalConditioned_Decay.pth'
		torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
	if test_loss < min_test_loss:
		min_test_loss = test_loss

		
			
		checkpoint_path = 'checkpoints/'+ filename + '_hdim_512_GoalConditioned_Decay_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
				'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)
