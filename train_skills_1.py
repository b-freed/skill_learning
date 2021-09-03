import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
from pointmass_env import PointmassEnv



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


batch_size = 100

def get_data():

	env = PointmassEnv()
	obs = []
	goals = []
	actions = []
	for i in range(10000):
		start_loc = 2*np.random.uniform(size=2) - 1
		start_state = np.concatenate([start_loc,np.zeros(2)])
		goal_loc = 2*np.random.uniform(size=2) - 1
		state = env.reset(start_state)
		states = [state]
		action = []
		goal = []

		for t in range(100):
			#print('env.x: ', env.x)
			u = env.get_stabilizing_control(goal_loc)
			#print('u: ', u)
			state = env.step(u)
			if t != 99:
				states.append(state)
			action.append(u)
			goal.append(goal_loc)

		obs.append(states)
		actions.append(action)
		goals.append(goal)
	
	obs = np.stack(obs)
	actions = np.stack(actions)
	goals = np.stack(goals)

	return obs, actions, goals

states, actions, goals = get_data()
state_dim = states.shape[2]
a_dim = actions.shape[2]
h_dim = 128
N = states.shape[0]


# First, instantiate a skill model
model = SkillModel(state_dim, a_dim, 20, h_dim).cuda()
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # default lr-0.0001

# add chunks of data to a pytorch dataloader
inputs = np.concatenate([states, actions],axis=-1) # array that is dataset_size x T x state_dim+action_dim 
targets = goals
train_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets,dtype=torch.float32))

train_loader = DataLoader(
	train_data,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

n_epochs = 1000 # initial value
for i in range(n_epochs):
	train(model,model_optimizer)

'''
PATH = 
# torch.save(model.state_dict(), PATH)
torch.save(model, PATH)
torch.save(model_optimizer, PATH)
'''
