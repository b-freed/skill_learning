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

		loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses()

		model_optimizer.zero_grad()
		loss_tot.backward()
		model_optimizer.step()

		# log losses

		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses) 


# First, instantiate a skill model
model = SkillModel()
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # default lr-0.0001

batch_size = 100

def get_data():

    env = PointmassEnv() 
    obs = []
    actions = []
    goals = []
    for i in range(10000):
        start_loc = 2*np.random.uniform(size=2) - 1
        start_state = np.concatenate([start_loc,np.zeros(2)]) 
        goal_loc = 2*np.random.uniform(size=2) - 1
        state = env.reset(start_state)
        states = [state]
        action = []
    
        for t in range(100):
            # print('env.x: ', env.x)
            u = env.get_stabilizing_control(goal_loc)
            # print('u: ', u)
            state = env.step(u)
            states.append(state)
            action.append(u)
        
        obs.append(states)
        actions.append(action)
        goals.append(goal_loc)
    
    
    obs = np.stack(obs)
    actions = np.stack(actions)
    goals = np.stack(goals)

    return obs, actions, goals
    

states, actions, goals = get_data()
state_dim = states.shape[1]
a_dim = actions.shape[1]
# splitting up the dataset into subsequences in which we're going to a particular goal.  Every time the goal changes we make a new subsequence.
# chunks might not all be same length, might have to split long chunks down into sub-chunks, discarding leftover chunks that are shorter than our chunck length.
# so if I have a chunk that's 125 long, I can split into 6 x 20 sub chunks, discard last 5
N = states.shape[0]
			
def ben_chunk(obs,actions,goals,H,stride):
	'''
	obs is a N x 4 array
	goals is a N x 2 array
	H is length of chunck
	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
	'''
	
	obs_chunks = []
	action_chunks = []
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
			
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
			
	return np.stack(obs_chunks),np.stack(action_chunks)


H = 20
stride = 20
obs_chunks, action_chunks = ben_chunk(states, actions, goals, H, stride)

# add chunks of data to a pytorch dataloader
inputs = np.concatenate([obs_chunks, action_chunks],axis=-1) # array that is dataset_size x T x state_dim+action_dim 
# targets = data['infos/goal'] can be anyhing, maybe make this the goals but we probably won't use it
train_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32)) # ,torch.tensor(targets,dtype=torch.float32))

train_loader = DataLoader(
	train_data,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

n_epochs = 1000 # initial value
for i in range(n_epochs):
	train(model,model_optimizer)

PATH = 
# torch.save(model.state_dict(), PATH)
torch.save(model, PATH)
torch.save(model_optimizer, PATH)