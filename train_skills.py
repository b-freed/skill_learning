import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel

import d4rl



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

# instantiating the environmnet, getting the dataset.
# the data is in a big dictionary, containing long sequences of obs, rew, actions, goals
env = 'maze2d-large-v1'  # maze whatever
env = gym.make(env)
data = env.get_dataset()  # dictionary, with 'observations', 'rewards', 'actions', 'infos/goal'

states = data['observations']
state_dim = states.shape[1]
actions = data['actions']
a_dim = actions.shape[1]
# splitting up the dataset into subsequences in which we're going to a particular goal.  Every time the goal changes we make a new subsequence.
# chunks might not all be same length, might have to split long chunks down into sub-chunks, discarding leftover chunks that are shorter than our chunck length.
# so if I have a chunk that's 125 long, I can split into 6 x 20 sub chunks, discard last 5
N = states.shape[0]
paths = collections.defaultdict(list)
initial_goal = data['infos/goal'][0]
for i in range(N):
	if initial_goal == data['infos/goal'][i]:
		# We append first set of obs corresponding to first goal into the list as first subsequence
	# Once the goal changes set initial_goal to next goal
	else:
		initial_goal = data['infos/goal'][i]
		# Continue appending list for this set of obs corresponding to the current goal but indicate that this is new subsequence.

# add chunks of data to a pytorch dataloader
inputs = np.concatenate([paths,actions],axis=-1) # array that is dataset_size x T x state_dim+action_dim 
# targets = data['infos/goal'] can be anyhing, maybe make this the goals but we probably won't use it
train_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32)) # ,torch.tensor(targets,dtype=torch.float32))

train_loader = DataLoader(
	train_data,
	batch_size=self.batch_size,
	num_workers=0)  # not really sure about num_workers...


for i in range(n_epochs):
	train()
