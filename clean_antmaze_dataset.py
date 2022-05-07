import numpy as np
import torch

#PATH = 'antmaze-large-diverse-v0/'

#obs = np.load(PATH + 'observations.npy')
#next_obs = np.load(PATH + 'next_observations.npy')
#actions = np.load(PATH + 'actions.npy')

def clean_data(obs_tensor, next_obs_tensor, actions_tensor):
	obs = obs_tensor.cpu().numpy()
	next_obs = next_obs_tensor.cpu().numpy()
	actions = actions_tensor.cpu().numpy()

	transition_norm = np.linalg.norm(obs[:,:3]-next_obs[:,:3], axis=1)

	bad_idx = (transition_norm>0.8).nonzero()[0]
	obs = np.delete(obs, bad_idx, axis=0)
	next_obs = np.delete(next_obs, bad_idx, axis=0)
	actions = np.delete(actions, bad_idx, axis=0)

	return torch.tensor(obs).float().cuda(), torch.tensor(next_obs).float().cuda(), torch.tensor(actions).float().cuda()
