import numpy as np
import torch

def clean_data(obs_tensor, next_obs_tensor, actions_tensor):
	#Cleans d4rl datasets because terminals are not always correct

	obs = obs_tensor.cpu().numpy()
	next_obs = next_obs_tensor.cpu().numpy()
	actions = actions_tensor.cpu().numpy()

	#UNCOMMENT DEPENDING ON ENVIRONMENT
	transition_norm = np.linalg.norm(obs[:,:2]-next_obs[:,:2], axis=1) #Antmaze
	#transition_norm = np.linalg.norm(obs[:,:]-next_obs[:,:], axis=1) #Franka or Maze2d
	
	sorted_norm = np.sort(transition_norm)
	
	#UNCOMMENT DEPENDING ON ENVIRONMENT
	#bad_idx = (transition_norm>0.8).nonzero()[0] #Antmaze large 0.8 medium 0.67
	bad_idx = (transition_norm>0.23).nonzero()[0] #Franka 0.23 mixed/complete 0.25 partial
	#bad_idx = (transition_norm>0.22).nonzero()[0] #Maze2d 0.22

	obs = np.delete(obs, bad_idx, axis=0)
	next_obs = np.delete(next_obs, bad_idx, axis=0)
	actions = np.delete(actions, bad_idx, axis=0)

	return torch.tensor(obs).float().cuda(), torch.tensor(next_obs).float().cuda(), torch.tensor(actions).float().cuda()
