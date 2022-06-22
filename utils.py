import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
# from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
#import d4rl
import random
#import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from matplotlib import animation
from PIL import Image
import cv2
import imageio


def make_gif(frames,name):
	frames = [Image.fromarray(image) for image in frames]
	frame_one = frames[0]
	frame_one.save(name+'.gif', format="GIF", append_images=frames,
			   save_all=True, duration=100, loop=0)

def make_video(frames,name):
	writer = imageio.get_writer(name+'.mp4', fps=20)

	for im in frames:
		writer.append_data(im)
	writer.close()

def reparameterize(mean, std):
	eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
	return mean + std*eps

def stable_weighted_log_sum_exp(x,w,sum_dim):
	a = torch.min(x)
	ipdb.set_trace()

	weighted_sum = torch.sum(w * torch.exp(x - a),sum_dim)

	return a + torch.log(weighted_sum)

def chunks(obs,actions,H,stride):
	'''
	obs is a N x 4 array
	goals is a N x 2 array
	H is length of chunck
	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
	'''
	
	obs_chunks = []
	action_chunks = []
	N = obs.shape[0]
	for i in range(N//stride - H):
		start_ind = i*stride
		end_ind = start_ind + H

		obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

		action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)

		#UNCOMMENT BASED ON ENVIRONMENT
		#loc_deltas = obs_chunk[1:,:2] - obs_chunk[:-1,:2] #Antmaze
		loc_deltas = obs_chunk[1:,:] - obs_chunk[:-1,:] #Franka or Maze2d
		
		norms = np.linalg.norm(loc_deltas,axis=-1)
		#USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
		if np.all(norms <= 0.8): #Antmaze large 0.8 medium 0.67 / Franka 0.23 mixed/complete 0.25 partial / Maze2d 0.22
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
		else:
			pass

		'''
		if(np.all(terminals[start_ind:end_ind]==False)):
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
		else:
			pass
		'''

	print('len(obs_chunks): ',len(obs_chunks))
	print('len(action_chunks): ',len(action_chunks))
			
	
	return torch.stack(obs_chunks),torch.stack(action_chunks)


def z_score(x,mu,sig):

	return torch.mean(((x-mu)/sig)**2)