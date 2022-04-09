import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
# from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
import d4rl
import random
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from matplotlib import animation
from PIL import Image
import cv2
# from pygifsicle import optimize
import imageio

# device = torch.device('cuda:0')

# def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

# 	#Mess with this to change frame size
# 	plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

# 	patch = plt.imshow(frames[0])
# 	plt.axis('off')

# 	def animate(i):
# 		patch.set_data(frames[i])

# 	anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
# 	anim.save(path, writer='imagemagick', fps=60)

def make_gif(frames,name):
	frames = [Image.fromarray(image) for image in frames]
	frame_one = frames[0]
	frame_one.save(name+'.gif', format="GIF", append_images=frames,
			   save_all=True, duration=100, loop=0)

# def make_video(frames,name):
#     height,width,_ = frames[0].shape
#     out = cv2.VideoWriter(name+'.avi',0,15, (height,width))
 
#     for i in range(len(frames)):
#         out.write(frames[i])
#     out.release()

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

def chunks(obs,next_obs,actions,H,stride):
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
		# If end_ind = 4000000, it goes out of bounds
		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
		# if end_ind == N:
		# 	end_ind = N-1
		
		obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

		action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
		
		# print('obs[start_ind+1:end_ind]: ',obs[start_ind+1:end_ind])
		# print('next_obs[start_ind:end_ind-1]: ', next_obs[start_ind:end_ind-1])

		if np.all(obs[start_ind+1:end_ind] == next_obs[start_ind:end_ind-1]): 
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
		else:
			pass
			# print('NOT INCLUDING ',i)

	print('len(obs_chunks): ',len(obs_chunks))
	print('len(action_chunks): ',len(action_chunks))
			
	
	return torch.stack(obs_chunks),torch.stack(action_chunks)


def chunks_online(obs,actions,H,stride):
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
		# If end_ind = 4000000, it goes out of bounds
		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
		# if end_ind == N:
		# 	end_ind = N-1
		
		obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

		action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
		
		# print('obs[start_ind+1:end_ind]: ',obs[start_ind+1:end_ind])
		# print('next_obs[start_ind:end_ind-1]: ', next_obs[start_ind:end_ind-1])
		obs_chunks.append(obs_chunk)
		action_chunks.append(action_chunk)

	print('len(obs_chunks): ',len(obs_chunks))
	print('len(action_chunks): ',len(action_chunks))
			
	
	return torch.stack(obs_chunks),torch.stack(action_chunks)
