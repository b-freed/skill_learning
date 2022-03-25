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


def create_dataset_raw(env_name):
    """
    Creates a dictionary of lists of arrays.
    """
    env = gym.make(env_name)
    
    dataset = env.get_dataset()
    dataset_ = d4rl.qlearning_dataset(env)
    split_data = {}

    diff = abs(dataset_['next_observations'][:, :2] - dataset_['observations'][:, :2]).sum(axis=-1)
    rst_idxs = np.where(diff > 1.)[0]

    new_dones = np.zeros_like(dataset['terminals'])
    new_dones[rst_idxs] = True

    split_idxs = np.where(new_dones == True)[0]

    for key, value in dataset.items():
        # OG data has incorrect timeouts; insert correct ones
        if key == 'timeouts':
            split_data[key] = np.split(new_dones, split_idxs + 1) # adding 1 to split_idxs since .split() doesn't include the last index
        else:
            split_data[key] = np.split(value, split_idxs + 1) # adding 1 to split_idxs since .split() doesn't include the last index

    # Sanity checks
    # last timeout is always true; false otherwise
    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        assert t[-1] == True, f'idx: {idx}'
        assert (t[:-1] == False).all(), f'idx: {idx}' 

    # OG and new data should have the same dimensions
    tot_len = lambda data_lis: sum([len(data) for data in data_lis])
    for key, value in dataset.items():
        assert len(value) == tot_len(split_data[key])

    return split_data


def create_dataset_padded(raw_data_creator, *data_args):
    """
    Convert list of arrays to n-d array by zero padding smaller length trajectories
    """
    dataset_raw = raw_data_creator(*data_args)
    data_padded = {}

    n_episodes = len(dataset_raw['rewards'])
    max_horizon = max([len(traj) for traj in dataset_raw['rewards']])

    for key, value in dataset_raw.items():
        data_padded[key] = np.zeros((n_episodes, max_horizon, *value[0].shape[1:]))
        for i in range(n_episodes):
            padding_pattern = ((0, int(max_horizon - len(value[i]))), *[(0, 0) for i in range(value[i].ndim - 1)])
            data_padded[key][i, ...] = np.pad(value[i], padding_pattern)
    
    return data_padded