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
# import cv2
# from pygifsicle import optimize
import imageio

def kl_divergence_bernoulli(post_logits, prior_probs, eps=1e-16): 
    """
    PyTorch's torch.distributions.kl.kl_divergence seems to have issues in gradient computation 
    with Bernoulli distribution - results into NaN gradients. Hence, the implementation.
    """
    post_probs = nn.Sigmoid()(post_logits)
    log_post_probs = (post_probs + eps).log()
    log_prior_probs = (prior_probs + eps).log()
    kl_div = torch.sum(post_probs * (log_post_probs - log_prior_probs), dim=-1)

    return kl_div

# def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

# 	#Mess with this to change frame size
# 	plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

# 	patch = plt.imshow(frames[0])
# 	plt.axis('off')

# 	def animate(i):
# 		patch.set_data(frames[i])

# 	anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
# 	anim.save(path, writer='imagemagick', fps=60)

def boundary_sampler(log_alpha, temperature=1.0):
    # sample and return corresponding logit
    log_sample_alpha = gumbel_softmax(log_alpha, temperature)

    # probability
    log_sample_alpha = log_sample_alpha - torch.logsumexp(log_sample_alpha, dim=-1, keepdim=True)
    sample_prob = log_sample_alpha.exp()
    sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[torch.max(sample_prob, dim=-1)[1]]

    # sample with rounding and st-estimator
    sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

    # return sample data and logit
    return sample_data, log_sample_alpha

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape).to(device)
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

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

def create_dataset_raw2(env_name):
    """
    Creates a dictionary of lists of arrays. Version 2 clips all lists to the same length. Removes shorter sequences.
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

    # Clip all lists to the same length
    for key, value in split_data.items():
        l = []
        for v in split_data[key]:
            if len(v) < 1000:
                continue
            l.append(v[:1000])
        split_data[key] = l

    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        t[-1] = True

    # Sanity checks
    # last timeout is always true; false otherwise
    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        assert t[-1] == True, f'idx: {idx}'
        assert (t[:-1] == False).all(), f'idx: {idx}' 

    return split_data


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


def create_dataset_auto_horizon(obs, actions, H):
    data = []
    for o_, h_ in zip(obs, actions):
        preserved_data = len(o_) - len(o_)%H
        o = np.stack(np.split(o_[:preserved_data], preserved_data//H))
        h = np.stack(np.split(h_[:preserved_data], preserved_data//H))
        data.append(np.concatenate([o, h], axis=-1))
    return np.concatenate(data)
