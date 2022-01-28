import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py

env = 'antmaze-medium-diverse-v0'
env = gym.make(env)
data = env.get_dataset()

H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 1
episodes = 1
wd = 5e-4

filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
PATH = 'checkpoints/'+filename
skill_model_sdp = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda() #SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
checkpoint = torch.load(PATH)
skill_model_sdp.load_state_dict(checkpoint['model_state_dict'])

