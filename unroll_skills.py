
from fileinput import filename
from tokenize import ContStr
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
import d4rl
import random
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem, cem_variable_length
# from utils import make_video, save_frames_as_gif
# from gym.wrappers.monitoring import video_recorder
from utils import make_gif,make_video, reparameterize

device = torch.device('cuda:0')

# env = 'antmaze-medium-diverse-v0'
env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

# vid = video_recorder.VideoRecorder(env,path="recording")

H = 20
replan_freq = H
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
skill_seq_len = 1
lr = 1e-4
wd = .001
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
max_sig = None
fixed_sig = None
n_iters = 10
a_dist = 'normal'
keep_frac = 0.01
background_img = mpimg.imread('maze_medium.png')
use_epsilon = True

# filename = 'AntMaze_H20_l2reg_0.01_a_1.0_b_1.0_sg_True_max_sig_None_fixed_sig_0.1_log_best.pth'#'AntMaze_H20_l2reg_0.001_a_1.0_b_0.01_sg_False_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.001_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_1.0_log.pth'
# filename = 'AntMaze_H20_l2reg_0.001_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_1.0_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.001_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.001_a_1.0_b_0.1_sg_True_log.pth'
filename = 'maze2d-large-v1_tsdp_H40_l2reg_0.0_a_1.0_b_1.0_sg_True_max_sig_None_fixed_sig_None_log_best.pth'

PATH = 'checkpoints/'+filename

if not state_dependent_prior:
  	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])


def sample_rollouts(model,state,T):

	epsilons = torch.randn((T,z_dim),device = device)
	state = torch.stack(batch_size * [state])
	states = [state]
	for i in range(T):

		# ipdb.set_trace()
		mu_z, sigma_z = model.prior(state)
		ep = epsilons[i,:]
		ep = torch.stack(batch_size * [ep])
		z = mu_z + sigma_z*ep
		z = torch.stack(batch_size*[z[0,:]])
		s_mu,s_sig = model.decoder.abstract_dynamics(state,z)
		print('s_mu: ', s_mu)
		print('s_sig: ', s_sig)

		state = reparameterize(s_mu,s_sig)

		states.append(state)
	
	return torch.stack(states,dim=1)



state = torch.tensor(env.reset(),dtype=torch.float32,device=device)
states = sample_rollouts(skill_model,state,10)
states = states.detach().cpu().numpy()



plt.plot(states[:,:,0].T,states[:,:,1].T)
plt.savefig('unrolled_skills')



