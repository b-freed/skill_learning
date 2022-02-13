'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoints'''

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem
from utils import save_frames_as_gif
from gym.wrappers.monitoring import video_recorder
from utils import make_gif

device = torch.device('cuda:0')

env = 'antmaze-medium-diverse-v0'
# env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

# vid = video_recorder.VideoRecorder(env,path="recording")

H = 20
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
epochs = 300000
skill_seq_len = 10
lr = 1e-4
wd = 0
state_dependent_prior = True
n_iters = 10
import glob
'''
if not state_dependent_prior:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
else:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
'''
# filename = 'maze2d_H'+str(H)+'_log_best.pth'
filename = 'AntMaze_H20_l2reg_0.001_log_best.pth'

PATH = 'checkpoints/'+filename

if not state_dependent_prior:
  	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
  	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
# s0 = env.reset()
# initial_loc = s0[:2]
# env.env.reset_to_location
# print('s0: ', s0)
# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape(1,1,-1)

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

# z_mean,z_sig = skill_model.prior(s0_torch)
# print('z_mean: ', z_mean)
# print('z_sig: ', z_sig)
# skill_seq = z_mean.detach() + z_sig.detach() * torch.randn((1,skill_seq_len,z_dim), device=device)
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device)
skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

# s0_torch = torch.stack(batch_size*[torch.tensor(s0,dtype=torch.float32).cuda().unsqueeze(0)])

# s0 = torch.zeros((batch_size,1,state_dim), device=device)

# s0_torch = torch.stack([env.reset_to_location(initial_loc)])
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=lr)
# determine waypoints
goal_seq = torch.tensor([[random.choice(data['observations'])]], device=device)
#goal_seq = 2*torch.rand((1,skill_seq_len,state_dim), device=device) - 1

# experiment = Experiment(api_key = 'yQQo8E8TOCWYiVSruS7nxHaB5', project_name = 'skill-learning', workspace="anirudh-27")
# experiment.add_tag('Skill PLanning for '+env_name)
# experiment.log_parameters({'lr':lr,
# 							   'h_dim':h_dim,
# 							   'state_dependent_prior':state_dependent_prior,
# 							   'z_dim':z_dim,
# 				 						   'skill_seq_len':skill_seq_len,
# 			  				   'H':H,
# 			  				   'a_dim':a_dim,
# 			  				   'state_dim':state_dim,
# 			  				   'l2_reg':wd})
#experiment.log_metric('Goals', goal_seq)


# def make_gif(frame_folder):
#     frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
#     frame_one = frames[0]
#     frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
#                save_all=True, duration=100, loop=0)



def run_skills_iterative_replanning(env,model,goals):
	
	s0 = env.reset()
	state = s0
	plt.scatter(s0[0],s0[1], label='Initial States')
	plt.scatter(goals[:,:,0].detach().cpu().numpy(),goals[:,:,1].detach().cpu().numpy(), label='Goals')
	plt.figure()
	# for i in range(skill_seq_len):
	# ipdb.set_trace()
	frames = []
	while np.sum((state[:2] - goals.flatten().detach().cpu().numpy()[:2])**2) > 1.0:
	# for i in range(2):
		state_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))])
		# ipdb.set_trace()
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(state_torch, skill_seq, goal_seq)
		skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,100,.5,n_iters)
		skill = skill_seq[0,:].unsqueeze(0)
		mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
		z = mu_z + sigma_z*skill
		
		for j in range(H):
			frames.append(env.render(mode='rgb_array'))
			env.render()
			# vid.capture_frame()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			# skill_seq_states.append(state)
			plt.scatter(state[0],state[1], label='Trajectory',c='b')
		plt.savefig('ant_skills_iterative_replanning')
	# ipdb.set_trace()

	# save_frames_as_gif(frames)
	# for i,f in enumerate(frames):
	# 	plt.figure()
	# 	plt.imshow(f)
	# 	plt.savefig('ant'+str())
	env.close()
	make_gif(frames,name='ant')


		


def run_skill_seq(env,s0,model):
	'''
	'''
	# env.env.set_state(s0[:2],s0[2:])
	state = s0
	# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape((1,1,-1))

	pred_states = []
	pred_sigs = []
	states = []
	plt.figure()
	for i in range(skill_seq_len):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
		z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		skill_seq_states = []
		state_torch = torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_state = s_mean.squeeze().detach().cpu().numpy()
		pred_sig = s_sig.squeeze().detach().cpu().numpy()
		pred_states.append(pred_state)
		pred_sigs.append(pred_sig)
		# run skill for H timesteps
		for j in range(H):
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
			plt.scatter(state[0],state[1], label='Trajectory',c='b')
		states.append(skill_seq_states)
		plt.scatter(state[0],state[1], label='Term state',c='r')
		plt.scatter(pred_state[0],pred_state[1], label='Pred states',c='g')
		plt.errorbar(pred_state[0],pred_state[1],xerr=pred_sig[0],yerr=pred_sig[1],c='g')
		

	states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)



	# plt.figure()
	# plt.scatter(states[:,:,0],states[:,:,1], label='Trajectory')
	plt.scatter(s0[0],s0[1], label='Initial States')
	# plt.scatter(states[:,-1,0],states[:,-1,1], label='Predicted Final States')
	plt.scatter(goals[:,:,0],goals[:,:,1], label='Goals')
	# print('pred_states: ', pred_states)
	# print('pred_states.shape: ', pred_states.shape)
	# plt.scatter(pred_states[:,0],pred_states[:,1], label='Pred states')
	# plt.errorbar(pred_states[:,0],pred_states[:,1],xerr=pred_sigs[:,0],yerr=pred_sigs[:,1])
	# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol= 4)
	plt.axis('square')

	if not state_dependent_prior:
		plt.title('Planned Skills (No State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'_sdp_'+'false'+'.png')
	else:
		plt.title('Planned skills (State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'.png')

	print('SAVED FIG!')



# cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s0_torch, skill_seq, goal_seq)
		
# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,100,.5,n_iters)

# skill_seq = skill_seq.unsqueeze(0)		

# s0 = env.reset()
# run_skill_seq(env,s0,skill_model)

run_skills_iterative_replanning(env,skill_model,goal_seq)








