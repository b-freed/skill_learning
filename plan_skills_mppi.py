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
from mppi import mppi_update
from utils import make_gif, make_video
import cv2

device = torch.device('cuda:0')

env = 'antmaze-medium-diverse-v0'
env_name = env
env = gym.make(env)
data = env.get_dataset()

H = 20
replan_freq = H
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
skill_seq_len = 100
lr = 1e-4
wd = 0.001
state_dependent_prior = True
state_dec_stop_grad = False
beta = 1.0
alpha = 1.0
max_sig = None
fixed_sig = None
use_epsilon = True
ent_pen = 0.0
n_iters = 100
a_dist = 'normal'
render = False
max_replans = 50

# background_img = mpimg.imread('maze_medium.png')


filename = 'Antmaze_medium_H20_l2reg_0.001_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'


PATH = 'checkpoints/'+filename

if not state_dependent_prior:
  	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
  	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim,a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
#s0 = env.reset()
# initial_loc = s0[:2]
# env.env.reset_to_location
#print('s0: ', s0)
s0_torch = torch.tensor(env.reset(),dtype=torch.float32).cuda()

# z_mean,z_sig = skill_model.prior(s0_torch)
# print('z_mean: ', z_mean)
# print('z_sig: ', z_sig)
# skill_seq = z_mean.detach() + z_sig.detach() * torch.randn((1,skill_seq_len,z_dim), device=device)
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device)
skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

# s0 = torch.zeros((batch_size,1,state_dim), device=device)

# s0_torch = torch.stack([env.reset_to_location(initial_loc)])
# initialize optimizer for skill sequence
seq_optimizer = torch.optim.Adam([skill_seq], lr=lr)
# determine waypoints
goal_state = np.array(env.target_goal)
print('goal_state: ', goal_state)
#env.set_target(goal_state[:2])
goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)
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

def run_skills_iterative_replanning(env,model,goals,use_epsilon,replan_freq,ep_num):
	
	s0 = env.reset()
	state = s0
	plt.scatter(s0[0],s0[1], label='Initial States')
	plt.scatter(goals[:,:,0].detach().cpu().numpy(),goals[:,:,1].detach().cpu().numpy(), label='Goals')
	plt.figure()
	# for i in range(skill_seq_len):
	# ipdb.set_trace()
	states = [s0]
	frames = []
	n=0
	timeout = False
	# success = True
	l = skill_seq_len
	while np.sum((state[:2] - goals.flatten().detach().cpu().numpy()[:2])**2) > 1.0:
	# for i in range(2):
		state_torch = torch.tensor(state,dtype=torch.float32).cuda()
		cost_fn = lambda s: skill_model.cost_for_mppi(s, goal_seq)
		skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
		skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)


		skill_seq_mean = mppi_update(skill_seq_mean, state_torch, skill_model, cost_fn, batch_size, skill_seq_len, z_dim, True, goal_seq)



		if skill_seq_mean.shape[0] == 0:

			print('OUT OF SKILLS!!!')
			# out_of_skills = True
			# break
		else:
			skill = skill_seq_mean[0,:].unsqueeze(0)

			
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill
		else:
			z = skill
		for j in range(replan_freq):
		# for j in range(100):
			if render:
				frames.append(env.render(mode='rgb_array'))
			# env.render()
			# vid.capture_frame()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,done,_ = env.step(action)
			# print('state: ', state)
			states.append(state)
			
			# skill_seq_states.append(state)
			# plt.scatter(state[0],state[1], label='Trajectory',c='b')
			if np.sum((state[:2] - goals.flatten().detach().cpu().numpy()[:2])**2) <= 1.0:
				break
			if done:
				print('DOOOOOOOOONE!!!!!!!!!!!!!!')
				print('state: ', state)
			# 	print('n: ',n)
				# break
		n += 1

		
	

		fig = plt.figure()
		# plt.imshow(background_img, extent = [-8,28,-8,28])
		plt.scatter(np.stack(states)[:,0],np.stack(states)[:,1])
		plt.scatter(goals.flatten().detach().cpu().numpy()[0],goals.flatten().detach().cpu().numpy()[1])
		plt.axis('equal')
		plt.savefig('ant_iterative_replanning_actual_states_niters'+str(n_iters)+'.png')
	
		if n > max_replans*H/replan_freq:
			print('TIMEOUT!!!!!!!!!!!!!!')
			timeout = True
			break 
	


		# plt.savefig('ant_skills_iterative_replanning')
	# ipdb.set_trace()


	# save_frames_as_gif(frames)
	# for i,f in enumerate(frames):
	# 	plt.figure()
	# 	plt.imshow(f)
	# 	plt.savefig('ant'+str())
	env.close()
	# make_gif(frames,name='ant')
	if render or timeout:
		print('MAKING VIDEO!')
		# if timeout: 
		# 	print('making timout vid')
		# 	make_video(frames,name='failed_ant_'+str(j))
		# else:
		# 	make_video(frames,name='ant')
		make_video(frames,name='ant'+str(ep_num))
		# make_video(frames,name='yant')
		# make_video(frames,name='yant')


	states = np.stack(states)
	return states, np.min(np.sum((states[:,:2] - goals.flatten().detach().cpu().numpy()[:2])**2,axis=-1))





def run_skill_seq(env,s0,model, skill_seq):
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
	plt.scatter(s0[0].detach().cpu().numpy(),s0[1].detach().cpu().numpy(), label='Initial States')
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

'''

cost_fn = lambda s: skill_model.cost_for_mppi(s, goal_seq)
skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
for i in range(n_iters):
	skill_seq = mppi_update(skill_seq_mean, s0_torch, skill_model, cost_fn, batch_size, skill_seq_len, z_dim, True, goal_seq)
	skill_seq_mean = skill_seq
skill_seq = skill_seq.unsqueeze(0)
#skill_seq = skill_seq.tile([batch_size,1,1])
run_skill_seq(env,s0_torch,skill_model, skill_seq)

'''

min_dists_list = []
for i in range(100):
	states,min_dist = run_skills_iterative_replanning(env,skill_model,goal_seq,use_epsilon,replan_freq,i)
	min_dists_list.append(min_dist)
	print('min_dists_list: ',min_dists_list)
	np.save('min_dists_list_n_iters_'+str(n_iters),min_dists_list)
	print('----------------'+str(i)+'-----------------')
