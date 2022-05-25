import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, LowLevelDynamicsFF, Prior
import ipdb
import d4rl
import random
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem, cem_variable_length
from utils import make_gif,make_video

def convert_epsilon_to_z(epsilon,s0,model):

	s = s0
	z_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_z, sigma_z = model.prior(s)
		z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		z_seq.append(z_i)
		s_mean,_ = model.decoder.abstract_dynamics(s,z_i)
		s = s_mean

	return torch.cat(z_seq,dim=1)

def convert_epsilon_to_a(epsilon,s0,model):

	s = s0
	a_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_a, sigma_a = model.prior(s)
		a_i = mu_a + sigma_a*action_seq[:,i:i+1,:]
		a_seq.append(a_i)
		s_mean,_ = model(s,a_i)
		s = s_mean

	return torch.cat(a_seq,dim=1)[0]

pred_term_states_x = []
pred_term_states_y = []

def run_skill_seq(ax,skill_seq,env,s0,model,use_epsilon,use_dynamics_model=False,dynamics_model=None,init_state_arr=[],term_state_arr=[],abs_term_state_arr=[],ll_term_state_arr=[]):
	'''
	'''
	ax.set_xlim([-3,3])
	env.env.set_state(s0[:15],s0[15:])
	state = s0
	# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape((1,1,-1))

	pred_states = []
	pred_sigs = []
	states = []
	# plt.figure()
	for i in range(skill_seq.shape[1]):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		else:
			z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		state_torch = torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_state = s_mean.squeeze().detach().cpu().numpy()
		pred_sig = s_sig.squeeze().detach().cpu().numpy()
		pred_states.append(pred_state)
		pred_sigs.append(pred_sig)
		
		state_pred = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
		# run skill for H timesteps
		for j in range(H):
			env.render()
			if(not use_dynamics_model):
				action = model.decoder.ll_policy.numpy_policy(state,z)
			else:
				action = model.decoder.ll_policy.numpy_policy(state_pred[0,0,:].detach().cpu().numpy(),z)
				action_torch = torch.reshape(torch.tensor(action,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
				state_pred_mean,state_pred_sig = dynamics_model(state_pred,action_torch)
				eps = torch.normal(torch.zeros(state_pred_mean.size()).cuda(), torch.ones(state_pred_mean.size()).cuda())
				state_pred = state_pred_mean + eps*state_pred_sig
				#ax.scatter(state_pred[0,0,0].detach().cpu().numpy(),state_pred[0,0,1].detach().cpu().numpy(), label='Trajectory',c='b')
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
			if(not use_dynamics_model):
				if(j==0):
					ax.scatter(state[0],state[1], label='Executed Trajectory',c='orange')
				else:
					ax.scatter(state[0],state[1], c='orange')
		states.append(skill_seq_states)
		if(use_dynamics_model):
			pred_term_states_x.append(state_pred[0,0,0].detach().cpu().numpy())
			pred_term_states_y.append(state_pred[0,0,1].detach().cpu().numpy())
		#	ax.scatter(state[0],state[1], label='Trajectory',c='purple')

		if(not use_dynamics_model):
			ax.scatter(state[0],state[1], label='True Terminal State',c='r')
			ax.scatter(pred_state[0],pred_state[1], label='Pred term state distr(Abstract dynamics)',c='g')
			ax.errorbar(pred_state[0],pred_state[1],xerr=pred_sig[0],yerr=pred_sig[1],c='g')
		#ax.savefig('plots/plot.png')
		#plt.close()

	states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)

	return state,states

device = torch.device('cuda:0')

env = 'antmaze-large-diverse-v0'
# env = 'antmaze-medium-diverse-v0'
# env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

skill_seq_len = 10
H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig =  0.0
n_iters = 100
a_dist = 'normal'
keep_frac = 0.5
a_dist = 'normal'
keep_frac = 0.5

use_epsilon = True
max_ep = None
cem_l2_pen = 0.0
var_pen = 0.0
render = False
variable_length = False
max_replans = 1
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True

filename = 'EM_model_05_08_22_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_a_dist_normal_log_best_sT.pth'
PATH = 'checkpoints/'+filename

if term_state_dependent_prior:
	skill_model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,init_state_dependent=init_state_dependent).cuda()
else:
	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

h_dim = 512
goal_conditioned = False

PATH_DYNAMICS = 'checkpoints/ll_dynamics_antmaze-large-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'
PATH_PRIOR = 'checkpoints/ll_prior_antmaze-large-diverse-v0_l2reg_0.001_lr_0.0001_log_hdim_512_Decay_best.pth'

dynamics_model = LowLevelDynamicsFF(state_dim,a_dim,h_dim,deterministic=False).cuda()
checkpoint = torch.load(PATH_DYNAMICS)
dynamics_model.load_state_dict(checkpoint['model_state_dict'])
if(use_epsilon):
	low_level_prior = Prior(state_dim,a_dim,h_dim,goal_conditioned).cuda()
	checkpoint = torch.load(PATH_PRIOR)
	low_level_prior.load_state_dict(checkpoint['model_state_dict'])
	dynamics_model.prior = low_level_prior

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])
skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
skill_seq.requires_grad = True
#goal_state = np.array(env.target_goal)
goal_state = np.array([5.0,8])
goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)

execute_n_skills = 1
fig, ax = plt.subplots()

print('RUNNING SKILLS WITH TRUE STATES')
init_state = env.reset()
for j in range(1):
	#print(init_state[:2])
	for i in range(max_replans):
		s_torch = torch.cat(batch_size*[torch.tensor(init_state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
		if(j==0):
			skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
			skill_seq = skill_seq[:execute_n_skills,:]
			skill_seq = skill_seq.unsqueeze(0)	
			skill_seq = convert_epsilon_to_z(skill_seq,s_torch[:1,:,:],skill_model)
		state,states = run_skill_seq(ax,skill_seq,env,init_state,skill_model,use_epsilon=False)
	#print(state[:2])

fig.savefig('plots/plot.png')

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])
#skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
#skill_seq.requires_grad = True
print('RUNNING SKILLS WITH LOW LEVEL DYNAMICS MODEL')
for j in range(10):
	#init_state = env.reset()
	#print(init_state[:2])
	for i in range(max_replans):
		s_torch = torch.cat(batch_size*[torch.tensor(init_state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		state,states = run_skill_seq(ax,skill_seq,env,init_state,skill_model,use_epsilon=False,use_dynamics_model=True,dynamics_model=dynamics_model)
	#print(state[:2])
#ax.scatter(pred_term_states_x,pred_term_states_y, label='Pred term states(LL dynamics)',c='blue')
mean_x = np.mean(pred_term_states_x)
mean_y = np.mean(pred_term_states_y)
std_x = np.std(pred_term_states_x)
std_y = np.std(pred_term_states_y)
ax.scatter(mean_x,mean_y, label='Pred term state distr(LL dynamics)',c='blue')
ax.errorbar(mean_x,mean_y,xerr=std_x,yerr=std_y,c='blue')
ax.legend()
fig.savefig('plots/plot.png')