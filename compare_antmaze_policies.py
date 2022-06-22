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
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.mlab as mlab

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
term_states_x = []
term_states_y = []
term_states = np.zeros((30,2))
pred_term_states = np.zeros((30,2))

x_grid = np.arange(-10,10,0.01)
y_grid = np.arange(0,20,0.01)
X,Y = np.meshgrid(x_grid,y_grid)
abs_stat = []
#print(x_grid.shape)

def run_skill_seq(ax,skill_seq,env,s0,model,use_epsilon,use_dynamics_model=True,dynamics_model=None,init_state_arr=[],term_state_arr=[],abs_term_state_arr=[],ll_term_state_arr=[],plot_abstract=False,ITER=0):
	state = s0
	pred_states = []
	pred_sigs = []
	states = np.zeros((0,state.shape[0]))
	ll_states = np.zeros((0,state.shape[0]))
	abs_states = np.zeros((0,state.shape[0]))
	abs_states = np.vstack([abs_states,state])
	abs_sigs = np.zeros((0,state.shape[0]))
	states = np.vstack([states,state])
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
		state_torch = torch.tensor(abs_states[i],dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_state = s_mean.squeeze().detach().cpu().numpy()
		abs_states = np.vstack([abs_states,pred_state])
		pred_sig = s_sig.squeeze().detach().cpu().numpy()
		abs_sigs = np.vstack([abs_sigs,pred_sig])
		pred_states.append(pred_state)
		pred_sigs.append(pred_sig)
		pred_sig_matrix = np.diag(pred_sig[:2])
		if(i==0):
			state_pred = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
		# run skill for H timesteps
		min_log_prob = 1000000
		min_state = state
		for j in range(H):
			#env.render()
			if(True):
				action = model.decoder.ll_policy.numpy_policy(state,z)
				state,_,_,_ = env.step(action)
				action = model.decoder.ll_policy.numpy_policy(state_pred[0,0,:].detach().cpu().numpy(),z)
				action_torch = torch.reshape(torch.tensor(action,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
				state_pred_mean,state_pred_sig = dynamics_model(state_pred,action_torch)
				eps = torch.normal(torch.zeros(state_pred_mean.size()).cuda(), torch.ones(state_pred_mean.size()).cuda())
				state_pred = state_pred_mean #+ eps*state_pred_sig
				pred_s = state_pred[0,0,:].detach().cpu().numpy()
				ll_states = np.vstack([ll_states,pred_s])
				pred_s = pred_s[:2]
				log_prob = (np.expand_dims(pred_s.T-pred_state[:2].T,axis=0)@np.linalg.inv(pred_sig_matrix)@np.expand_dims(pred_s.T-pred_state[:2].T,axis=0).T)[0]
				if(log_prob<min_log_prob):
					min_log_prob = log_prob
					min_state = pred_s

			states = np.vstack([states,state])
			if(not use_dynamics_model):

				log_prob = (np.expand_dims(state[:2].T-pred_state[:2].T,axis=0)@np.linalg.inv(pred_sig_matrix)@np.expand_dims(state[:2].T-pred_state[:2].T,axis=0).T)[0]
				if(log_prob<min_log_prob):
					min_log_prob = log_prob
					min_state = state[:2]
			skill_seq_states.append(state)

		if(use_dynamics_model):
			pred_term_states_x.append(state_pred[0,0,0].detach().cpu().numpy())
			pred_term_states_y.append(state_pred[0,0,1].detach().cpu().numpy())
			pred_term_states[ITER,0] = min_state[0]#state_pred[0,0,0].detach().cpu().numpy()
			pred_term_states[ITER,1] = min_state[1]#state_pred[0,0,1].detach().cpu().numpy()

		if(not use_dynamics_model):

			term_states_x.append(state[0])
			term_states_y.append(state[1])
			term_states[ITER] = min_state[:2]
			
			if(plot_abstract):
				abs_stat.append(pred_state[:2])
				abs_stat.append(pred_sig[:2])

	#states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)

	return state,states,ll_states,abs_states,abs_sigs

device = torch.device('cuda:0')

env = 'antmaze-large-diverse-v0'
# env = 'antmaze-medium-diverse-v0'
#env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

skill_seq_len = 8#10
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
fixed_sig =  None#0.0
n_iters = 100
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
PATH = 'checkpoints/antmaze_skill_models/'+filename

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
goal_state = np.array(env.target_goal)
env.target_goal = goal_state
goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)

execute_n_skills = 8#10
fig, ax = plt.subplots()

print('RUNNING SKILLS WITH TRUE STATES')

init_state = env.reset()
init_state[:2] = np.array([0,0])

for j in range(10):
	state = env.reset()
	env.env.set_state(init_state[:15],init_state[15:])
	state = init_state

	for i in range(max_replans):
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
		if(j==0):
			skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
			skill_seq = skill_seq[:execute_n_skills,:]
			skill_seq = skill_seq.unsqueeze(0)	
			skill_seq = convert_epsilon_to_z(skill_seq,s_torch[:1,:,:],skill_model)
		if(j==0):
			state,states,ll_states,abs_states,abs_sigs = run_skill_seq(ax,skill_seq,env,state,skill_model,dynamics_model=dynamics_model,use_epsilon=False,plot_abstract=True,ITER=j)
		else:
			state,states,ll_states,abs_states,abs_sigs = run_skill_seq(ax,skill_seq,env,state,skill_model,dynamics_model=dynamics_model,use_epsilon=False,ITER=j)
		plt.plot(states[:,0],states[:,1],color='red')
		plt.plot(ll_states[:,0],ll_states[:,1],color='blue')

for i in range(execute_n_skills):
	n = 10000
	mean = [abs_states[1+i,0],abs_states[1+i,1]]
	cov = [[abs_sigs[i,0],0],[0,abs_sigs[i,1]]]
	rng = np.random.RandomState(0)
	x, y = rng.multivariate_normal(mean, cov, n).T
	sns.kdeplot(x=x, y=y, levels=4, color="green", fill=True)

img = plt.imread('plots/antmaze_plots/antmaze.jpeg')
plt.imshow(img,extent = [-6,42,-6,30])
#plt.scatter(7,9,c='purple')
plt.scatter(0,0,c='orange')
plt.scatter(env.target_goal[0],env.target_goal[1],c='purple',marker='x')
red_key = mlines.Line2D([], [], color='red', marker='s', ls='', label='True trajectories')
blue_key = mlines.Line2D([], [], color='blue', marker='s', ls='', label='Pred trajectories(LL dynamics)')
green_key = mlines.Line2D([], [], color='green', marker='s', ls='', label='Pred trajectory distr(Abs dynamics)')
orange_key = mlines.Line2D([], [], color='orange', marker='s', ls='', label='Initial state')
purple_key = mlines.Line2D([], [], color='purple', marker='s', ls='', label='Goal state')
plt.legend(handles=[red_key,green_key,blue_key,orange_key,purple_key],loc='lower right')
plt.xlim([-6,42])
plt.ylim([-6,30])
plt.axis('equal')
plt.savefig('plots/antmaze_pdf_plots/antmaze_final_goal.pdf')
