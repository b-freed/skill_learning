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
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from utils import make_gif, make_video
from franka_reward_fn import FrankaRewardFn,franka_plan_cost_fn,franka_plan_cost_fn_mppi
from cem import cem
from mppi import mppi
from random import sample
#from wrappers import FrankaSliceWrapper

def run_skill(skill_model,s0,skill,env,H,render,use_epsilon=False):
	try:
		state = s0.flatten().detach().cpu().numpy()
	except:
		state = s0


	if use_epsilon:
		mu_z, sigma_z = skill_model.prior(torch.tensor(s0,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0))
		z = mu_z + sigma_z*skill
	else:
		z = skill
	
	states = [state]
	
	actions = []
	frames = []
	rewards = []
	rew_fn_rewards = []
	for j in range(H): #H-1 if H!=1
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,z)
		actions.append(action)
		state,r,_,_ = env.step(action)
		#print('r: ', r)
		# r_fn = reward_fn.step(state)
		# print('r_fn: ', r_fn)
		
		states.append(state)
		rewards.append(r)

	return state,np.stack(states),np.stack(actions),np.sum(rewards),frames

def run_skill_with_disturbance(skill_model,s0,skill,env,H):
	state = s0.flatten().detach().cpu().numpy()
	states = []
	
	actions = []
	for j in range(H-1):
		action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
		
		if j == H//2:
			print('adding disturbance!!!!')
			action = action + 20
		actions.append(action)
		state,_,_,_ = env.step(action)
		
		states.append(state)
	  
	return np.stack(states),np.stack(actions)



if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	# env = 'antmaze-medium-diverse-v0'
	#env_name = 'kitchen-partial-v0'
	env_name = 'kitchen-mixed-v0'
	#env_name = 'kitchen-complete-v0'
	env = gym.make(env_name)
	# env = FrankaSliceWrapper(gym.make(env))


	data = env.get_dataset()
	
	H = 10#10
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	batch_size = 1000#1000
	# episodes = 4
	wd = .01
	beta = 1.0
	alpha = 1.0
	state_dependent_prior = True
	a_dist = 'normal'#'autoregressive' #'normal'
	max_sig = None
	fixed_sig = 0.0#None
	state_dec_stop_grad = False # I don't think this really matters
	skill_seq_len = 10#10#50
	keep_frac = 0.1
	n_iters = 5
	n_additional_iters = 5
	cem_l2_pen = 0
	replan_iters = 300//H#500//H
	per_element_sigma = False
	render = False
	warm_start = False

	#filename = 'EM_model_05_24_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best.pth'

	#filename = 'franka_skills/EM_model_06_03_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.01_a_1.0_b_1.0_per_el_sig_True_a_dist_autoregressive_log_best_sT.pth'
	#filename = 'franka_skills/EM_model_06_04_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.01_a_1.0_b_1.0_per_el_sig_True_a_dist_autoregressive_log_best_decay_100.pth'
	#filename = 'franka_skills/EM_model_06_04_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.01_a_1.0_b_1.0_per_el_sig_True_a_dist_autoregressive_log_best_decay_250.pth'

	#filename = 'franka_skills/EM_model_06_04_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best_decay_250.pth'
	#BEST ONE
	#filename = 'franka_skills/EM_model_06_05_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_best_decay_250.pth'

	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_15_zdim_64_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.1_decay_300_best_sT.pth'
	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_15_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.1_decay_300_best_sT.pth'
	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_zdim_16_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.1_decay_300_best_sT.pth'

	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_300_best_sT.pth'
	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_zdim_128_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_300_best.pth'
	#filename = 'franka_skill_models/EM_model_06_06_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_a_1.0_b_10.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_300_best.pth'

	#filename = 'franka_skill_models2/EM_model_06_07_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_20_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_300_best.pth'
	#filename = 'franka_skill_models2/EM_model_06_07_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_20_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_300_best.pth'
	
	#filename = 'franka_skill_models2/EM_model_06_08_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_300_best.pth'
	#BEST
	filename = 'franka_skill_models2/EM_model_06_08_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_300_best_sT.pth'
	#filename = 'franka_skill_models2/EM_model_06_07_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_20_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_300_best.pth'
	#filename = 'franka_skill_models2/EM_model_06_07_22_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_20_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_300_best.pth'
	#filename = 'franka_skill_models2/EM_model_06_08_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_30_zdim_256_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_650_best_sT.pth'	

	#filename = 'franka_mixed_skill_models/EM_model_06_14_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_10_zdim_256_l2reg_0_lr_5e-05_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.2_decay_1500_best_a.pth'
	#BEST
	#filename = 'franka_complete_skill_models/EM_model_06_14_22_kitchen-complete-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0_lr_5e-05_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.1_decay_1500_best_a_old.pth'

	#filename = 'franka_mixed_skill_models2/EM_model_06_15_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_lr_5e-05_a_1.0_b_1.0_per_el_sig_False_a_dist_autoregressive_log_test_split_0.05_decay_1500_best_a.pth'
	#filename = 'franka_mixed_skill_models2/EM_model_06_15_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_5_zdim_256_l2reg_0.0_lr_5e-05_a_1.0_b_1.0_per_el_sig_False_a_dist_normal_log_test_split_0.2_decay_1500_best_a.pth'
	#filename = 'franka_mixed_skill_models2/EM_model_06_15_22_kitchen-mixed-v0state_dec_mlp_init_state_dep_True_H_20_zdim_256_l2reg_0.0_lr_5e-05_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_test_split_0.05_decay_1500_best_a.pth'

	PATH = 'checkpoints/'+filename
	
	if not state_dependent_prior:
		skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	else:
		skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,per_element_sigma=per_element_sigma).cuda()
		
	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])
	

	experiment = Experiment(api_key = 'wb7Q8i6WX2nuxXfHJdBSs9PUV', project_name = 'skill-learning')
	experiment.log_parameters({'env_name':env_name,
						   'filename':filename,
						   'H':H,
						   'fixed_sig':fixed_sig,
						   'max_sig':max_sig,
						   'cem_l2_pen':cem_l2_pen,
						   'n_iters':n_iters,
						   'n_additional_iters':n_additional_iters,
						   'skill_seq_len':skill_seq_len,
						   'warm_start':warm_start
						  })

	ep_rewards = []
	random_goals = False
	tasks = ['bottom burner','top burner','light switch','slide cabinet','hinge cabinet','microwave','kettle']

	for j in range(1000):
		H=10
		if(random_goals):
			ep_tasks = sample(tasks,4)
			franka_reward_fn = FrankaRewardFn(ep_tasks)
			print('TASKS: ',ep_tasks)
		initial_state = env.reset()
		prev_states = np.expand_dims(initial_state,0)  # add dummy time dimension
		frames = []
		rewards = []
		t_since_last_save = 0
		state = initial_state
		for i in range(replan_iters):
			print('==============================================')
			print('i: ', i)
			if(random_goals):
				cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True,ep_tasks=ep_tasks)
			else:
				cost_fn = lambda skill_seq: franka_plan_cost_fn(prev_states,skill_seq,skill_model,use_eps=True)
			#cost_fn = lambda skill_seq: franka_plan_cost_fn_mppi(prev_states,skill_seq,skill_model,use_eps=True)
			# run CEM on this cost fn
			if i == 0:
				skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
				#skill_seq = mppi(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
			else:
				if not warm_start: 
					skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)		
					#skill_seq = mppi(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
				else:
					skill_seq = torch.cat([skill_seq[1:,:],torch.zeros_like(skill_seq[:1,:])])
					init_sig =  torch.ones_like(skill_seq)
					init_sig[:-1,:] = .1*init_sig[:-1,:]

					skill_seq,_ = cem(skill_seq,init_sig,cost_fn,batch_size,keep_frac,n_additional_iters,l2_pen=cem_l2_pen)

			z = skill_seq[:1,:]
			# skill_seq = skill_seq[1:,:]
			
			state,states_actual,actions,skill_rewards,skill_frames = run_skill(skill_model, state,z,env,H,render,use_epsilon=True)
			
			if(random_goals):
				skill_rewards = 0.0
				for h in range(H):
					r,_ = franka_reward_fn.step(states_actual[h])
					skill_rewards += r

			prev_states = np.concatenate([prev_states,states_actual[1:,:]],axis=0)
			print('states_actual.shape: ', states_actual.shape)
			frames += skill_frames
			rewards.append(skill_rewards)
			print('np.sum(rewards): ', np.sum(rewards))
			if(np.sum(rewards)==2):
				H=5
			if(np.sum(rewards)==3):
				break

		experiment.log_metric('reward',np.sum(rewards),step=j)
		ep_rewards.append(np.sum(rewards))
		print('EPISODE: ',j)
		print('MEAN REWARD: ',sum(ep_rewards)/(j+1))
		print('STD DEV: ',np.std(ep_rewards,ddof=1))
		experiment.log_metric('mean_reward',np.mean(ep_rewards),step=j)
