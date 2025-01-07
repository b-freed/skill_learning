from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillModelTerminalStateDependentPrior
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
from utils import chunks, reparameterize
import config
import os
import matplotlib.pyplot as plt
from utils import make_video, z_score

from franka_reward_fn import franka_plan_cost_fn


def set_arm_position(env,pos):
    state = env.sim.get_state()
    state[:30] = pos
    env.sim.set_state(state)
    # ipdb.set_trace()
    env.sim.forward()


def visualize_state_seq(states):
    env = gym.make('kitchen-partial-v0')
    frames = []
    print('IN VISUALIZinFG STATE SEQ')
    pos0 = states[0,:30]
    for i in range(states.shape[0]):

        print('i: ', i)
        pos = states[i,:30]
        # print('pos[29]: ', pos[29])
        # pos[np.abs(pos-pos0) < 1e-3] = pos0
        sim_state = env.sim.get_state()
        sim_state[:30] = pos
        sim_state[30:] = 0
        env.sim.set_state(sim_state)
        env.sim.forward()

        print(env.sim.get_state()[29])
        frames.append(env.render(mode='rgb_array'))
    make_video(frames,'franka_visualize_state_seq')

    



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
		print('r: ', r)
		# r_fn = reward_fn.step(state)
		# print('r_fn: ', r_fn)
		
		states.append(state)
		rewards.append(r)

	return state,np.stack(states),np.stack(actions),np.sum(rewards),frames


device = torch.device('cuda:0')
	
# env = 'antmaze-medium-diverse-v0'
env = 'kitchen-partial-v0'
# env = 'kitchen-complete-v0'
env = gym.make(env)
# env = FrankaSliceWrapper(gym.make(env))


data = env.get_dataset()

H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
# episodes = 4
wd = .01
beta = 1.0
alpha = 1.0
state_dependent_prior = True
a_dist = 'autoregressive'
max_sig = None
fixed_sig = None
state_dec_stop_grad = False # I don't think this really matters
skill_seq_len = 10
keep_frac = 0.5
n_iters = 100
n_additional_iters = 20
cem_l2_pen = 0.005
replan_iters = 1000
per_element_sigma = False
render = True
initial_ind = 114075
n_skills = 4


# load skill model
# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best_sT.pth'
# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.5_per_el_sig_False_log_best_sT.pth'
# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_log_best_sT.pth'

# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best.pth'
# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_log_best_sT.pth'
# filename = 'EM_model_kitchen-partial-v0_sliced_state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best_sT.pth'

# filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best_sT.pth'
filename = 'EM_model_kitchen-partial-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_0.1_per_el_sig_False_a_dist_autoregressive_log_best.pth'


PATH = 'checkpoints/'+filename



if not state_dependent_prior:
    skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
else:
    skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,per_element_sigma=per_element_sigma).cuda()
    
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# take slice of data we know flips light switch
data = env.get_dataset()
states = data['observations']
s_dim = states.shape[1]
actions = data['actions']


# visualize_state_seq(states[initial_ind:initial_ind+n_skills*H,:])



env.reset()
# set_arm_position(env,states[0,:30])

# for i in range(10):
#     print('dataset state: ', states[i,:])
#     print('env.sim.get_state(): ', env.sim.get_state())
#     state,r,_,_ = env.step(actions[i,:])
#     print('state: ', state)
    

# ipdb.set_trace()



for j in range(1):
    print('j: ',j)
    state = env.reset()
    s0 = states[initial_ind,:]
    print('s0: ', s0)
    sim_state = env.sim.get_state()
    sim_state[:30] = s0[:30]
    env.sim.set_state(sim_state)
    env.sim.forward()
    print('env.sim.get_state(): ',env.sim.get_state())
    frames = []
    states_actual = []
    actions_actual = []
    state = s0
    for i in range(n_skills):
        states_subtraj  = states[ initial_ind+H*i:initial_ind+H*(i+1),:]
        actions_subtraj = actions[initial_ind+H*i:initial_ind+H*(i+1),:]
        print('states_subtraj.shape: ',states_subtraj.shape)

        states_subtraj = torch.tensor(states_subtraj,device=torch.device('cuda:0')).unsqueeze(0)
        actions_subtraj = torch.tensor(actions_subtraj,device=torch.device('cuda:0')).unsqueeze(0)

        print('states_subtraj.shape: ',states_subtraj.shape)

        # pass thru encoder, sample skill
        z_post_means,z_post_sigs = skill_model.encoder(states_subtraj,actions_subtraj)

        # get prior
        z_prior_means,z_prior_sigs = skill_model.prior(states_subtraj[:,0:1,:])


        # print('z_post_means: ', z_post_means)
        # print('z_post_sigs: ', z_post_sigs)
        print('z_prior_means: ', z_prior_means)
        print('z_prior_sigs: ', z_prior_sigs)
        print('z score: ', z_score(z_post_means,z_prior_means,z_prior_sigs))


        z = reparameterize(z_post_means,z_post_sigs)
        print('z: ',z)
        print('z score of z: ', z_score(z,z_prior_means,z_prior_sigs))


        assert False
        # pass thru decoder
        s_T_mean, s_T_sig, a_means, a_sigs = skill_model.decoder(states_subtraj,actions_subtraj,z)

        # get s dist and a dist
        # a_dist = Normal.Normal(a_means,a_sigs)
        # sT_dist = Normal.Normal(s_T_mean,s_T_sig)

        # a_ll = torch.mean(torch.sum(a_dist.log_prob(actions_subtraj),dim=-1))
        # sT_ll = torch.mean(torch.sum(sT_dist.log_prob(states_subtraj[:,-1:,:]),dim=-1))

        # print('a_ll: ', a_ll)
        # print('sT_ll: ', sT_ll)

        # evaluate log likelihood


        state,skill_states,skill_actions,_,skill_frames = run_skill(skill_model,state,z,env,H,render,use_epsilon=False)
        frames += skill_frames
        # print('skill_states.shape: ', skill_states.shape)
        # print('states_actual.shape: ', states_actual.shape)
        states_actual.append(skill_states)
        actions_actual.append(skill_actions)


        make_video(frames,'debug_franka_sampling_from_post_'+str(j))

states_actual = np.concatenate(states_actual)
actions_actual = np.concatenate(actions_actual)

# for i in range(s_dim):
#     plt.figure()
#     plt.plot(states[initial_ind:initial_ind+states_actual.shape[0],i])
#     plt.plot(states_actual[:,i])
#     plt.legend(['data','actual'])
#     plt.savefig('debug_franka_states_'+str(i))




for i in range(a_dim):
    plt.figure()
    plt.plot(a_means[...,i].squeeze().detach().cpu().numpy())
    plt.plot(actions_subtraj[...,i].squeeze().detach().cpu().numpy())

    plt.savefig('debug_franka_actions_'+str(i))

# for i in range(state_dim):
#     plt.figure()
#     plt.scatter(40,s_T_mean[...,i].squeeze().detach().cpu().numpy())
#     plt.plot(states_subtraj[...,i].squeeze().detach().cpu().numpy())

#     plt.savefig('debug_franka_states_'+str(i))

# dists = np.linalg.norm(states[150:190,17:19] - np.array([-0.69, -0.05]),axis=-1)

# pred_dist = np.linalg.norm(s_T_mean[...,17:19].squeeze().detach().cpu().numpy() - np.array([-0.69, -0.05]))


# ipdb.set_trace()

# run skill