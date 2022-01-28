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


def run_skill(skill_model,s0,skill,env,H):
	state = s0.flatten().detach().cpu().numpy()
	states = []
	
	actions = []
	for j in range(H-1):
	    action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
	    actions.append(action)
	    state,_,_,_ = env.step(action)
	    
	    states.append(state)
	  
	return np.stack(states),np.stack(actions)

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
	# ll_policy = skill_model_sdp.decoder.ll_policy


	# filename = 'z_dim_4.pth' #'z_dim_4.pth'#'log.pth'
	# PATH = 'checkpoints/'+filename
	# skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	# checkpoint = torch.load(PATH)
	# skill_model.load_state_dict(checkpoint['model_state_dict'])
	# ll_policy = skill_model.decoder.ll_policy


	# # start out in s0
	# s0 = torch.zeros((batch_size,1,state_dim), device=device)
	# skill_seq = torch.randn((1,episodes,z_dim), device=device)

	# for i in range(episodes):
	# 	# sample a skill
	# 	skill = skill_seq[:,i:i+1,:]
	# 	# predict outcome of that skill
	# 	pred_next_s_mean, pred_next_s_sig = skill_model.decoder.abstract_dynamics(s0,skill)

	# 	# run skill in real world
	# 	state_seq = run_skill(s0,skill,env,H)
	# 	print('pred_next_s_sig: ', pred_next_s_sig)
	# 	# circle = plt.Circle((pred_next_s_sig[0,0,0].item(),pred_next_s_sig[0,0,1].item()), 0), 0.2, color='r')

	# 	plt.scatter(pred_next_s_mean[0,0,0].flatten().cpu().detach().numpy(),pred_next_s_mean[0,0,1].flatten().cpu().detach().numpy())
	# 	plt.scatter(state_seq[:,0],state_seq[:,1])
	# 	plt.show()





actual_states = []
pred_states_sig = []
pred_states_mean = []
action_dist = []
# collect an episode of data
for i in range(episodes):
	initial_state = env.reset()
	#actions = data['actions']
	#goals = data['infos/goal']
	initial_state = torch.reshape(torch.tensor(initial_state,dtype=torch.float32).cuda(), (1,1,state_dim))
	#actions = torch.tensor(actions,dtype=torch.float32).cuda()

	z_mean,z_sig = skill_model_sdp.prior(initial_state)

	z = skill_model_sdp.reparameterize(z_mean,z_sig)
	sT_mean,sT_sig = skill_model_sdp.decoder.abstract_dynamics(initial_state,z)
	#ipdb.set_trace()
	

# 	# infer the skill
# 	z_mean,z_sig = skill_model.encoder(states,actions)

# 	z = skill_model.reparameterize(z_mean,z_sig)

# 	# from the skill, predict the actions and terminal state
# 	# sT_mean,sT_sig,a_mean,a_sig = skill_model.decoder(states,z)
# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(states[:,0:1,:],z)
	

	states_actual,actions = run_skill(skill_model_sdp, initial_state,z,env,H)
	# states_actual,actions = run_skill_with_disturbance(skill_model_sdp, states[:,0:1,:],z,env,H)
	'''
	plt.figure()
	plt.scatter(states_actual[:,0],states_actual[:,1])
	plt.scatter(states_actual[0,0],states_actual[0,1])
	plt.errorbar(sT_mean[0,0,0].detach().cpu().numpy(),sT_mean[0,0,1].detach().cpu().numpy(),xerr=sT_sig[0,0,0].detach().cpu().numpy(),yerr=sT_sig[0,0,1].detach().cpu().numpy())

	plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
	plt.title('Skill Execution & Prediction (Skill-Dependent Prior) '+str(i))
	plt.savefig('Skill_Prediction_H'+str(H)+'_'+str(i)+'.png')
	'''
	
	actual_states.append(states_actual)
	action_dist.append(actions)
	pred_states_mean.append([sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy()])
	pred_states_sig.append([sT_sig[0,-1,0].detach().cpu().numpy(),sT_sig[0,-1,1].detach().cpu().numpy()])
	
	

actual_states = np.stack(actual_states)
pred_states_sig = np.stack(pred_states_sig)
pred_states_mean = np.stack(pred_states_mean)

# x = u + a cos(t) ; y = v + b sin(t)
plt.figure()
for i in range(episodes):
	u = pred_states_mean[i,0]       #x-position of the center
	v = pred_states_mean[i,1]       #y-position of the center
	a = (pred_states_sig[i,0])/2    #radius on the x-axis
	b = (pred_states_sig[i,1])/2    #radius on the y-axis

	t = np.linspace(0, 2*pi, 100)

	plt.plot( u+a*np.cos(t) , v+b*np.sin(t)) #label='Std dev of Predicted terminal states'
#plt.scatter(u,v, c='g')
plt.grid(color='lightgray',linestyle='--')

plt.scatter(actual_states[:,:,0],actual_states[:,:,1], c='r', label='Actual Trajectory')
plt.scatter(actual_states[:,0,0],actual_states[:,0,1], c='b', label='Initial State')
plt.scatter(pred_states_mean[:,0],pred_states_mean[:,1], c='g', label='Mean of Predicted terminal states')

plt.legend()
plt.title('Skill Execution & Prediction (Skill-Dependent Prior)')
plt.savefig('Skill_Prediction_H'+str(H)+'.png')

#ipdb.set_trace()
#plt.figure()
#plt.scatter(actual_states[:,:,0],actual_states[:,:,1], c='r')
#plt.scatter(actual_states[:,0,0],actual_states[:,0,1], c='b')
#plt.scatter(pred_states[:,0],pred_states[:,1], c='g')
#plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
#plt.title('Skill Execution & Prediction (Skill-Dependent Prior)')
#plt.savefig('Skill_Prediction_H'+str(H)+'.png')

#action_dist = np.stack(action_dist)
