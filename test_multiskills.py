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
	states = [state]
	
	actions = []
	for j in range(H):
	    action = skill_model.decoder.ll_policy.numpy_policy(state,skill)
	    actions.append(action)
	    state,_,_,_ = env.step(action)
	    
	    states.append(state)
	  
	return np.stack(states),np.stack(actions)


if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	#env = 'antmaze-medium-diverse-v0'
	env = 'maze2d-large-v1'
	env = gym.make(env)
	data = env.get_dataset()
	
	H = 40
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 256
	batch_size = 1
	episodes = 3
	#wd = 0.001


	#filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
	filename = 'maze2d_H'+str(H)+'_log_best.pth'
	PATH = 'checkpoints/'+filename
	skill_model_sdp = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda() #SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	checkpoint = torch.load(PATH)
	skill_model_sdp.load_state_dict(checkpoint['model_state_dict'])



actual_states = []
pred_states_sig = []
pred_states_mean = []
# collect an episode of data
initial_state = env.reset()
for i in range(episodes):
	
	initial_state = torch.reshape(torch.tensor(initial_state,dtype=torch.float32).cuda(), (1,1,state_dim))

	z_mean,z_sig = skill_model_sdp.prior(initial_state)

	z = skill_model_sdp.reparameterize(z_mean,z_sig)
	sT_mean,sT_sig = skill_model_sdp.decoder.abstract_dynamics(initial_state,z)
	#ipdb.set_trace()

	states_actual,actions = run_skill(skill_model_sdp, initial_state,z,env,H)
	
	actual_states.append(states_actual)
	pred_states_mean.append([sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy()])
	pred_states_sig.append([sT_sig[0,-1,0].detach().cpu().numpy(),sT_sig[0,-1,1].detach().cpu().numpy()])
	
	initial_state = states_actual[-1,:]
	
	

actual_states = np.stack(actual_states)
pred_states_sig = np.stack(pred_states_sig)
pred_states_mean = np.stack(pred_states_mean)

# x = u + a cos(t) ; y = v + b sin(t)
plt.figure()
for i in range(episodes):
	u = pred_states_mean[i,0]       #x-position of the center
	v = pred_states_mean[i,1]       #y-position of the center
	a = (pred_states_sig[i,0])      #radius on the x-axis
	b = (pred_states_sig[i,1])      #radius on the y-axis

	t = np.linspace(0, 2*pi, 100)

	plt.plot( u+a*np.cos(t) , v+b*np.sin(t)) #label='Std dev of Predicted terminal states'
#plt.scatter(u,v, c='g')
plt.grid(color='lightgray',linestyle='--')

plt.scatter(actual_states[:,:,0],actual_states[:,:,1], c='r', label='Actual Trajectory')
plt.scatter(actual_states[:,0,0],actual_states[:,0,1], c='b', marker='x', label='Initial State')
plt.scatter(pred_states_mean[:,0],pred_states_mean[:,1], c='g', label='Mean of Predicted terminal states')

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol= 3)
plt.title('Multi-Skill Execution & Prediction (Skill-Dependent Prior)')
plt.savefig('Multi-Skill_Prediction_H'+str(H)+'.png')

#ipdb.set_trace()
#plt.figure()
#plt.scatter(actual_states[:,:,0],actual_states[:,:,1], c='r')
#plt.scatter(actual_states[:,0,0],actual_states[:,0,1], c='b')
#plt.scatter(pred_states[:,0],pred_states[:,1], c='g')
#plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
#plt.title('Skill Execution & Prediction (Skill-Dependent Prior)')
#plt.savefig('Skill_Prediction_H'+str(H)+'.png')
