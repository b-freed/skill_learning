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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
	H = 80
	state_dim = 4
	a_dim = 2
	h_dim = 128
	z_dim = 20
	batch_size = 1
	epochs = 100000
	episodes = 5


	filename = 'maze2d_H'+str(H)+'_log.pth'
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

	env = 'maze2d-large-v1'
	env = gym.make(env)
	data = env.get_dataset()


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
pred_states = []
# collect an episode of data
for i in range(episodes):
	initial_state = env.reset()
	#actions = data['actions']
	#goals = data['infos/goal']
	initial_state = torch.reshape(torch.tensor(initial_state,dtype=torch.float32).cuda(), (1,1,4))
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
	# ipdb.set_trace()
	
	actual_states.append(states_actual)
	pred_states.append([sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy()])
	
	

actual_states = np.stack(actual_states)
pred_states = np.stack(pred_states)
plt.figure()
plt.scatter(actual_states[:,:,0],actual_states[:,:,1], c='r')
plt.scatter(actual_states[:,0,0],actual_states[:,0,1], c='b')
plt.scatter(pred_states[:,0],pred_states[:,1], c='g')
plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
plt.title('Skill Execution & Prediction (Skill-Dependent Prior)')
plt.savefig('Skill_Prediction_H'+str(H)+'.png')
