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
	H = 100
	state_dim = 4
	a_dim = 2
	h_dim = 128
	z_dim = 4
	batch_size = 1
	epochs = 100000
	episodes = 5


	filename = 'maze2d_state_dep_prior_log.pth'
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






# collect an episode of data
for i in range(episodes):
	initial_state = env.reset()
	#actions = data['actions']
	#goals = data['infos/goal']
	initial_state = torch.tensor(initial_state,dtype=torch.float32).cuda()
	#actions = torch.tensor(actions,dtype=torch.float32).cuda()

	z_mean,z_sig = skill_model_sdp.prior(initial_state)

	z = skill_model_sdp.reparameterize(z_mean,z_sig)
	sT_mean,sT_sig = skill_model_sdp.decoder.abstract_dynamics(initial_state,z)
	print(sT_mean.shape)
	print(sT_sig.shape)
	ipdb.set_trace()
	

# 	# infer the skill
# 	z_mean,z_sig = skill_model.encoder(states,actions)

# 	z = skill_model.reparameterize(z_mean,z_sig)

# 	# from the skill, predict the actions and terminal state
# 	# sT_mean,sT_sig,a_mean,a_sig = skill_model.decoder(states,z)
# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(states[:,0:1,:],z)
	

	states_actual,actions = run_skill(skill_model_sdp, initial_state,z,env,H)
	# states_actual,actions = run_skill_with_disturbance(skill_model_sdp, states[:,0:1,:],z,env,H)
	# ipdb.set_trace()
	
	plt.figure()
	plt.scatter(states_actual[:,0],states_actual[:,1],c='r')
	plt.scatter(states_actual[0,0],states_actual[0,1],c='b')
	plt.scatter(sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy(),c='g')
	# plt.show()

plt.legend(['Actual Trajectory','Initial State','Predicted Terminal State'])
plt.title('Skill Execution & Prediction (Skill-Dependent Prior)')
plt.savefig('Skill Prediction.png')

	
# 	# plt.scatter(states_actual[:,0],states_actual[:,1])
	
	

# 	# # perturb z slightly and repeate
# 	# epsilon = torch.randn(torch.zeros_like(z),)
# 	# z = skill_model.reparameterize(z_mean+.1,z_sig+.1)
# 	# z = z + epsilon
# 	z = skill_model.reparameterize(torch.zeros_like(z_mean),torch.ones_like(z_sig))
# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(states[:,0:1,:],z)
# 	# print('sT_sig: ', sT_sig)
# 	print('z: ', z)
# 	print('initial state: ', states[:,0,:])

# 	states_actual = run_skill(states[:,0:1,:],z,env,H)
# 	plt.scatter(states_actual[:,0],states_actual[:,1])
# 	plt.scatter(sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy())



# 	plt.show()


# z:  tensor([[[-1.0387, -0.0625, -1.9619, -0.7790]]], device='cuda:0')
# initial state:  tensor([[-0.9506, -0.6553,  0.0000,  0.0000]], device='cuda:0')

# s0 = torch.tensor([[[-0.9506, -0.6553,  0.0000,  0.0000]]], device='cuda:0')
# s0 = torch.tensor([[[0.0000, 0.0000,  0.0000,  0.0000]]], device='cuda:0')

# z =  torch.tensor([[[-1.0387, -0.0625, -1.9619, -0.7790]]], device='cuda:0')


# for i in range(10):
# 	z = skill_model.reparameterize(torch.zeros_like(z),torch.ones_like(z))

# 	sT_mean,sT_sig = skill_model.decoder.abstract_dynamics(s0,z)
# 	states_actual,actions = run_skill(s0,z,env,H)
# 	# print('states_actual: ', states_actual)

# 	plt.scatter(sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy())
# 	plt.scatter(states_actual[:,0],states_actual[:,1])

	
	
# 	states_actual_dist,actions_dist = run_skill_with_disturbance(s0,z,env,H)
	

# 	# print('states_actual: ', states_actual)

# 	plt.scatter(sT_mean[0,-1,0].detach().cpu().numpy(),sT_mean[0,-1,1].detach().cpu().numpy())
# 	plt.scatter(states_actual_dist[:,0],states_actual_dist[:,1])
# 	plt.show()
	

# 	plt.plot(actions[:,0])
# 	plt.plot(actions[:,1])
# 	plt.plot(actions_dist[:,0])
# 	plt.plot(actions_dist[:,1])

# 	plt.show()
