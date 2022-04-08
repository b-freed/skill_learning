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
from utils import chunks
from train_skills_maze2d import train, test
from tokenize import ContStr
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem, cem_variable_length
from utils import make_gif,make_video

def offline_training():
    
    for i in range(n_epochs):
        loss, s_T_loss, a_loss, kl_loss, s_T_ent = train(model,model_optimizer)
        
        print("--------TRAIN---------")
        
        print('loss: ', loss)
        print('s_T_loss: ', s_T_loss)
        print('a_loss: ', a_loss)
        print('kl_loss: ', kl_loss)
        print('s_T_ent: ', s_T_ent)
        print(i)
        experiment.log_metric("loss", loss, step=i)
        experiment.log_metric("s_T_loss", s_T_loss, step=i)
        experiment.log_metric("a_loss", a_loss, step=i)
        experiment.log_metric("kl_loss", kl_loss, step=i)
        experiment.log_metric("s_T_ent", s_T_ent, step=i)


        test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent = test(model)
        
        print("--------TEST---------")
        
        print('test_loss: ', test_loss)
        print('test_s_T_loss: ', test_s_T_loss)
        print('test_a_loss: ', test_a_loss)
        print('test_kl_loss: ', test_kl_loss)
        print('test_s_T_ent: ', test_s_T_ent)

        print(i)
        experiment.log_metric("test_loss", test_loss, step=i)
        experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
        experiment.log_metric("test_a_loss", test_a_loss, step=i)
        experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
        experiment.log_metric("test_s_T_ent", test_s_T_ent, step=i)

        if i % 10 == 0:
            
                
            checkpoint_path = 'checkpoints/'+ filename + '.pth'
            torch.save({
                                'model_state_dict': model.state_dict(),
                                'model_optimizer_state_dict': model_optimizer.state_dict(),
                                }, checkpoint_path)
        if test_loss < min_test_loss:
            min_test_loss = test_loss

            
                
            checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
            torch.save({'model_state_dict': model.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)
    
    return loss, test_loss, checkpoint_path

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

def run_skills_iterative_replanning(env,model,goals,use_epsilon,replan_freq,variable_length,ep_num):
	
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
		state_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))])
		
		if variable_length:
			cost_fn = lambda skill_seq,lengths: skill_model.get_expected_cost_variable_length(state_torch, skill_seq, lengths, goal_seq, use_epsilons=use_epsilon)
			skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
			skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			p_lengths = (1/(skill_seq_len)) * torch.ones(skill_seq_len+1,device=device)
			p_lengths[0] = 0.0
		
		
			skill_seq_mean,skill_seq_std = cem_variable_length(skill_seq_mean,skill_seq_std,p_lengths,cost_fn,batch_size,keep_frac,n_iters,max_ep=max_ep,l2_pen=cem_l2_pen)

		else:
			cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(state_torch, skill_seq, goal_seq, use_epsilons=use_epsilon,length_cost=plan_length_cost)
			if n == 0:
				skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
				skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			else:
				skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
				skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			
				# skill_seq_mean = torch.cat([skill_seq_mean[1:,:],torch.zeros((1,z_dim),device=device)])
				# skill_seq_std  = torch.cat([skill_seq_std[1:,:], torch.ones((1,z_dim),device=device)])
		
			# 								           x_mean,        x_std,cost_fn,  pop_size,frac_keep,n_iters,l2_pen
			skill_seq_mean,skill_seq_std = cem(skill_seq_mean,skill_seq_std,cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)



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
		print('executing skill')
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
		plt.savefig('ant_iterative_replanning_actual_states_niters'+str(n_iters)+'_l2pen_'+str(cem_l2_pen)+'.png')
	
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


def run_skill_seq(skill_seq,env,s0,model,use_epsilon):
	'''
	'''
	# env.env.set_state(s0[:2],s0[2:])
	state = s0
	# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape((1,1,-1))

	pred_states = []
	pred_sigs = []
	states = []
	actions = []
	plt.figure()
	for i in range(skill_seq.shape[1]):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		else:
			z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		skill_seq_actions = []
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
			tensor_state = torch.tensor(state,dtype=torch.float32)
			tensor_action = torch.tensor(action,dtype=torch.float32)
			skill_seq_states.append(tensor_state)
			skill_seq_actions.append(tensor_action)
			plt.scatter(state[0],state[1], label='Trajectory',c='b')
		states.append(skill_seq_states)
		actions.append(skill_seq_actions)
		plt.scatter(state[0],state[1], label='Term state',c='r')
		plt.scatter(pred_state[0],pred_state[1], label='Pred states',c='g')
		plt.errorbar(pred_state[0],pred_state[1],xerr=pred_sig[0],yerr=pred_sig[1],c='g')
		

	states = np.stack(states)
	actions = np.stack(actions)
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
	plt.xlim([-1,38])
	plt.ylim([-1,30])


	if not state_dependent_prior:
		plt.title('Planned Skills (No State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'_sdp_'+'false'+'.png')
	else:
		plt.title('Planned skills (State Dependent Prior)')
		plt.savefig('Skill_planning_H'+str(H)+'.png')

	# print('SAVED FIG!')

	return state,states, actions

def create_online_dataset(new_states,new_actions,train_loader):


	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]  # first state_dim elements are the state
		actions = data[:,:,model.state_dim:]

		states = torch.utils.data.ConcatDataset([states, new_states])
		actions = torch.utils.data.ConcatDataset([actions, new_actions])

	new_inputs_train = torch.cat([states, actions],dim=-1)
	train_loader = DataLoader(new_inputs_train,
								batch_size=batch_size,
								num_workers=0)

	return train_loader



env_name = 'antmaze-large-diverse-v0'
env = gym.make(env_name)

dataset_file = None

if dataset_file is None:
	dataset = d4rl.qlearning_dataset(env) #env.get_dataset()
else:
	dataset = d4rl.qlearning_dataset(env,h5py.File(dataset_file, "r"))

batch_size = 100

h_dim = 256
z_dim = 256
lr = 5e-5
wd = 0.0
state_dependent_prior = True
term_state_dependent_prior = False
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0.0
max_sig = None
fixed_sig = None
H = 40
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'normal' # 'tanh_normal' or 'normal'
encoder_type = 'state_action_sequence' #'state_sequence'
state_decoder_type = 'mlp' #'autoregressive'
n_iters = 100
use_epsilon = True
max_ep = None
cem_l2_pen = 0.0
var_pen = 0.0
render = False
variable_length = False
max_replans = 200
plan_length_cost = 0.0
init_state_dependent = True

experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')

if term_state_dependent_prior:
	model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,state_decoder_type=state_decoder_type).cuda()

else:
	model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
	
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

filename = env_name+'_enc_type_'+str(encoder_type)+'state_dec_'+str(state_decoder_type)+'_H_'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_sg_'+str(state_dec_stop_grad)+'_max_sig_'+str(max_sig)+'_fixed_sig_'+str(fixed_sig)+'_ent_pen_'+str(ent_pen)+'_log'

if term_state_dependent_prior:
	filename = env_name+'_tsdp'+'_H'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_sg_'+str(state_dec_stop_grad)+'_max_sig_'+str(max_sig)+'_fixed_sig_'+str(fixed_sig)+'_log'


experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'state_dependent_prior':state_dependent_prior,
							'term_state_dependent_prior':term_state_dependent_prior,
							'z_dim':z_dim,
							'H':H,
							'a_dist':a_dist,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'state_dec_stop_grad':state_dec_stop_grad,
							'beta':beta,
							'alpha':alpha,
							'max_sig':max_sig,
							'fixed_sig':fixed_sig,
							'ent_pen':ent_pen,
							'env_name':env_name,
							'filename':filename,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type})
experiment.add_tag('online learning')

print('##### Making offline dataset for training model #####')
states = dataset['observations']
next_states = dataset['next_observations']
actions = dataset['actions']

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

states_train  = states[:N_train,:]
next_states_train = next_states[:N_train,:]
actions_train = actions[:N_train,:]


states_test  = states[N_train:,:]
next_states_test = next_states[N_train:,:]
actions_test = actions[N_train:,:]

assert states_train.shape[0] == N_train
assert states_test.shape[0] == N_test
										            #  obs,next_obs,actions,H,stride
obs_chunks_train, action_chunks_train = chunks(states_train, next_states_train, actions_train, H, stride)

print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')

obs_chunks_test,  action_chunks_test  = chunks(states_test,  next_states_test,  actions_test,  H, stride)


inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10

print('--------Offline training starting----------')

train_loss, test_loss, PATH = offline_training()

print('--------Offline training done----------')

checkpoint = torch.load(PATH)
skill_model = model
skill_model.load_state_dict(checkpoint['model_state_dict'])

print('------Planning------')
s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

goal_state = np.array(env.target_goal)
print('goal_state: ', goal_state)
goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)


execute_n_skills = 1

min_dists_list = []
total_rewards = []
for j in range(100):
	state = env.reset()
	goal_loc = goal_state[:2]
	min_dist = 10**10
	for i in range(max_replans):
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
		skill_seq,_,cost = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
		skill_seq = skill_seq[:execute_n_skills,:]
		skill_seq = skill_seq.unsqueeze(0)	
		skill_seq = convert_epsilon_to_z(skill_seq,s_torch[:1,:,:],skill_model)
		state,states,actions = run_skill_seq(skill_seq,env,state,skill_model,use_epsilon=False)
		print('states.shape: ', states.shape)

		dists = np.sum((states[0,:,:2] - goal_loc)**2,axis=-1)

		if np.min(dists) < min_dist:
			min_dist = np.min(dists)

		if min_dist < 1.0:
			break
	
	min_dists_list.append(min_dist)
	np.save('min_dists_list', min_dists_list)
	print('min_dists_list: ', min_dists_list)

	total_rewards.append(cost)
	print('Rewards_list:', total_rewards)

print('------Planning done------')

print('Creating new dataset with states from planning')

train_loader = create_online_dataset(states, actions, train_loader)

print('Done')

print('Training new model on online dataset')

if term_state_dependent_prior:
	new_model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	new_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,state_decoder_type=state_decoder_type).cuda()

else:
	new_model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
	
new_model_optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, weight_decay=wd)


for i in range(n_epochs):
	new_loss, new_s_T_loss, new_a_loss, new_kl_loss, new_s_T_ent = train(new_model,new_model_optimizer)
	
	print("--------TRAIN---------")
	
	print('new_loss: ', new_loss)
	print('new_s_T_loss: ', new_s_T_loss)
	print('new_a_loss: ', new_a_loss)
	print('new_kl_loss: ', new_kl_loss)
	print('new_s_T_ent: ', new_s_T_ent)
	print(i)
	experiment.log_metric("new_loss", new_loss, step=i)
	experiment.log_metric("new_s_T_loss", new_s_T_loss, step=i)
	experiment.log_metric("new_a_loss", new_a_loss, step=i)
	experiment.log_metric("new_kl_loss", new_kl_loss, step=i)
	experiment.log_metric("new_s_T_ent", new_s_T_ent, step=i)
	






