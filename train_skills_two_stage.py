from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
# import torch.distributions.normal as Normal
# import torch.distributions.categorical as Categorical
# import torch.distributions.mixture_same_family as MixtureSameFamily
# import torch.distributions.kl as KL
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import matplotlib.pyplot as plt
from utils import chunks
# from skill_model import LowLevelPolicy,Encoder,Prior,AbstractDynamics,GenerativeModel
from two_stage_skill_model import TwoStageSkillModel
import os
from datetime import date



def train(model,train_loader):
	
	if model.stage == 'train_skills':
		losses, a_losses, kl_losses= [],[],[]
		for batch_id, data in enumerate(train_loader):
			loss,a_loss,kl_loss = model.train(data)
			losses.append(loss.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())
		return np.mean(losses),np.mean(a_losses),np.mean(kl_losses)

			
	elif model.stage == 'train_dynamics':
		E_losses,M_losses = [],[]
		for batch_id, data in enumerate(train_loader):
			E_loss,M_loss = model.train(data)
			E_losses.append(E_loss.item())
			M_losses.append(M_loss.item())
		return np.mean(E_losses),np.mean(M_losses)
 

	else: 
		assert False    


		 


def validate(model,test_loader):
	
	losses,like_losses,kl_losses = [],[],[]
	for batch_id, data in enumerate(test_loader):
		loss,like_loss,kl_loss = model.validate(data)
		losses.append(loss.item())
		like_losses.append(like_loss.item())
		kl_losses.append(kl_loss.item())

	return np.mean(losses),np.mean(like_losses),np.mean(kl_losses)

stage = 'train_dynamics' # 'train_skills' # 
assert stage in ['train_skills','train_dynamics']


env_name = 'kitchen-partial-v0'
dataset_file = 'datasets/'+env_name+'.npz'

batch_size = 100
h_dim = 256
z_dim = 256
lr = 5e-5
wd = 0.0
skill_beta = 0.1
model_beta = 1.0
H = 40
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'autoregressive'#, 'tanh_normal' or 'normal'
state_dependent_prior = True
encoder_type = 'state_action_sequence' #'state_sequence'
state_decoder_type = 'mlp' #'autoregressive'
init_state_dependent = True
fixed_sig = None
per_element_sigma = False

if dataset_file is None:
	env = gym.make(env_name)
	dataset = env.get_dataset()
else:
	if '.npz' in dataset_file:
		# load numpy file
		dataset = np.load(dataset_file)
	else:
		raise NotImplementedError

states = dataset['observations']
actions = dataset['actions']

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

states_train  = states[:N_train,:]
actions_train = actions[:N_train,:]


states_test  = states[N_train:,:]
actions_test = actions[N_train:,:]

													#  obs,next_obs,actions,H,stride
obs_chunks_train, action_chunks_train = chunks(states_train, actions_train, H, stride)

print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')

obs_chunks_test,  action_chunks_test  = chunks(states_test,  actions_test,  H, stride)

inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_data = TensorDataset(inputs_train)
test_data  = TensorDataset(inputs_test)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

# make path
if stage == 'train_skills':
	date =  date.today().strftime("%m_%d_%y")
	path = 'checkpoints/two_stage_'+date+'_'+env_name+'_H_'+str(H)+'_l2reg_'+str(wd)+'_skill_b_'+str(skill_beta)+'_model_b_'+str(model_beta)+'_per_el_sig_'+str(per_element_sigma)+'_a_dist_'+str(a_dist) # TODO
else:
	path = 'checkpoints/two_stage_05_20_22_kitchen-partial-v0_H_40_l2reg_0.0_skill_b_0.1_model_b_1.0_per_el_sig_False_a_dist_autoregressive'

assert path != None

if not os.path.isdir(path):
	os.mkdir(path)

print('a_dim: ', a_dim)
model = TwoStageSkillModel(state_dim,a_dim,z_dim,h_dim,a_dist,stage,path,skill_beta=skill_beta,model_beta=model_beta,fixed_sig=fixed_sig,per_element_sigma=per_element_sigma,lr=lr,wd=wd).cuda()

experiment = Experiment(api_key = '9mxH2vYX20hn9laEr0KtHLjAa', project_name = 'skill-learning')
experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'state_dependent_prior':state_dependent_prior,
							'z_dim':z_dim,
							'H':H,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'skill_beta':skill_beta,
							'model_beta':model_beta,
							'env_name':env_name,
							'path':path,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type,
							'per_element_sigma':per_element_sigma,
							'a_dist':a_dist})

# TODO get data

min_test_loss = 10**10
min_test_a_loss = 10**10
min_test_sT_loss = 10**10
for i in range(n_epochs):
	if stage == 'train_skills':
		# test
		test_loss,test_a_loss,test_kl_loss = validate(model,test_loader)
		# log test losses
		experiment.log_metric("test_loss", test_loss, step=i)
		experiment.log_metric("test_a_loss", test_a_loss, step=i)
		experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
		print('test_loss: ', test_loss)
		print('test_a_loss: ', test_a_loss)
		print('test_kl_loss: ', test_kl_loss)
		# decide whether to save a checkpoint
		if test_loss < min_test_loss:
			min_test_loss = test_loss
			# save 
			model.save_checkpoint('log_best.pth')

		# train
		loss,a_loss,kl_loss = train(model,train_loader)
		# log train losses
		experiment.log_metric("loss", loss, step=i)
		experiment.log_metric("a_loss", a_loss, step=i)
		experiment.log_metric("kl_loss", kl_loss, step=i)

		print('loss: ', loss)
		print('a_loss: ', a_loss)
		print('kl_loss: ', kl_loss)
		
		
		if i % 10 == 0:
		   model.save_checkpoint('log.pth')

		pass
	elif stage == 'train_dynamics':
		# test
		test_loss,test_s_T_loss,test_kl_loss = validate(model,test_loader)
		# log test losses
		experiment.log_metric("test_loss", test_loss, step=i)
		experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
		experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
		# decide whether to save a checkpoint
		if test_loss < min_test_loss:
			min_test_loss = test_loss
			# save 
			model.save_checkpoint('log_best.pth')
		# train
		E_loss,M_loss = train(model,train_loader)
		# log train losses
		experiment.log_metric("E_loss", E_loss, step=i)
		experiment.log_metric("M_loss", M_loss, step=i)

		if i % 10 == 0:
			model.save_checkpoint('log.pth')
	else:
		assert False

	





