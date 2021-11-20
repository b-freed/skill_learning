from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
from pointmass_env import PointmassEnv



def train(model,model_optimizer):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []

	for batch_id, (data, target) in enumerate(train_loader):
		data, target = data.cuda(), target.cuda()
		states = data[:,:,:model.state_dim]  # first state_dim elements are the state
		actions = data[:,:,model.state_dim:]	 # rest are actions

		loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses(states, actions)

		model_optimizer.zero_grad()
		loss_tot.backward()
		model_optimizer.step()

		# log losses

		losses.append(loss_tot.item())
		s_T_losses.append(s_T_loss.item())
		a_losses.append(a_loss.item())
		kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)

def validate(model):

    val_losses = []
    val_s_T_losses = []
    val_a_losses = []
    val_kl_losses = []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        states = data[:,:,:model.state_dim]
        actions = data[:,:,model.state_dim:]

        with torch.no_grad():
            val_loss_tot, val_s_T_loss, val_a_loss, val_kl_loss = model.get_losses(states, actions)
        
        # log losses
        val_losses.append(val_loss_tot.item())
        val_s_T_losses.append(val_s_T_loss.item())
        val_a_losses.append(val_a_loss.item())
        val_kl_losses.append(val_kl_loss.item())

    return np.mean(val_losses), np.mean(val_s_T_losses), np.mean(val_a_losses), np.mean(val_kl_losses) 

batch_size = 100

def get_data(datasize):

	env = PointmassEnv()
	obs = []
	goals = []
	actions = []
	for i in range(datasize):
		start_loc = 2*np.random.uniform(size=2) - 1
		start_state = np.concatenate([start_loc,np.zeros(2)])
		goal_loc = 2*np.random.uniform(size=2) - 1
		state = env.reset(start_state)
		states = [state]
		action = []
		goal = []

		for t in range(100):
			#print('env.x: ', env.x)
			u = env.get_stabilizing_control(goal_loc)
			#print('u: ', u)
			state = env.step(u)
			if t != 99:
				states.append(state)
			action.append(u)
			goal.append(goal_loc)

		obs.append(states)
		actions.append(action)
		goals.append(goal)
	
	obs = np.stack(obs)
	actions = np.stack(actions)
	goals = np.stack(goals)

	return obs, actions, goals

if __name__ == '__main__':
	experiment = Experiment(api_key = 'yQQo8E8TOCWYiVSruS7nxHaB5', project_name = 'skill-learning', workspace="anirudh-27")
	experiment.add_tag('2 mean/sigma layers')

	'''FOR TRAINING'''
	
	states, actions, goals = get_data(datasize=10000)
	state_dim = states.shape[2]
	a_dim = actions.shape[2]
	h_dim = 128
	z_dim = 4
	N = states.shape[0]
	lr = 1e-4
	n_epochs = 10000

	# First, instantiate a skill model
	model = SkillModel(state_dim, a_dim, z_dim, h_dim).cuda()
	model_optimizer = torch.optim.Adam(model.parameters(), lr=lr) # default lr-0.0001

	experiment.log_parameters({'lr':lr,
							   'h_dim':h_dim, 'z_dim':z_dim})

	inputs = np.concatenate([states, actions],axis=-1) # array that is dataset_size x T x state_dim+action_dim 
	targets = goals
	train_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets,dtype=torch.float32))

	train_loader = DataLoader(
		train_data,
		batch_size=batch_size,
		num_workers=0)  

	'''FOR VALIDATION'''

	states_val, actions_val, goals_val = get_data(datasize=500)
	inputs_val = np.concatenate([states_val, actions_val],axis=-1)
	targets_val = goals_val
	test_data = TensorDataset(torch.tensor(inputs_val, dtype=torch.float32), torch.tensor(targets_val,dtype=torch.float32))

	test_loader = DataLoader(
		test_data,
		batch_size=batch_size,
		num_workers=0) 

	min_test = 10**10
	for i in range(n_epochs):

		'''TRAINING'''
		print('TRAINING-------------------------------------------------')
		
		loss, s_T_loss, a_loss, kl_loss = train(model,model_optimizer)

		print('loss: ', loss)
		print('s_T_loss: ', s_T_loss)
		print('a_loss: ', a_loss)
		print('kl_loss: ', kl_loss)
		print('EPOCH:', i)

		experiment.log_metric("loss", loss, step=i)
		experiment.log_metric("s_T_loss", s_T_loss, step=i)
		experiment.log_metric("a_loss", a_loss, step=i)
		experiment.log_metric("kl_loss", kl_loss, step=i)

		'''VALIDATION'''
		print('VALIDATION------------------------------------------------')
		
		val_loss, val_s_T_loss, val_a_loss, val_kl_loss = validate(model)

		print('val_loss: ', val_loss)
		print('val_s_T_loss: ', val_s_T_loss)
		print('val_a_loss: ', val_a_loss)
		print('val_kl_loss: ', val_kl_loss)
		print('EPOCH:', i)

		experiment.log_metric("val_loss", val_loss, step=i)
		experiment.log_metric("val_s_T_loss", val_s_T_loss, step=i)
		experiment.log_metric("val_a_loss", val_a_loss, step=i)
		experiment.log_metric("val_kl_loss", val_kl_loss, step=i)


		if i % 10 == 0:
			filename = 'log_z_dim4_2layers.pth'
			checkpoint_path = 'checkpoints/'
			torch.save({'model_state_dict': model.state_dict(),
						'model_optimizer_state_dict': model_optimizer.state_dict(),
					}, checkpoint_path+filename)
		
		if val_loss < min_test:
			min_test = val_loss
			torch.save({'model_state_dict': model.state_dict(),
						'model_optimizer_state_dict': model_optimizer.state_dict(),
						}, checkpoint_path+'log_best.pth')
