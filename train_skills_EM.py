from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior, SkillModelTerminalStateDependentPrior
from utils import chunks
import config
import os
from datetime import date

def train(model,E_optimizer,M_optimizer):
  
  E_losses = []
  M_losses = []
  
  for batch_id, data in enumerate(train_loader):
    data = data.cuda()
    states = data[:,:,:model.state_dim]
    actions = data[:,:,model.state_dim:]   # rest are actions

    ########### E STEP ###########
    E_loss = model.get_E_loss(states,actions)
    model.zero_grad()
    E_loss.backward()
    E_optimizer.step()

    ########### M STEP ###########
    M_loss = model.get_M_loss(states,actions)
    model.zero_grad()
    M_loss.backward()
    M_optimizer.step()

    # log losses
    E_losses.append(E_loss.item())
    M_losses.append(M_loss.item())
    
  return np.mean(E_losses),np.mean(M_losses)

def test(model):
  
  losses = []
  s_T_losses = []
  a_losses = []
  kl_losses = []
  s_T_ents = []

  with torch.no_grad():
    for batch_id, data in enumerate(test_loader):
      data = data.cuda()
      states = data[:,:,:model.state_dim]  # first state_dim elements are the state
      actions = data[:,:,model.state_dim:]   # rest are actions

      loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent  = model.get_losses(states, actions)

      # log losses
      losses.append(loss_tot.item())
      s_T_losses.append(s_T_loss.item())
      a_losses.append(a_loss.item())
      kl_losses.append(kl_loss.item())
      s_T_ents.append(s_T_ent.item())


  return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(s_T_ents)


batch_size = 100
decay_iter = 300
h_dim = 256
z_dim = 256#16#64
lr = 5e-5
wd = 0.0
beta = 1.0
alpha = 1.0
H = 40
stride = 1
n_epochs = 50000
test_split = .05
a_dist = 'normal'#'autoregressive' or 'normal'
state_dependent_prior = True
encoder_type = 'state_action_sequence' #'state_sequence'
state_decoder_type = 'mlp' #'autoregressive'
init_state_dependent = True
load_from_checkpoint = False
per_element_sigma = True #False for Franka

#env_name = 'antmaze-large-diverse-v0'
env_name = 'antmaze-medium-diverse-v0'
#env_name = 'kitchen-partial-v0'
#env_name = 'kitchen-complete-v0'
#env_name = 'kitchen-mixed-v0'

dataset_file = None

if dataset_file is None:
  env = gym.make(env_name)
  dataset = env.get_dataset()
else:
  if '.npz' in dataset_file:
    # load numpy file
    dataset = np.load(dataset_file)
  elif '.hdf5' in dataset_file:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env,h5py.File(dataset_file, "r"))

  else:
    print('Unrecognized data format!!!')
    assert False

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
obs_chunks_train, action_chunks_train = chunks(states_train, actions_train, H, stride,terminals)

print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')

obs_chunks_test,  action_chunks_test  = chunks(states_test,  actions_test,  H, stride,terminals)

# First, instantiate a skill model

if state_dependent_prior:
  model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()

else:
  raise NotImplementedError
  model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()

decay = True
lr_decay = 0.1
E_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr, weight_decay=wd)
M_optimizer = torch.optim.Adam(model.gen_model.parameters(), lr=lr, weight_decay=wd)
E_scheduler = torch.optim.lr_scheduler.StepLR(E_optimizer, 1, lr_decay)
M_scheduler = torch.optim.lr_scheduler.StepLR(M_optimizer, 1, lr_decay)

date =  date.today().strftime("%m_%d_%y")

filename = 'EM_model_'+date+'_'+env_name+'state_dec_'+str(state_decoder_type)+'_init_state_dep_'+str(init_state_dependent)+'_H_'+str(H)+'_zdim_'+str(z_dim)+'_l2reg_'+str(wd)+'_lr_'+str(lr)+'_a_'+str(alpha)+'_b_'+str(beta)+'_per_el_sig_'+str(per_element_sigma)+'_a_dist_'+str(a_dist)+'_log'+'_test_split_'+str(test_split)+'_decay_'+str(decay_iter)

if load_from_checkpoint:
  PATH = os.path.join(config.ckpt_dir,filename+'_best_sT.pth')
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  E_optimizer.load_state_dict(checkpoint['E_optimizer_state_dict'])
  M_optimizer.load_state_dict(checkpoint['M_optimizer_state_dict'])


inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_data = TensorDataset(inputs_train)
test_data  = TensorDataset(inputs_test)

train_loader = DataLoader(
  inputs_train,
  batch_size=batch_size,
  num_workers=0)

test_loader = DataLoader(
  inputs_test,
  batch_size=batch_size,
  num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):
  if (i+1)%decay_iter == 0 and decay:
    E_scheduler.step()
    M_scheduler.step()
  test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent = test(model)
  
  print("--------TEST---------")
  
  print('test_loss: ', test_loss)
  print('test_s_T_loss: ', test_s_T_loss)
  print('test_a_loss: ', test_a_loss)
  print('test_kl_loss: ', test_kl_loss)
  print('test_s_T_ent: ', test_s_T_ent)

  print(i)

  
  if test_s_T_loss < min_test_s_T_loss:
    min_test_s_T_loss = test_s_T_loss

    
    checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best_sT.pth')
    # checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
    torch.save({'model_state_dict': model.state_dict(),
        'E_optimizer_state_dict': E_optimizer.state_dict(),
              'M_optimizer_state_dict': M_optimizer.state_dict()}, checkpoint_path)

  if test_a_loss < min_test_a_loss:
    min_test_a_loss = test_a_loss

    checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best_a.pth')
    # checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
    torch.save({'model_state_dict': model.state_dict(),
        'E_optimizer_state_dict': E_optimizer.state_dict(),
              'M_optimizer_state_dict': M_optimizer.state_dict()}, checkpoint_path)

  E_loss,M_loss = train(model,E_optimizer,M_optimizer)
  
  print("--------TRAIN---------")
  
  print('E_loss: ', E_loss)
  print('M_loss: ', M_loss)
  print(i)

  if i % 10 == 0:
    
    checkpoint_path = os.path.join(config.ckpt_dir,filename+'.pth')
    # checkpoint_path = 'checkpoints/'+ filename + '.pth'
    torch.save({
              'model_state_dict': model.state_dict(),
              'E_optimizer_state_dict': E_optimizer.state_dict(),
              'M_optimizer_state_dict': M_optimizer.state_dict()
              }, checkpoint_path)
