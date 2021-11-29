from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import gym
import d4rl
import matplotlib.pyplot as plt


device = torch.device('cuda:0')
H = 100
filename = 'maze2d_state_dep_prior_log.pth'
PATH = 'checkpoints/'+filename

state_dim = 4
a_dim = 2
h_dim = 128
z_dim = 4
skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim).cuda()
# Load trained skill model
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# get low-level policy
ll_policy = skill_model.decoder.ll_policy

# create env
env = 'maze2d-large-v1'
env = gym.make(env)

# simulate low-level policy in env
epochs = 100
states = []
for j in range(epochs):

    # sample a skill vector from prior
    z_prior_means = torch.zeros((1,1,z_dim),device=device)
    z_prior_sigs   = torch.ones((1,1,z_dim),device=device)
    z_sampled = skill_model.reparameterize(z_prior_means, z_prior_sigs)

    state = env.reset()
    epoch_states = []
    for i in range(H):

        #env.render()  # for visualization

        state = torch.reshape(torch.tensor(state,device=device,dtype=torch.float32),(1,1,-1))

        action_mean, action_sig = ll_policy(state,z_sampled)
        action_sampled = skill_model.reparameterize(action_mean, action_sig)

        # converting action_sampled to array
        action_sampled = action_sampled.cpu().detach().numpy()
        action_sampled = action_sampled.reshape([2,])

        state,_,_,_ = env.step(action_sampled)
        epoch_states.append(state)
    
    states.append(epoch_states)

states = np.stack(states)
#print(states)
print(np.shape(states))


#plt.figure()
plt.scatter(states[:,:,0],states[:,:,1])
plt.scatter(states[:,0,0],states[:,0,1])
plt.scatter(states[:,H-1,0],states[:,H-1,1])
plt.title('Skill Demonstration')
plt.legend(['Trajectory','Initial State','Terminal State'])
#plt.show()
plt.savefig('skill_demonstration_maze2d.png')
