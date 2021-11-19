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

def validate(model,model_optimizer):

    losses = []
    s_T_losses = []
    a_losses = []
    kl_losses = []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        states = data[:,:,:model.state_dim]
        actions = data[:,:,model.state_dim:]

        with torch.no_grad():
            loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses(states, actions)
        
        # log losses
        losses.append(loss_tot.item())
        s_T_losses.append(s_T_loss.item())
        a_losses.append(a_loss.item())
        kl_losses.append(kl_loss.item())

    return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)

batch_size = 100

def get_data():

    env = PointmassEnv()
    obs = []
    goals = []
    actions = []
    for i in range(10000):
        start_loc = 2*np.random.uniform(size=2) - 1
        start_state = np.concatenate([start_loc,np.zeros(2)])
        goal_loc = 2*np.random.uniform(size=2) - 1
        state = env.reset(start_state)
        states = [state]
        action = []
        goal = []

        for t in range(100):
            u = env.get_stabilizing_control(goal_loc)
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
    experiment.add_tag('Validation set')

    states, actions, goals = get_data()
    state_dim = states.shape[2]
    a_dim = actions.shape[2]
    h_dim = 128
    N = states.shape[0]
    lr = 1e-4
    n_epochs = 100000

    model = SkillModel(state_dim, a_dim, 20, h_dim).cuda()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    experiment.log_parameters({'lr':lr,'h_dim':h_dim})

    inputs = np.concatenate([states, actions],axis=-1)
    targets = goals
    test_data = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets,dtype=torch.float32))

    test_loader = DataLoader(
		test_data,
		batch_size=batch_size,
		num_workers=0)

    for i in range(n_epochs):
        loss, s_T_loss, a_loss, kl_loss = validate(model,model_optimizer)
        print('loss: ', loss)
        print('s_T_loss: ', s_T_loss)
        print('a_loss: ', a_loss)
        print('kl_loss: ', kl_loss)
        print(i)
        experiment.log_metric("loss", loss, step=i)
        experiment.log_metric("s_T_loss", s_T_loss, step=i)
        experiment.log_metric("a_loss", a_loss, step=i)
        experiment.log_metric("kl_loss", kl_loss, step=i)

        if i % 10 == 0:
		filename = 'log_v.pth'
            	checkpoint_path = 'checkpoints/'+filename
            	torch.save({
							'model_state_dict': model.state_dict(),
							'model_optimizer_state_dict': model_optimizer.state_dict(),
							}, checkpoint_path)
