
import numpy as np
import torch
from utils import reparameterize

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3


class FrankaRewardFn():
    def __init__(self):
        self.TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']
        # self.TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'light switch']
        # self.TASK_ELEMENTS = ['light switch']
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.REMOVE_TASKS_WHEN_COMPLETE = True
        self.TERMINATE_ON_TASK_COMPLETE = True
        self.done = False

    def step(self,obs):

        # obs_dict =  
        # next_q_obs = obs_dict['qp']
        # next_obj_obs = obs_dict['obj_qp']
        # next_goal = obs_dict['goal']

        #  np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])

        if self.done:
            return 0,[]
        
        next_q_obs = obs[:9]
        next_obj_obs = obs[9:9+21]
        # next_goal = obs[9+21:]

        idx_offset = len(next_q_obs) # 9
        completions = []
        for element in self.tasks_to_complete:
            # print('element: ', element)
            element_idx = OBS_ELEMENT_INDICES[element]
            
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                OBS_ELEMENT_GOALS[element])

            # distance = np.linalg.norm(
            #     next_obj_obs[..., element_idx - idx_offset] -
            #     next_goal[element_idx])
            # print('distance: ', distance)
            complete = distance < BONUS_THRESH
            if complete:
                
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        # print('completions: ', completions)
        # print('bonus: ', bonus)
        # reward_dict['bonus'] = bonus
        # reward_dict['r_total'] = bonus
        score = bonus

        reward = bonus

        self.done = not self.tasks_to_complete
        return reward,completions  #, done


def franka_plan_cost_fn(prev_states,skill_seq,model,use_eps=True):
    '''
    prev_states: batch_size x tau x state_dim sequence of states that have been gone thru so far, 
        from s0 up to & including current state
    skill_seq: batch_size x H x z_dim sequence of skills to be executed for remainder of the plan
    '''
    batch_size,H,_ = skill_seq.shape

    tau = prev_states.shape[0]


    reward_fns = [FrankaRewardFn() for _ in range(batch_size)]

    # step all reward fns up to current timestep
    for i in range(batch_size):
        for t in range(tau):
            obs = prev_states[t,:]
            _ = reward_fns[i].step(obs)

    # now all reward functions are up to date.  
    # Now, step each of them forward according to our predictions based on the skill_seq.
    
    cum_rew = batch_size * [0]
    # H = skill_seq.shape[1]
    # convert current state to torch tensor
    s = torch.stack(batch_size*[torch.tensor(prev_states[-1:,:],device=torch.device('cuda:0'),dtype=torch.float32)])
    for t in range(H):

        # print('=================================================')
        # get predicted next state
       # z_i = skill_seq[:,i:i+1,:] # might need to reshape
        if use_eps:
            mu_z, sigma_z = model.prior(s)
            z_t = mu_z + sigma_z*skill_seq[:,t:t+1,:]
        else:
            z_t = skill_seq[:,t:t+1,:]
        

        s_mean, s_sig = model.decoder.abstract_dynamics(s,z_t)
        # s = reparameterize(s_mean,s_sig)
        s = s_mean

        # convert to numpy 
        s_np = s.squeeze().detach().cpu().numpy() # idk if the squeeze is necessary?
        for i in range(batch_size):
            r_i,comp = reward_fns[i].step(s_np[i,:])
            cum_rew[i] += r_i
            
    return -torch.tensor(cum_rew,device=torch.device('cuda:0'),dtype=torch.float32)





if __name__ == '__main__':
    pass
    # create franka env
    # get dataset
    # reset the env to 
