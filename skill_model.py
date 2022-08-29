import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.kl as KL
import torch.distributions.normal as Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class ContinuousEncoder(nn.Module):
    def __init__(self, cfgs):
        super(ContinuousEncoder, self).__init__()
        self.state_dim = cfgs['state_dim']
        # self.a_dim  = cfgs['a_dim']
        self.h_dim  = cfgs['h_dim']
        self.z_dim  = cfgs['z_dim']
        self.encode = nn.Sequential(nn.Linear(self.state_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, self.h_dim), nn.ReLU())

        self.skill_length_means = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, 1))
        self.skill_length_stds = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, 1), nn.Softplus())

        self.skill_means = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, self.z_dim))
        self.skill_stds = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, self.z_dim), nn.Softplus())

    def forward(self, states):
        h = self.encode(states)

        z_len_means, z_len_stds = self.skill_length_means(h), self.skill_length_stds(h)
        z_means, z_stds = self.skill_means(h), self.skill_stds(h)

        # z_len_dist = self.get_skill_len_dist(z_len_means, z_len_stds)
        # z_dist = Normal.Normal(z_means, z_stds)

        # z_lens = z_len_dist.rsample()
        # z_t = z_dist.rsample()

        # z = self.get_executable_skills(z_t, z_lens) # Get skill to be actually executed at each timestep.

        return z_means, z_stds, z_len_means, z_len_stds


    def get_skill_len_dist(self, z_len_means, z_len_stds):
        base_dist = Normal.Normal(z_len_means, z_len_stds)
        transform = torch.distributions.transforms.TanhTransform()
        transformed_dist = TransformedDistribution(base_dist, [transform])
        return transformed_dist


    def get_executable_skills(self, z, z_lens):
        """Returns skills that are to be executed at each timestep.
        Args:
            z (torch.Tensor): batch_size x T x z_dim tensor of skills
            z_lens (torch.Tensor): batch_size x T tensor of skill lengths
        Returns:
            torch.Tensor: batch_size x T x z_dim tensor of skills to be executed at each timestep
        """
        mask, forward_lens = self.find_mask(z_lens)
        execution_skills, skill_lens = self.choose_skill(z, mask, forward_lens)
        return execution_skills, skill_lens


    def find_mask(self, z_lens):
        """Find events where accumulations cross integer boundaries. Skills will be updated at these timesteps.
        Args:
            z_lens (torch.Tensor): batch_size x T x 1 tensor of skill lens
        """
        batch_size, T, _ = z_lens.shape
        z_cum_lens = torch.cumsum(z_lens, dim=-2)
        compressed_len, execution_mask = torch.ones(batch_size, dtype=z_lens.dtype, device=z_lens.device), []
        forward_lens = []
        for t_idx in range(z_cum_lens.shape[1]):
            cum_len = z_cum_lens[:, t_idx, 0]
            update = (cum_len >= compressed_len) # Gradients won’t pass through (the time predictor) here. :(
            execution_mask.append(update)
            forward_len = torch.zeros(batch_size, dtype=z_lens.dtype, device=z_lens.device)
            forward_len[update] = cum_len[update] - compressed_len[update]
            compressed_len[update] += 1
            forward_lens.append(forward_len)
        return execution_mask, forward_lens


    def choose_skill(self, z, mask, forward_lens):
        """Repeats skills until mask events are encountered.
        Args:
            z (torch.Tensor):  T x z_dim tensor of skills.
            mask (torch.Tensor): T x 1 tensor of boolean masks.
            forward_lens (torch.Tensor): T x 1 tensor of forward skill lens at mask events.
        """
        batch_size, T, z_dim = z.shape
        z_choosen = []
        z_update = z[:, 0, :]
        skill_lens, _sl = [], torch.ones(batch_size, dtype=z.dtype, device=z.device)
        for t_idx, (mask_slice, forward_lens_slice) in enumerate(zip(mask, forward_lens)):
            # Update sl list with terminated skill lengths and reset the terminated skill counters.
            skill_lens.extend(_sl[mask_slice].tolist())
            _sl[mask_slice] = 1

            # Update the skills that were terminated.
            if t_idx == T-1:
                z_update[mask_slice] = z[mask_slice, t_idx, :]
            else:
                z_update[mask_slice] = (1 - forward_lens_slice[mask_slice].unsqueeze(dim=-1)) * z[mask_slice, t_idx, :] + \
                    forward_lens_slice[mask_slice].unsqueeze(dim=-1) * z[mask_slice, t_idx + 1, :]

            # Increment the skills that weren't terminated.
            _sl[torch.logical_not(mask_slice)] += 1

            z_choosen.append(z_update.clone())
        skill_lens.extend(_sl[torch.logical_not(mask_slice)].tolist())
        return torch.stack(z_choosen, dim=1), skill_lens


class ContinuousPrior(ContinuousEncoder):
    def __init__(self, *args, **kwargs):
        super(ContinuousPrior, self).__init__(*args, **kwargs)

    def forward(self, s0):
        h = self.encode(s0)

        z_len_means, z_len_stds = self.skill_length_means(h), self.skill_length_stds(h)
        z_means, z_stds = self.skill_means(h), self.skill_stds(h)

        # z_len_dist = self.get_skill_len_dist(z_len_means, z_len_stds)
        # z_dist = Normal.Normal(z_means, z_stds)

        # z_lens = z_len_dist.rsample()
        # z_t = z_dist.rsample()

        # z = self.get_executable_skills(z_t, z_lens) # Get skill to be actually executed at each timestep.

        return z_means, z_stds, z_len_means, z_len_stds


class LowLevelPolicy(nn.Module):
    """
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    """
    def __init__(self, state_dim=0, a_dim=0, z_dim=0, h_dim=0, max_sig=None, fixed_sig=None, **kwargs):

        super(LowLevelPolicy,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())

        self.mean_layer = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, a_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, a_dim))

        self.a_dim = a_dim
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig


    def forward(self, state, z):
        """
        Args:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        Returns:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        """
        # Concat state and z_tiled
        state_z = torch.cat([state, z],dim=-1)
        # pass z and state through layers
        feats = self.layers(state_z)
        # get mean and stand dev of action distribution
        a_mean = self.mean_layer(feats)
        if self.max_sig is None:
            a_sig  = nn.Softplus()(self.sig_layer(feats))
        else:
            a_sig = self.max_sig * nn.Sigmoid()(self.sig_layer(feats))

        if self.fixed_sig is not None:
            a_sig = self.fixed_sig*torch.ones_like(a_sig)

        return a_mean, a_sig
    
    def numpy_policy(self,state,z):
        """
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        """
        state = torch.reshape(torch.tensor(state,device=os.environ.get('_DEVICE', 'cpu'),dtype=torch.float32),(1,1,-1))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)

        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps




class AbstractDynamics(nn.Module):
    """
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    """
    def __init__(self, state_dim=0, z_dim=0, h_dim=0, **kwargs):

        super(AbstractDynamics,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,state_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,state_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())

    def forward(self,s0,z):

        """
        Args:
            s0: batch_size x (1 or k) x state_dim initial state (first state in execution of skill)
            z:  batch_size x (1 or k) x z_dim "skill"/z
        Returns: 
            sT_mean: batch_size x (1 or k) x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x (1 or k) x state_dim tensor of terminal (time=T) state standard devs
        """

        # concatenate s0 and z
        if s0.shape[-2] != 1:
            z = z.tile([1, s0.shape[-2], 1])

        s0_z = torch.cat([s0, z],dim=-1)

        # pass s0_z through layers
        feats = self.layers(s0_z)

        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)

        return sT_mean,sT_sig


class Decoder(nn.Module):
    """
    Decoder module.
    Decoder takes states, actions, and a sampled z and Returns parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    """
    def __init__(self, cfgs):
        super(Decoder,self).__init__()
        
        self.state_dim = cfgs['state_dim']
        self.a_dim = cfgs['a_dim']
        self.h_dim = cfgs['h_dim']
        self.z_dim = cfgs['z_dim']

        self.state_dec_stop_grad = cfgs['state_dec_stop_grad']

        self.abstract_dynamics = AbstractDynamics(**cfgs)
        self.ll_policy = LowLevelPolicy(**cfgs)

        
    def forward(self, s, z, s0):
        """
        Args: 
            s: batch_size x T x state_dim state sequence tensor
            z: batch_size x 1 x z_dim sampled z/skill variable

        Returns:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        """
        a_mean,a_sig = self.ll_policy(s,z)

        # z_decoder = z.detach() if self.state_dec_stop_grad else z
        
        # sT_mean, sT_sig = self.abstract_dynamics(s0, z_decoder)       

        return 0, 0, a_mean, a_sig


class SkillModel(nn.Module):
    """Skeleton for skill model(s). Defines shared planning and other utility code."""
    def __init__(self, configs):
        super(SkillModel, self).__init__()
        self.configs = configs


    def get_losses(self,states,actions):
        """
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)]
          + E_q [sum_t=0^T P(a_t|s_t,z)]
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
        Distributions we need:
        """

        s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)

        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        if self.decoder.ll_policy.a_dist == 'normal':
            a_dist = Normal.Normal(a_means, a_sigs)
        elif self.decoder.ll_policy.a_dist == 'tanh_normal':
            base_dist = Normal.Normal(a_means, a_sigs)
            transform = torch.distributions.transforms.TanhTransform()
            a_dist = TransformedDistribution(base_dist, [transform])
        else:
            assert False
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        # z_prior_means = torch.zeros_like(z_post_means)
        # z_prior_sigs = torch.ones_like(z_post_sigs)
        z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:])
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)

        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        T = states.shape[1]
        s_T = states[:,-1:,:]
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),   dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1))
        s_T_ent  = torch.mean(torch.sum(s_T_dist.entropy(),       dim=-1))/T

        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss + self.ent_pen*s_T_ent

        return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent


    def get_expected_cost(self, s0, skill_seq, goal_states):
        """
        s0 is initial state  # batch_size x 1 x s_dim
        skill sequence is a 1 x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        """
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_states = torch.cat(batch_size * [goal_states],dim=0)
        s_i = s0

        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        for i in range(skill_seq_len):
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            mu_z, sigma_z = self.prior(s_i)

            z_i = mu_z + sigma_z*torch.cat(batch_size*[skill_seq[:,i:i+1,:]],dim=0)
            # converting z_i from 1x1xz_dim to batch_size x 1 x z_dim
            # z_i = torch.cat(batch_size*[z_i],dim=0) # feel free to change this to tile
            # use abstract dynamics model to predict mean and variance of state after executing z_i, conditioned on s_i
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)

            # sample s_i+1 using reparameterize
            s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            pred_states.append(s_i)

        #compute cost for sequence of states/skills
        # print('predicted final loc: ', s_i[:,:,:2])
        s_term = s_i
        cost = torch.mean((s_term[:,:,:2] - goal_states[:,:,:2])**2)


        return cost, torch.cat(pred_states,dim=1)


    def get_expected_cost_for_cem(self, s0, skill_seq, goal_state, use_epsilons=True, plot=True, length_cost=0):
        """
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        """
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0

        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = [torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()]
        # costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        for i in range(skill_seq_len):

            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i)

                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]

            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)

            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze() + (i+1)*length_cost
            costs.append(cost_i)

            pred_states.append(s_i)

        costs = torch.stack(costs,dim=1)  # should be a batch_size x T or batch_size x T
        costs,_ = torch.min(costs,dim=1)  # should be of size batch_size

        return costs


    def get_expected_cost_variable_length(self, s0, skill_seq, lengths, goal_state, use_epsilons=True, plot=False):
        """
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        """
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0

        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        for i in range(skill_seq_len):

            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i)

                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]

            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)

            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = (lengths == i+1)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
            costs += cost_i

            pred_states.append(s_i)

        return costs


    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).to(os.environ.get('_DEVICE', 'cpu')), torch.ones(mean.size()).to(os.environ.get('_DEVICE', 'cpu')))
        return mean + std*eps


class ContinuousSkillModel(SkillModel):
    def __init__(self, configs):
        super(ContinuousSkillModel, self).__init__(configs)

        self.encoder = ContinuousEncoder(configs['encoder'])
        self.decoder = Decoder(configs['decoder'])

        self.prior = ContinuousPrior(configs['encoder'])

        self.grad_clip_threshold = configs['grad_clip_threshold']


    def forward(self, states, actions):
        """Takes states and actions, returns the distributions necessary for computing the objective function

        Args:
            states:       batch_size x t x state_dim sequence tensor
            actions:      batch_size x t x a_dim action sequence tensor

        Returns:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution
        """
        # Encode states and actions to get posterior over z
        z, z_lens = self.encoder(states, actions)

        # Pass z_sampled and states through decoder
        s_T_mean, s_T_sig, a_means, a_sigs = self.decoder(states, z)

        return s_T_mean, s_T_sig, a_means, a_sigs, z, z_lens


    def compute_losses(self, s_T_mean, s_T_sig, a_means, a_sigs,
                        z_post_means, z_post_sigs, s_0, time_steps, s_T_gt, a_gt, mask):

        # compute skill prior
        z_prior_means, z_prior_sigs = self.prior(s_0)

        # Construct required distributions
        s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)

        if self.decoder.ll_policy.a_dist == 'normal':
            a_dist = Normal.Normal(a_means, a_sigs)
        elif self.decoder.ll_policy.a_dist == 'tanh_normal':
            base_dist = Normal.Normal(a_means, a_sigs)
            transform = torch.distributions.transforms.TanhTransform()
            a_dist = TransformedDistribution(base_dist, [transform])
        else:
            assert False, f'{self.decoder.ll_policy.a_dist} not supported'

        # Loss for predicting terminal state
        s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T_gt), dim=-1)/time_steps) # divide by T because all other losses we take mean over T dimension, effectively dividing by T
        # Los for predicting actions
        a_loss_raw = a_dist.log_prob(a_gt)
        a_loss_raw[mask, :] = 0.0
        a_loss = -torch.mean(torch.sum(a_loss_raw, dim=-1))
        # Entropy corresponding to terminal state
        s_T_ent = torch.mean(torch.sum(s_T_dist.entropy(), dim=-1)/time_steps)
        # Compute KL Divergence between prior and posterior
        kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1)/time_steps) # divide by T because all other losses we take mean over T dimension, effectively dividing by T

        loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss + self.ent_pen*s_T_ent

        return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent
    
    def clip_gradients(self):
        """Clip gradients to avoid explosions"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_threshold)
