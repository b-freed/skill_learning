import os
import time
import torch

class HyperParams:
    def __init__(self):
        self.batch_size = 32
        self.h_dim = 200
        self.z_dim = 256
        self.lr = 5e-4
        self.wd = 0.000
        self.state_dependent_prior = True
        self.term_state_dependent_prior = False
        self.state_dec_stop_grad = True
        self.gamma = 0.0 # TODO
        self.beta = 0.1
        self.alpha = 1.0
        self.ent_pen = 0.0

        self.max_temperature = 1.0 # gumbel-softmax temperature coeff
        self.min_temperature = 0.2
        self.temperature_anneal = 0 # Set to 0 to disable annealing
        self.temperature = self.max_temperature # initial temperature value

        self.max_sig = None
        self.fixed_sig = None
        self.H_min = 10
        self.H_max = 40
        self.min_skill_len = 10
        self.max_skill_len = 40
        self.max_skills_per_seq = 65 # I want avg skill length to be around 15
        self.stride = 1
        self.n_epochs = 50000
        self.test_split = .2
        self.a_dist = 'normal' # 'tanh_normal' or 'normal'
        self.encoder_type = 'state_action_sequence' #'state_sequence'
        self.state_decoder_type = 'mlp'
        self.env_name = 'antmaze-large-diverse-v0'
        self.device_id = 0
        self.device = f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
        self.exp_name = exp_name = f"T_{self.H_min}_{self.H_max}_slp_{self.gamma}"
        self.data_dir = 'datasets'
        self.log_online = True
        self.log_offline = True
        
        if self.term_state_dependent_prior:
            self.msg = f'{self.env_name}_tsdp_H{self.H_max}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_log'
        else:
            self.msg = f'{self.env_name}_enc_type_{self.encoder_type}_state_dec_{self.state_decoder_type}_H_{self.H_max}_l2reg_{self.wd}_a_{self.alpha}_b_{self.beta}_sg_{self.state_dec_stop_grad}_max_sig_{self.max_sig}_fixed_sig_{self.fixed_sig}_ent_pen_{self.ent_pen}_log'

        self.additional_msg = ''

        date_time = time.asctime()[4:16].replace(' ', '_').replace(':', '_')[:6].replace('__', '_')
        self.log_dir = os.path.join('checkpoints', date_time)

        run_num = 0
        self.exp_name = f'{exp_name}__r{run_num}'
        while os.path.exists(os.path.join(self.log_dir, self.exp_name)):
            run_num += 1
            self.exp_name = f'{exp_name}__r{run_num}'
        self.log_dir = os.path.join(self.log_dir, self.exp_name)

        os.makedirs(self.log_dir, exist_ok=True)
        

