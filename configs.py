import os
import time
import torch


class CfgBase:
    def __getitem__(self, key):
        return getattr(self, key)
    def __repr__(self):
        return f'Configuration file: {str(self.get_dict())}'
    @classmethod
    def get_dict(cls):
        full_dict = {}
        cleaned_dict = {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
        for k, v in cleaned_dict.items():
            if isinstance(v, type):
                full_dict[k] = v.get_dict()
            else:
                full_dict[k] = v
        return full_dict
    @property
    def dict(self):
        return self.get_dict()


class Configs(CfgBase):
    # Model / training HP
    class encoder(CfgBase):
        a_dim = 2
        z_dim = 256
        h_dim = 200
    class decoder(CfgBase):
        a_dim = 2
        z_dim = 256
        h_dim = 200
    class prior(CfgBase):
        z_dim = 256
        h_dim = 200

    # Optimizer configs
    batch_size = 4096
    lr = 5e-4
    wd = 0.0
    n_epochs = 50000
    test_split = .2

    env_name = 'antmaze-medium-diverse-v0'

    # Loss coefficients / relevant
    gamma = 1.0 # TODO: tuning required.
    beta = 0.1
    alpha = 1.0
    ent_pen = 0.0
    grad_clip_threshold = 2.0

    state_dec_stop_grad = True

    # Misc - manually assigned
    data_dir = 'datasets'
    base_log_dir = 'logs'
    log_online = True
    log_offline = True
    exp_identifier = f'' # TODO: enter each time.
    notes = f'' # TODO: enter each time.
    device_id = 0 

    # Automatically set
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    date_now = time.asctime()[4:10].replace(' ', '_')
    time_now = time.asctime()[11:19].replace(':', '_')
    exp_name = os.path.join(date_now, f'{exp_identifier}_{time_now}')
    log_dir = os.path.join(base_log_dir, exp_name)