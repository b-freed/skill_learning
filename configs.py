import os
import time
import torch

# class CfgBase2:
#     """Converts dict to objects: entries can be accessed as attributes"""
#     def __init__(self, d):
#         self.dict_to_attr(d)

#     def dict_to_attr(self, d):
#         for k, v in d.items():
#             if isinstance(v, dict):
#                 val = CfgBase2(v)
#             else:
#                 val = v
#             setattr(self, str(k), val)

# class CfgBase2:
#     """Converts dict to objects: entries can be accessed as attributes"""
#     def __init__(self, d):
#         self.dict_to_attr(d)

#     @classmethod
#     def dict_to_attr(cls, d):
#         for k, v in d.items():
#             if isinstance(v, dict):
#                 val = CfgBase2(v)
#             else:
#                 val = v
#             setattr(cls, str(k), val)


class CfgBase:
    """Base class for configuration files.
    
    Allows subscripting (accesing attributes as if they were dict entries). Allows conversion from nested objects to 
    dict. Prints contents in a pretty format. In future: Also allows loading from a dict/yaml file.

    """
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
        state_dim = 29
        a_dim = 8
        z_dim = 256
        h_dim = 200
    class decoder(CfgBase):
        state_dim = 29
        a_dim = 8
        z_dim = 256
        h_dim = 200
        state_dec_stop_grad = True
    class prior(CfgBase):
        state_dim = 29
        a_dim = 8
        z_dim = 256
        h_dim = 200

    # Optimizer configs
    batch_size = 128
    lr = 5e-4
    wd = 0.0
    n_epochs = 50000
    test_split = .2

    env_name = 'antmaze-medium-diverse-v0'

    # Loss coefficients / relevant
    a_loss_coeff = 1.0
    kl_loss_coeff = 0.1
    sT_loss_coeff = 1.0
    sT_ent_coeff = 0.0
    sl_loss_coeff = 1e-2
    grad_clip_threshold = 2.0

    # Misc - manually assigned
    data_dir = 'datasets'
    base_log_dir = 'logs'
    log_online = True
    log_offline = True
    use_tensorboard = True
    exp_identifier = f'' # TODO: enter each time.
    notes = f'' # TODO: enter each time.
    device_id = 0
    verbose = True

    # Automatically set
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    date_now = time.asctime()[4:10].replace(' ', '_')
    time_now = time.asctime()[11:19].replace(':', '_')
    exp_name = os.path.join(date_now, f'{exp_identifier}_{time_now}')
    log_dir = os.path.join(base_log_dir, exp_name)