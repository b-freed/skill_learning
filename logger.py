import torch
import numpy as np
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def errorband_plot(data, data_std, min_max=False, data_min=None, data_max=None, color='b', title=None, xlabel=None, ylabel=None, save_path=None):
    x = np.arange(len(data))
    plt.plot(x, data, f'{color}-')
    plt.fill_between(x, data - data_std, data + data_std, color=f'{color}', alpha=0.4)
    if min_max: plt.fill_between(x, data_min, data_max, color=f'{color}', alpha=0.2)
    # plt.legend(loc=loc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()

def save_plots(skill_lens, n_skills, base_path):
    skill_len_plot_args = {
        'data': skill_lens['mean'],
        'data_std': skill_lens['std'],
        'min_max': True,
        'data_min': skill_lens['min'],
        'data_max': skill_lens['max'],
        'color': 'b',
        'title': 'Skill Length',
        'xlabel': 'Training iteration',
        'ylabel': 'Skill length',
        'save_path': os.path.join(base_path, 'skill_length.png')
    }

    errorband_plot(**skill_len_plot_args)

    n_skills_plot_args = {
        'data': n_skills['mean'],
        'data_std': n_skills['std'],
        'min_max': True,
        'data_min': n_skills['min'],
        'data_max': n_skills['max'],
        'color': 'b',
        'title': 'Total executed skills per episode',
        'xlabel': 'Training iteration',
        'ylabel': 'total executed skills',
        'save_path': os.path.join(base_path, 'n_skills.png')
    }

    errorband_plot(**skill_len_plot_args)



class Logger:
    def __init__(self, config):
        self.args = config
        self.save_all_models = self.args['save_all_models']
        self.log_path = self.args['log_path']
        self.log_online = self.args['log_online']
        self.log_offline = self.args['log_offline']
        self.print_stats = self.args['print_stats']
        self.eval_episode = self.args['eval_episode']

        self.episode = 0
        self.step = 0
        self.best_return = -1e10
        exp_name = time.asctime()[4:16].replace(' ', '_').replace(':', '_')

        config_dict = omegaconf_to_dict(config)
        if self.print_stats: print_dict(config_dict)

        if self.log_online:
            self.exp = Experiment(self.args['api_key'], self.args['project_name'])
            self.exp.set_name(exp_name)
            if self.args['tags'] is not None: self.exp.add_tags(to_list(self.args['tags']))
            self.exp.log_parameters(config_dict)
        else:
            self.exp = None

        if self.log_offline:
            self.log_path = os.path.join(os.path.abspath(os.getcwd()), self.log_path, exp_name)
            os.makedirs(self.log_path, exist_ok=True)
            with open(os.path.join(self.log_path, 'config.yaml'), 'w') as outfile:
                OmegaConf.save(config=config, f=outfile.name)

        if self.log_online and not self.log_offline:
            print('Warning: Model logging not supported.')


    def _save(self, **kwargs):
        r'''(Variable inps) Update stats and save the log online and/or offline.'''

        for stat_label, stat in kwargs.items():
            if (self.exp is not None) and self.log_online: 
                self.exp.log_metric(stat_label, np.mean(stat))
            if self.log_offline:
                with open(os.path.join(self.log_path, stat_label + '.txt'), 'a') as f:
                    f.write(str(stat) + "\n")


    def update(self, rets, model):
        r"""Update stats and save the log online and/or offline.
    
        Args:
            pred (torch.Tensor): Network prediction
            target (torch.Tensor): Labels
            step_size (Float): log interval step size
            batch_idx (Float): Index of the current mini batch
            loss (Float): Loss at the current step
        """

        rets = to_list(to_list(rets))
        stats = {
            'returns': rets
        }

        self._save(**stats)

        for ret in rets:
            if ret > self.best_return:
                self.best_return = ret
                self.save_model(model, 'best_model')
        if self.save_all_models:
            self.save_model(model, f'{self.episode}')
    
        if self.print_stats:
            print(f'Eval @ episode {self.episode} | Best return: {self.best_return} | Current returns: {rets}')
        self.episode += self.eval_episode

    
    def save_model(self, model, file_name):

        file_path = os.path.join(self.log_path, file_name)

        if self.log_offline:
            if isinstance(model, np.ndarray):
                np.save(file_path, model)
            elif isinstance(model, torch.Tensor):
                torch.save(model, file_path)
            else: 
                raise NotImplementedError
            if self.log_online:
                self.exp.log_model(file_name, file_path)


    def show_plots(self):
        pass


    def save_plots(self):
        pass


    def end(self):
        if self.exp is not None: self.exp.end()
