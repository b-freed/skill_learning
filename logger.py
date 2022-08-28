from comet_ml import Experiment
import torch
import numpy as np
import os
import yaml
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import DataTracker

sns.set_theme(style="white")


def append_to_dict(d1, k, v):
    if not k in d1:
        d1[k] = []
    d1[k].append(float(v))
    return d1


def errorband_plot(data, data_std, min_max=False, data_min=None, data_max=None, color='b', title=None, xlabel=None, ylabel=None, label=None):
    x = np.arange(len(data))
    plt.plot(x, data, f'{color}-', label=label)
    plt.fill_between(x, data - data_std, data + data_std, color=f'{color}', alpha=0.4)
    if min_max: plt.fill_between(x, data_min, data_max, color=f'{color}', alpha=0.2)
    # plt.legend(loc=loc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')

def save_plots(skill_lens, skill_lens_test, n_skills, n_skills_test, base_path):
    skill_len_plot_args = {
        'data': np.array(skill_lens['mean']),
        'data_std': np.array(skill_lens['std']),
        'min_max': True,
        'data_min': np.array(skill_lens['min']),
        'data_max': np.array(skill_lens['max']),
        'color': 'b',
        'label': 'Train',
        'title': 'Skill Length',
        'xlabel': 'Training iteration',
        'ylabel': 'Skill length',
    }
    errorband_plot(**skill_len_plot_args)

    skill_len_plot_args = {
        'data': np.array(skill_lens_test['mean']),
        'data_std': np.array(skill_lens_test['std']),
        'min_max': True,
        'data_min': np.array(skill_lens_test['min']),
        'data_max': np.array(skill_lens_test['max']),
        'color': 'r',
        'label': 'Test',
        'title': 'Skill Length',
        'xlabel': 'Training iteration',
        'ylabel': 'Skill length',
    }
    errorband_plot(**skill_len_plot_args)
    
    save_path = os.path.join(base_path, 'skill_length.png')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()

    n_skills_plot_args = {
        'data': np.array(n_skills['mean']),
        'data_std': np.array(n_skills['std']),
        'min_max': True,
        'data_min': np.array(n_skills['min']),
        'data_max': np.array(n_skills['max']),
        'color': 'b',
        'label': 'Train',
        'title': 'Total executed skills per episode',
        'xlabel': 'Training iteration',
        'ylabel': 'total executed skills',
    }
    errorband_plot(**n_skills_plot_args)

    n_skills_plot_args = {
        'data': np.array(n_skills_test['mean']),
        'data_std': np.array(n_skills_test['std']),
        'min_max': True,
        'data_min': np.array(n_skills_test['min']),
        'data_max': np.array(n_skills_test['max']),
        'color': 'r',
        'label': 'Test',
        'title': 'Total executed skills per episode',
        'xlabel': 'Training iteration',
        'ylabel': 'total executed skills',
    }
    errorband_plot(**n_skills_plot_args)

    save_path = os.path.join(base_path, 'n_skills.png')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()



class Logger:
    def __init__(self, hp):
        self.hp = hp
        self.verbose = hp.verbose
        self.log_online = self.hp.log_online
        self.log_offline = self.hp.log_offline

        self.log_dir = hp.log_dir
        os.makedirs(self.log_dir, exist_ok=True) # safely create log dir

        self.experiment = Experiment(api_key='Wlh5wstMNYkxV0yWRxN7JXZRu', project_name='temp', display_summary_level=0)
        self.experiment.set_name(hp.exp_name)

        # Log config files
        self.experiment.log_parameter("params", str(hp.dict))
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.hp.dict, f)
        os.system(f'cp configs.py {self.log_dir}')

        self.min_test_loss = 10**10
        self.losses = DataTracker(verbose=False)

        if self.hp.use_tensorboard:
            self.tb_writer = torch.utils.tensorboard.writer.SummaryWriter(self.log_dir)

    def update(self, iteration_num, losses, mode='train'):
        if self.verbose: print(f' Iter: {iteration_num} | {self.hp.exp_name} - {self.hp.notes}')

        for _loss_name, loss_value in losses.items():
            loss_name = f'{_loss_name}_{mode}'

            if self.log_online:
                self.experiment.log_metric(loss_name, loss_value, step=iteration_num)
            if self.log_offline:
                self.losses.update(loss_name, loss_value)
                self.tb_writer.add_scalar(loss_name, loss_value, iteration_num)

            if self.verbose: print(f'{loss_name}: {loss_value}')


    def add_length_histogram(self, skill_len, iteration_num):
        """Adds the histogram (distribution) for the skill length to the tensorboard."""
        save_path = os.path.join(self.log_dir, f'skill_length_{iteration_num}.png')
        plt.figure()
        plt.hist(skill_len, bins=np.arange(0, max(skill_len) + 1, 1))
        plt.xlabel('Skill length')
        plt.ylabel('Frequency')
        plt.title(f'Skill length distribution @ iter: {iteration_num}')
        self.tb_writer.add_figure('Skill length dist', plt.gcf(), iteration_num)
        plt.savefig(save_path, dpi=300)
        plt.close()


    def save_training_state(self, iteration_num, model, model_optimizer, file_name):
        file_path = os.path.join(self.log_dir, file_name)

        if self.log_offline:
            torch.save({
                'model_state_dict': model.state_dict() if not isinstance(model, dict) else model, 
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'hp': self.hp.dict,
                'iteration': iteration_num,
                'min_test_loss': self.min_test_loss,
                'losses': self.losses.to_dict(mean=False),
                }, file_path)


    def show_plots(self):
        pass


    def save_plots(self):
        pass


    def end(self):
        if self.experiment is not None: self.experiment.end()
