from comet_ml import Experiment
import torch
import numpy as np
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.log_online = self.hp.log_online
        self.log_offline = self.hp.log_offline

        self.log_dir = hp.log_dir

        self.experiment = Experiment(api_key = 'Wlh5wstMNYkxV0yWRxN7JXZRu', project_name = 'dump')
        self.experiment.set_name(hp.exp_name)

        self.experiment.log_parameters(hp.__dict__)
        # save hyperparams locally
        os.makedirs(self.log_dir, exist_ok=True) # safely create log dir
        os.system(f'cp configs.py {self.log_dir}')

        self.min_test_loss = 10**10

        self.test_sl_data = {}
        self.train_sl_data = {}
        self.test_ns_data = {}
        self.train_ns_data = {}

    def update_train(self, iteration_num, loss, s_T_loss, a_loss, kl_loss, s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max):

        print(f'Exp: {self.hp.exp_name} | Iter: {iteration_num}')
        print("--------TRAIN---------")
        print('loss: ', loss)
        print('s_T_loss: ', s_T_loss)
        print('a_loss: ', a_loss)
        print('kl_loss: ', kl_loss)
        print('s_T_ent: ', s_T_ent)
        print('sl_loss: ', sl_loss)
        print('sl_mean: ', sl_mean)
        print('ns_mean: ', ns_mean)
        print('')

        self.experiment.log_metric("loss", loss, step=iteration_num)
        self.experiment.log_metric("s_T_loss", s_T_loss, step=iteration_num)
        self.experiment.log_metric("a_loss", a_loss, step=iteration_num)
        self.experiment.log_metric("kl_loss", kl_loss, step=iteration_num)
        self.experiment.log_metric("s_T_ent", s_T_ent, step=iteration_num)
        self.experiment.log_metric("sl_loss", sl_loss, step=iteration_num)
        self.experiment.log_metric("sl_mean", sl_mean, step=iteration_num)
        self.experiment.log_metric("ns_mean", ns_mean, step=iteration_num)

        self.train_sl_data = append_to_dict(self.train_sl_data, 'mean', sl_mean)
        self.train_sl_data = append_to_dict(self.train_sl_data, 'std', sl_std)
        self.train_sl_data = append_to_dict(self.train_sl_data, 'min', sl_min)
        self.train_sl_data = append_to_dict(self.train_sl_data, 'max', sl_max)
    
        self.train_ns_data = append_to_dict(self.train_ns_data, 'mean', ns_mean)
        self.train_ns_data = append_to_dict(self.train_ns_data, 'std', ns_std)
        self.train_ns_data = append_to_dict(self.train_ns_data, 'min', ns_min)
        self.train_ns_data = append_to_dict(self.train_ns_data, 'max', ns_max)

        torch.save(self.train_sl_data, os.path.join(self.log_dir, 'train_sl_data.pt'))
        torch.save(self.train_ns_data, os.path.join(self.log_dir, 'train_ns_data.pt'))

    def update_test(self, iteration_num, test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_s_T_ent, sl_loss, sl_mean, sl_std, sl_min, sl_max, ns_mean, ns_std, ns_min, ns_max):

        print("--------TEST---------")
        print('loss: ', test_loss)
        print('s_T_loss: ', test_s_T_loss)
        print('a_loss: ', test_a_loss)
        print('kl_loss: ', test_kl_loss)
        print('s_T_ent: ', test_s_T_ent)
        print('sl_loss: ', sl_loss)
        print('sl_mean: ', sl_mean)
        print('ns_mean: ', ns_mean)
        print('')

        self.experiment.log_metric("test_loss", test_loss, step=iteration_num)
        self.experiment.log_metric("test_s_T_loss", test_s_T_loss, step=iteration_num)
        self.experiment.log_metric("test_a_loss", test_a_loss, step=iteration_num)
        self.experiment.log_metric("test_kl_loss", test_kl_loss, step=iteration_num)
        self.experiment.log_metric("test_s_T_ent", test_s_T_ent, step=iteration_num)
        self.experiment.log_metric("test_sl_loss", sl_loss, step=iteration_num)
        self.experiment.log_metric("test_sl_mean", sl_mean, step=iteration_num)
        self.experiment.log_metric("test_ns_mean", ns_mean, step=iteration_num)

        self.test_sl_data = append_to_dict(self.test_sl_data, 'mean', sl_mean)
        self.test_sl_data = append_to_dict(self.test_sl_data, 'std', sl_std)
        self.test_sl_data = append_to_dict(self.test_sl_data, 'min', sl_min)
        self.test_sl_data = append_to_dict(self.test_sl_data, 'max', sl_max)
    
        self.test_ns_data = append_to_dict(self.test_ns_data, 'mean', ns_mean)
        self.test_ns_data = append_to_dict(self.test_ns_data, 'std', ns_std)
        self.test_ns_data = append_to_dict(self.test_ns_data, 'min', ns_min)
        self.test_ns_data = append_to_dict(self.test_ns_data, 'max', ns_max)

        torch.save(self.test_sl_data, os.path.join(self.log_dir, 'test_sl_data.pt'))
        torch.save(self.test_ns_data, os.path.join(self.log_dir, 'test_ns_data.pt'))

    def save_training_state(self, iteration_num, model, model_optimizer, file_name):

        file_path = os.path.join(self.log_dir, file_name)

        if self.log_offline:
            torch.save({
                'model_state_dict': model.state_dict(), 
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'hp': self.hp.__dict__,
                'iteration': iteration_num,
                'min_test_loss': self.min_test_loss,
                }, file_path)


    def show_plots(self):
        pass


    def save_plots(self):
        pass


    def end(self):
        if self.exp is not None: self.exp.end()
