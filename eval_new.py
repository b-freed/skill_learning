import os 
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

from comet_ml import Experiment
import gym
import torch
import numpy as np
import matplotlib
import importlib.util
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import make_gif,make_video
from cem import cem
from statsmodels.stats.proportion import proportion_confint
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)


to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def run_skills_iterative_replanning(env, model, hp, planning_args, ep_num):	
	device = planning_args.device
	state = s0 = env.reset()
	env.set_target()
	trajectory = [s0]
	frames = []

	to_torch = lambda x, d=device: torch.tensor(x, dtype=torch.float, device=d)
	goal = to_torch(np.array(env.target_goal)).reshape(1, 1, -1)
	reached_goal = lambda s, g=goal, t=planning_args.goal_threshold: \
		np.sum((to_numpy(s.squeeze())[:2] - to_numpy(g.squeeze())[:2])**2) <= t

	# for _ in range(args.):
	while not reached_goal(state):
		state_tensor = torch.cat(planning_args.n_samples * [to_torch(state).reshape((1, 1, -1))])
		cost_fn = lambda skill_samples: skill_model.get_expected_cost_for_cem(state_tensor, skill_samples, goal, \
			use_epsilons=planning_args.use_epsilon, length_cost=planning_args.plan_length_cost)

		skill_sample_mus = torch.zeros((planning_args.skill_seq_len, hp.z_dim),device=device)
		skill_sample_stds = planning_args.skill_std * torch.ones((planning_args.skill_seq_len, hp.z_dim), device=device)

		skill_sample_mus, skill_sample_stds = cem(skill_sample_mus, skill_sample_stds, cost_fn, planning_args.n_samples, \
			planning_args.keep_frac, planning_args.n_iters, l2_pen=planning_args.cem_l2_pen)

		skill = skill_sample_mus[0, :].unsqueeze(dim=0)

		if planning_args.use_epsilon:
			z_mu, z_std = model.prior(to_torch(state).reshape(1, -1))
			skill = z_mu + z_std * skill

		for j in range(planning_args.execution_len):
			if planning_args.render: env.render()
			if planning_args.save_video: frames.append(env.render(mode='rgb_array')) # FOV is terrible

			terminate = float(model.decoder.termination_decoder(to_torch(state).reshape(1, 1, -1), skill.unsqueeze(dim=0)).squeeze())
			action = model.decoder.ll_policy.numpy_policy(state, skill.unsqueeze(dim=0))
			state, reward, done, _ = env.step(action)
			trajectory.append(state)

			if reached_goal(state):
				if planning_args.debug: print('Reached goal! :)')
				break

			if (j > 3) and (terminate > 0.5): # TODO: this is for sigmoid
				break 
	
		if len(trajectory) > args.max_ep_len:
			if planning_args.debug: print('Timeout :(')
			break 

	env.close()
	trajectory = np.stack(trajectory)

	return trajectory, to_numpy(goal.squeeze()), bool(reached_goal(state)), frames


def load_config(load_path):	
	# load hps from given path
	spec = importlib.util.spec_from_file_location('configs', load_path)
	foo = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(foo)
	hp = foo.HyperParams()
	return hp


def plot_trajectory(trajectory, goal, planned_trajectory=None, bg_img_path=None, save_path=None):
	matplotlib.use('Agg') if save_path is not None else matplotlib.use('TkAgg')
	plt.close()
	plt.figure('Trajectory', figsize=(10, 10))

	# Load bg_img
	if bg_img_path is not None:
		bg_img = plt.imread(bg_img_path)
		plt.imshow(bg_img, extent = [-10, 30, -10, 30]) # TODO: Antmaze medium?

	plt.scatter(np.stack(trajectory)[0, 0], np.stack(trajectory)[0, 1], marker='o', label='Spawn location')
	plt.scatter(to_numpy(goal)[0], to_numpy(goal)[1], marker='x', label='Goal location')
	plt.plot(trajectory[:, 0], trajectory[:, 1], label='Executed trajectory', linewidth=1, c='g')
	if planned_trajectory is not None:
		plt.plot(planned_trajectory[:, 0], planned_trajectory[:, 1], marker='-', label='Planned trajectory')

	plt.legend()
	plt.axis('equal')

	if save_path is not None:
		plt.savefig(save_path, dpi=200)
	else:
		print('plotting')
		plt.show()


if __name__ == '__main__':
	import argparse
	from skill_model import SkillModelStateDependentPriorAutoTermination as SkillModel

	parser = argparse.ArgumentParser(description='Test skills')
	# Testing arguments
	# parser.add_argument('-l', '--log_dir', default='checkpoints/Jun_15/T_10_40_slp_1.0__r2/', help='log_dir')
	# parser.add_argument('-l', '--log_dir', default='checkpoints/Jun_8/T_10_40_slp_1.0__r6/', help='log_dir')
	parser.add_argument('-l', '--log_dir', default='checkpoints/Jun_10/T_10_40_slp_1.0__r2/', help='log_dir') # Best performer yet
	parser.add_argument('-r', '--render', default=False, help='render?')
	parser.add_argument('-s', '--save_video', default=False, help='save_video?')
	parser.add_argument('-d', '--save_dir', default='eval')
	# Planning arguments
	parser.add_argument('-e', '--use_epsilon', default=True, help='use epsilon')
	parser.add_argument('-v', '--variable_length', default=False, help='variable length planning')
	parser.add_argument('-p', '--max_ep', default=None, help='max epsilon')
	parser.add_argument('-m', '--max_replans', default=50, help='max replanning')
	parser.add_argument('-n', '--n_iters', default=20, help='number of iterations')
	parser.add_argument('-k', '--keep_frac', default=0.02, help='keep frequency')
	parser.add_argument('-c', '--cem_l2_pen', default=0.0, help='L2 penalty on CEM')
	parser.add_argument('--skill_std', default=0.3, help='std of skill space to be searched over')
	parser.add_argument('--max_ep_len', default=1000, help='Max time steps to plan')
	parser.add_argument('--device', default='cuda:0', help='Device to run eval on')
	parser.add_argument('--n_samples', default=4096, help='Number of CEM samples/candidates')
	parser.add_argument('--goal_threshold', default=1.0, help='Distance threshold from goal')
	parser.add_argument('--plan_length_cost', default=0.0, help='')
	parser.add_argument('--skill_seq_len', default=8, help='')
	parser.add_argument('--execution_len', default=40, help='')
	parser.add_argument('--n_evals', default=300, help='Number of times to evaluate')
	parser.add_argument('--bg_img_path', default='antmaze_medium.jpg')
	parser.add_argument('--debug', default=False)
	parser.add_argument('--msg', default='')

	args = parser.parse_args()

	assert not (args.save_video) # Only either is allowed at a time.

	config_path = os.path.join(args.log_dir, 'configs.py')
	hp = load_config(config_path)

	i = 0
	save_path = os.path.join(args.save_dir, f'{hp.exp_name}_eval_r{i}')
	while os.path.exists(save_path):
		save_path = os.path.join(args.save_dir, f'{hp.exp_name}_eval_r{i}')
		i += 1
	args.save_dir = save_path
	os.makedirs(args.save_dir, exist_ok=True)

	_device = args.device if torch.cuda.is_available() else 'cpu'
	hp.device = torch.device(_device)
	os.environ["_DEVICE"] = _device

	env = gym.make('antmaze-medium-play-v0') # gym.make(hp.env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	skill_model = SkillModel(state_dim, action_dim, hp.z_dim, hp.h_dim, a_dist=hp.a_dist, \
			state_dec_stop_grad=hp.state_dec_stop_grad, beta=hp.beta, alpha=hp.alpha,max_sig=hp.max_sig, \
			fixed_sig=hp.fixed_sig, ent_pen=hp.ent_pen, encoder_type=hp.encoder_type, \
			state_decoder_type=hp.state_decoder_type, max_skill_len=hp.max_skill_len, min_skill_len=hp.min_skill_len, \
			max_skills_per_seq=hp.max_skills_per_seq).to(hp.device)

	checkpoint = torch.load(os.path.join(args.log_dir, 'latest.pth'), map_location=hp.device)
	skill_model.load_state_dict(checkpoint['model_state_dict'])

	# Keep track of planned trajectories, executed trajectory, initial and goal locations. 
	eval_data = {
		'executed_trajectory': [],
		'init': [],
		'goal': [],
		'success_history': [],
		'total_steps': [],
		'planning_args': args,
		'mean_success': 0,
		'confidence_lower': 0,
		'confidence_upper': 0,
	}

	for i in tqdm(range(args.n_evals), desc=f'Eval: {args.log_dir}'):
		trajectory, goal, success, frames = run_skills_iterative_replanning(env, skill_model, hp, args, i)

		eval_data['success_history'].append(success)
		eval_data['total_steps'].append(len(trajectory))

		total_success = np.sum(eval_data['success_history'])
		total_runs = len(eval_data['success_history'])
		ci = proportion_confint(total_success, total_runs)

		eval_data['mean_success'] = total_success / total_runs
		eval_data['confidence_lower'], eval_data['confidence_upper'] = ci

		if args.render or args.save_video:
			eval_data['executed_trajectory'].append(trajectory)
			eval_data['init'].append(trajectory[0])
			eval_data['goal'].append(goal)
			# plot_path = None if args.render else os.path.join(args.save_dir, f'trajectory.png')
			plot_path = os.path.join(args.save_dir, f'trajectory.png')
			plot_trajectory(trajectory, goal, bg_img_path=args.bg_img_path, save_path=plot_path)

		print(f'success: {eval_data["mean_success"]} | CI: [{eval_data["confidence_lower"]}, \
			{eval_data["confidence_upper"]}]')

		if args.save_video:
			make_video(frames, os.path.join(args.save_dir, 'demo'))
			make_gif(frames, os.path.join(args.save_dir, 'demo'))
		torch.save(eval_data, os.path.join(args.save_dir, 'eval_data.pth'))