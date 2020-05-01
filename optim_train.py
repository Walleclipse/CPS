import json
import os

import matplotlib.pyplot as plt
import numpy as np
from Utils import get_optim, plot_pulse
from environment import simple_stacking_env

np.random.seed(2020)
'''
'noise_sigma': 3e-4,'perturb_scale': 0.001,'init_nonoptimal': 0.1,
stage:3, lr: 5,
stage:5. lr: 1,
stage:8, lr: 0.01, 
'''


def get_params():
	param = {}
	param['main'] = {'root_dir': 'log_optim/', 'epoch': 50}
	param['env'] = {'stage': 5, 'noise_sigma': 3e-4, 'init_nonoptimal':  0.1,
	                'action_scale': 0.1}
	param['optim'] = {'optim': 'sgd_cps', 'perturb_scale': 0.001, 'lr':1,  # 1e-5
	                  'momentum': 0, 'nesterov': False, 'betas': (0.7, 0.7), 'amsgrad': False}

	name = ''
	if 'sgd' in param['optim']['optim']:
		if param['optim']['nesterov']:
			name += 'nesterov'
		else:
			name += 'sgd'
		name += '_lr' + str(param['optim']['lr'])
		name += '_mom' + str(param['optim']['momentum'])
	if 'adam' in param['optim']['optim']:
		if param['optim']['amsgrad']:
			name += 'amsgrad'
		else:
			name += 'adam'
		name += '_lr' + str(param['optim']['lr'])
		name += '_bet' + str(param['optim']['betas'])
	param['optim']['name'] = name
	if not os.path.exists(param['main']['root_dir']):
		os.mkdir(param['main']['root_dir'])
	param['main']['dir'] = param['main']['root_dir'] + 'stage{}_noise{}/'.format(str(param['env']['stage']),
	                                                                             str(param['env']['noise_sigma']))
	if not os.path.exists(param['main']['dir']):
		os.mkdir(param['main']['dir'])
	param['main']['dir'] = param['main']['dir'] + param['optim']['optim'] + '/'
	if not os.path.exists(param['main']['dir']):
		os.mkdir(param['main']['dir'])
	with open(param['main']['dir'] + 'param.json', 'w') as f:
		json.dump(param, f)
	print(param)
	return param


def run_optim(param):
	env = simple_stacking_env(stage=param['env']['stage'], noise_sigma=param['env']['noise_sigma'],
	                          init_nonoptimal=param['env']['init_nonoptimal'],
	                          action_scale=param['env']['action_scale'], normalize_action=False)
	observation = env.reset()
	optim = get_optim(param['optim'])
	powers = []
	actions = []
	logs = []
	print("begin to run -> stage:{}, max_f2_power:{}".format(param['env']['stage'], env.max_f2_power))
	print("epoch:{}, f2-power:{}".format(0, observation['power']))
	for ii in range(1, param['main']['epoch'] + 1):
		log_dict = env.cal_approx_grad(perturb_scale=param['optim']['perturb_scale'], add_noise=True)
		act = optim.step(log_dict['grad'])
		env.update(act)

		power = log_dict['observation_0']['power']
		powers.append(power)
		power = log_dict['observation_1']['power']

		powers.append(power)
		actions.append(act)
		logs.append(log_dict)
		print("epoch:{}, f2-power:{}".format(ii, power))
		if ii % 2 == 1:
			plot_pulse(log_dict['info_0']['stacks_pulse'],save_dir=param['main']['dir'],name=param['optim']['name'] + str(ii))

	powers = np.array(powers)
	# joblib.dump({'powers':powers,'actions':np.array(actions),'logs':logs},param['main']['dir']+param['optim']['name']+'.pkl',compress=6)
	ts = np.arange(len(powers))
	plt.figure()
	plt.plot(ts, powers)
	plt.title('Convergence')
	plt.xlabel('iteration')
	plt.ylabel('power')
	plt.savefig(param['main']['dir'] + param['optim']['name'] + '_converge.png')
	plt.show()


def main():
	param = get_params()
	run_optim(param)


if __name__ == '__main__':
	main()
