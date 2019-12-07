import os
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
from environment import stacking_env
from Utils import process_pulse_log,get_optim

np.random.seed(2020)
def get_params():
	param={}
	param['main'] ={'root_dir':'log_optim/','loc':0.002,'feedback_iter':30}
	param['env'] ={'stage':2,'noise_sigma':0.0002,'init_state':'non_optimal',
	               'action_scale':0.5} # init_state = [ 'optimal','random','non_optimal']
	param['optim'] = {'optim':'sgd','lr':1, #1e-5
	               'momentum':0.5,'nesterov':False,
	               'betas':(0.7, 0.7), 'amsgrad':False}

	name=''
	if param['optim']['optim']=='sgd' or  param['optim']['optim']=='sgd_cps':
		if param['optim']['nesterov']:
			name += 'nesterov'
		else:
			name += 'sgd'
		name += '_lr'+str(param['optim']['lr'])
		name += '_mom'+str(param['optim']['momentum'])
	if param['optim']['optim']=='adam' or  param['optim']['optim']=='adam_cps':
		if param['optim']['amsgrad']:
			name += 'amsgrad'
		else:
			name += 'adam'
		name += '_lr'+str(param['optim']['lr'])
		name += '_bet'+str(param['optim']['betas'])
	param['optim']['name']=name
	if not os.path.exists(param['main']['root_dir']):
		os.mkdir(param['main']['root_dir'])
	param['main']['dir'] = param['main']['root_dir'] + 'stage{}_noise{}/'.format(str(param['env']['stage']), str(param['env']['noise_sigma']))
	if not os.path.exists(param['main']['dir']):
		os.mkdir(param['main']['dir'])
	param['main']['dir'] = param['main']['dir'] + param['optim']['optim'] +'/'
	if not os.path.exists(param['main']['dir']):
		os.mkdir(param['main']['dir'])
	with open(param['main']['dir']+'param.json','w') as f:
		json.dump(param,f)
	print(param)
	return param


def run_optim(param):
	env = stacking_env.simple_stacking_env(stage=param['env']['stage'], noise_sigma=param['env']['noise_sigma'],init_state=param['env']['init_state'],
	                          action_scale=param['env']['action_scale'],norm_action=False,obs_type='stack',obs_step=1,obs_feat=['power','action','l'])
	observation = env.reset()
	optim = get_optim(param['optim'])
	powers=[observation[0]]
	actions = []
	logs = []
	print("begin to run -> stage:{}, ideal power:{}".format(param['env']['stage'], env.ideal_power))
	print("epoch:{}, power:{}".format(0, powers[0]))
	for ii in range(1,param['main']['feedback_iter']+1):
		log_dict = env.cal_fake_grad(loc=param['main']['loc'],add_noise=True)
		act = optim.step(log_dict['grad'])
		env.update(act)

		power = log_dict['observation_1']['power'][-1]
		powers.append(power)
		actions.append(act)
		logs.append(log_dict)
		print("epoch:{}, power:{}".format(ii,power))

	powers = np.array(powers)
	actions = np.array(actions)
	joblib.dump({'powers':powers,'actions':actions,'logs':logs},param['main']['dir']+param['optim']['name']+'.pkl')
	ts = np.arange(0,param['main']['feedback_iter']+1)
	plt.figure()
	plt.plot(ts, powers)
	plt.title('Convergence')
	plt.xlabel('iteration')
	plt.ylabel('power')
	plt.savefig(param['main']['dir']+param['optim']['name']+'_converge.png')
	plt.show()



def main():
	param = get_params()
	run_optim(param)

if __name__=='__main__':
	main()

