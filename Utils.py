import joblib
import matplotlib.pyplot as plt
import numpy as np
from optimizer import sgd_cps, adam_cps


def process_pulse_log(logs, path, stage):
	cnt = len(logs['power'])
	ts = np.arange(cnt)

	power_stages = {}
	disp_stages = {}
	for ii in range(1, stage + 1):
		powers = []
		disp = []
		for t in ts:
			p = logs['power'][t][ii][0]
			d = logs['displacement'][t][ii]
			powers.append(p)
			disp.append(d)
		power_stages[ii] = np.array(powers)
		disp_stages[ii] = np.array(disp) - disp[0]
	joblib.dump({'ts': ts, 'power_stages': power_stages, 'disp_stages': disp_stages},
	            path + 'env_logs.pkl')

	for ii in range(1, stage + 1):
		power_log = power_stages[ii]
		offset_log = disp_stages[ii]
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
		ax1.plot(ts, power_log)
		ax1.set_title('free running EPP for' + str(ii) + 'nd stage stacking')
		ax1.set_xlabel('free running time (or epoch)', fontsize=20)
		ax1.set_ylabel('EPP (nj)', fontsize=20)

		ax2.plot(ts, offset_log)
		ax2.set_title('free running displacement for' + str(ii) + 'nd stage stacking')
		ax2.set_xlabel('free running time (or epoch)', fontsize=20)
		ax2.set_ylabel('PZM displacement (mm)', fontsize=20)
		plt.savefig(path + 'free_run_stage{}.png'.format(str(ii)))
		plt.close()


def get_optim(optim_param):
	if optim_param['optim'] == 'sgd' or optim_param['optim'] == 'sgd_cps':
		optim = sgd_cps.SGD_CPS(lr=optim_param['lr'], momentum=optim_param['momentum'], nesterov=optim_param['nesterov'])
	elif optim_param['optim'] == 'adam' or optim_param['optim'] == 'adam_cps':
		optim = adam_cps.Adam_CPS(lr=optim_param['lr'], betas=optim_param['betas'], amsgrad=optim_param['amsgrad'])
	else:
		raise NotImplementedError('only accept dsdg and dadam')
	return optim


def subplot(R, P, Q, S,save_name=''):
	joblib.dump([R, P, Q, S],save_name.replace('png','pkl'))

	r = list(zip(*R))
	p = list(zip(*P))
	q = list(zip(*Q))
	s = list(zip(*S))
	plt.figure()
	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

	ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
	ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
	ax[0, 1].plot(list(q[1]), list(q[0]), 'g')  # row=0, col=1
	ax[1, 1].plot(list(s[1]), list(s[0]), 'k')  # row=1, col=1
	ax[0, 0].title.set_text('Reward')
	ax[1, 0].title.set_text('Policy loss')
	ax[0, 1].title.set_text('Q loss')
	ax[1, 1].title.set_text('Max steps')
	plt.savefig(save_name)
	plt.show()

