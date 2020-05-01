import math
import numpy as np
import pynlo

from .PulseStacking import init_PZM_l0, StackStage, PhaseModulator, MultiStack, detect, copy_pulse, pulse_f2_power, \
	combine_trains, pulse_power
from .space import Box


class NormalizedObs(object):
	def __init__(self, obs_space):
		self.obs_space = obs_space
		self._action_space = {}
		for k, v in self.obs_space.items():
			self._action_space[k] = NormalizedAct(action_space=v)

	def _obs(self, obs):
		ret_obs = {}
		for k, v in obs.items():
			ret_obs[k] = self._action_space[k]._action(v)
		return ret_obs

	def _reverse_obs(self, obs):
		ret_obs = {}
		for k, v in obs.items():
			ret_obs[k] = self._action_space[k]._reverse_action(v)
		return ret_obs


class NormalizedAct(object):
	""" Wrap action """

	def __init__(self, action_space):
		self.action_space = action_space

	def _action(self, action):
		act_k = (self.action_space.high - self.action_space.low) / 2.
		act_b = (self.action_space.high + self.action_space.low) / 2.
		return act_k * action + act_b

	def _reverse_action(self, action):
		act_k_inv = 2. / (self.action_space.high - self.action_space.low)
		act_b = (self.action_space.high + self.action_space.low) / 2.
		return act_k_inv * (action - act_b)


class simple_stacking_env:
	def __init__(self, stage=7, noise_sigma=0.01, init_nonoptimal=0.1, action_scale=0.1, normalize_action=False):
		'''
		init_state = [ 'optimal','random','non_optimal']
		obs_feat =['power','action','PZM','pulse']
		'''
		self.stage = stage
		self.noise_sigma = noise_sigma
		self.init_nonoptimal = init_nonoptimal
		self.action_scale = action_scale
		self.normalize_action = normalize_action

		self.count = 0
		self._init_state()

	def _init_pulse(self):
		frep_MHz = 50000  # 5G
		period = 1e6 / frep_MHz  # =20 ps

		FWHM = 4.  # = 2ps,  pulse duration (ps)
		pulseWL = 1030  # pulse central wavelength (nm)
		EPP = 2e-10  # Energy per pulse (J) # 0.1 nj
		GDD = 0.0  # Group delay dispersion (ps^2)
		TOD = 0.0  # Third order dispersion (ps^3)

		# Window = 10.0  # simulation window (ps)
		Window = 2 ** (self.stage) * period  # simulation window (ps)
		Points = int(2 ** 3 * Window / FWHM)
		# Points = 2 ** 13  # simulation points

		# create the pulse!
		self.pulse = pynlo.light.DerivedPulses.SechPulse(power=1,  # Power will be scaled by set_epp
		                                                 T0_ps=FWHM / 1.76,
		                                                 center_wavelength_nm=pulseWL,
		                                                 time_window_ps=Window,
		                                                 GDD=GDD, TOD=TOD,
		                                                 NPTS=Points,
		                                                 frep_MHz=frep_MHz,
		                                                 power_is_avg=False)

		# set the pulse energy!
		self.pulse.set_epp(EPP)

		self.frep_MHz = frep_MHz
		self.pulseWL_mm = pulseWL * 1e-6


	def _init_stackstage(self):
		noise_type = 'gauss'
		noise_mean = 0
		stacks_list = []

		l0_list = init_PZM_l0(self.stage, self.frep_MHz)

		noise_sigma = self.noise_sigma * self.pulseWL_mm
		init_nois_scale = self.init_nonoptimal * self.pulseWL_mm

		for ii in range(1, self.stage + 1):
			fold = 2
			optim_l0 = l0_list[ii] / fold
			l0 = optim_l0
			if init_nois_scale > 0:
				l0 += ((-1) ** ii) * init_nois_scale
			elif init_nois_scale < 0:
				l0 += np.random.uniform(low=init_nois_scale, high=-init_nois_scale, size=1)[0]

			ss = StackStage(PZM_fold=fold, PZM_l0=l0, optim_l0=optim_l0, noise_type=noise_type, noise_mean=noise_mean,
			                noise_sigma=noise_sigma, name='s' + str(ii))
			stacks_list.append(ss)

		self.stacks_list = stacks_list


	def _init_state(self):
		self._init_pulse()
		self.pm = PhaseModulator(stage=self.stage)
		self._init_stackstage()
		self.pulse_stacking = MultiStack(stacks_list=self.stacks_list)

		self.orig_trains = self.pm.infer(self.pulse)

		ideal_pulse = copy_pulse(self.pulse)
		for _ in range(self.stage):
			new_aw = math.sqrt(1 / 2) * (ideal_pulse.AW + ideal_pulse.AW)
			ideal_pulse.set_AW(new_aw)
		self.ideal_pulse = ideal_pulse
		self.max_f2_power = round(pulse_f2_power(ideal_pulse), 2)
		self.max_avg_power = round(pulse_power(ideal_pulse), 2)

		action_scale = self.action_scale * self.pulseWL_mm

		self.action_space = Box(low=-action_scale, high=action_scale, shape=(self.stage,), dtype=np.float32)
		self.norm_action_fn = NormalizedAct(action_space=self.action_space)

		self.observation_space = {}
		# power
		obs_low = [0]
		obs_high = [math.ceil(self.max_f2_power)]
		self.observation_space['power'] = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
		# PZM
		obs_low = []
		obs_high = []
		for stk in self.stacks_list:
			obs_low.append(stk.l0 + self.action_space.low[0] * 1000)
			obs_high.append(stk.l0 + self.action_space.high[0] * 1000)
		self.observation_space['PZM'] = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
		# pulse
		dim = len(self.ideal_pulse.AT)
		obs_low = list([0] * dim)
		obs_high = list([self.max_avg_power] * dim)
		self.observation_space['pulse'] = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
		self.norm_observation_fn = NormalizedObs(obs_space=self.observation_space)

		self.peak_per_ind = int(len(self.ideal_pulse.AT) /  2 ** (self.stage + 1))

	def _extract_observation(self, trains_hist):
		stacks_power = []
		stacks_pulse = []
		stacks_PZM = []
		for ss in trains_hist:
			pulse_dict = combine_trains(ss)
			ret_pulse, f2_power = detect(pulse_dict)
			power = round(f2_power, 2)
			stacks_power.append(power)
			stacks_pulse.append(ret_pulse)
			stacks_PZM.append(ss[0]['displacement'])

		sample_pulse = stacks_pulse[-1]['I'][self.peak_per_ind:-self.peak_per_ind:self.peak_per_ind]
		info = {'final_f2power': stacks_power[-1], 'stacks_f2power': stacks_power, 'stacks_pulse': stacks_pulse,
		        'stacks_PZM': stacks_PZM,'sample_pulse':sample_pulse}
		observation = {'power': stacks_power[-1], 'pulse': stacks_pulse[-1], 'PZM': stacks_PZM,'sample_pulse':sample_pulse}
		return observation, info

	def _check_action(self, action):
		if action is None:
			action = self.random_action()
		else:
			if self.normalize_action:
				action = self.norm_action_fn._action(action)
			action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
		return action

	def prep_observation(self, observation, normalize=True, concat=False):
		if normalize:
			observation = self.norm_observation_fn._reverse_obs(observation)
		if concat:
			obs_array = None
			for k, v in observation.items():
				v = np.array([v]).flatten()
				if obs_array is None:
					obs_array = v
				else:
					obs_array = np.concatenate([obs_array, v])
			observation = obs_array
		return observation

	def reset(self):
		self.count = 0
		self._init_stackstage()
		self.pulse_stacking = MultiStack(stacks_list=self.stacks_list)
		self.action = np.zeros(self.action_space.shape)
		trains_hist = self.pulse_stacking.infer(self.orig_trains)
		observation, info = self._extract_observation(trains_hist)
		return observation

	def random_action(self):
		act = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.stage)
		return act

	def free_run(self):
		self.pulse_stacking.free_run()

	def measure(self, add_noise=True):
		if add_noise:
			self.free_run()
		trains_hist = self.pulse_stacking.infer(self.orig_trains)
		observation, info = self._extract_observation(trains_hist)
		return observation, info

	def update(self, action):
		action = self._check_action(action)
		self.pulse_stacking.feedback(action)

	def perturb(self, perturb_scale=0.001):
		loc = perturb_scale * self.pulseWL_mm
		action = np.random.uniform(low=-loc, high=loc, size=self.stage)
		self.pulse_stacking.feedback(action)
		return action

	def cal_approx_grad(self, perturb_scale=0.001, add_noise=True):
		observation_0, info_0 = self.measure(add_noise)
		y_0 = observation_0['power']
		delta_x = self.perturb(perturb_scale)
		observation_1, info_1 = self.measure(add_noise)
		y_1 = observation_1['power']

		grad = (y_1 - y_0) * delta_x

		ret_dict = {'observation_0': observation_0, 'info_0': info_0,
		            'observation_1': observation_1, 'info_1': info_1,
		            'y_0': y_0, 'y_1': y_1, 'delta_x': delta_x, 'grad': grad}
		return ret_dict

	def step(self, action):
		self.count += 1
		self.update(action)
		observation, info = self.measure(add_noise=True)
		reward = - (info['final_f2power'] - self.max_f2_power) ** 2 / self.max_f2_power
		done = False
		return observation, reward, done, info
