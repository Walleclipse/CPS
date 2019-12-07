
import math
import numpy as np
import pynlo
from .space import Box

from .PulseStacking import init_PZM_l0, StackStage,PhaseModulator, MultiStack, pulse_f2_power

class NormalizedEnv(object):
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
	def __init__(self, stage=7, noise_sigma=0.1,free_run_time=1,init_state='non_optimal',action_scale=0.1,norm_action=True,obs_type='final',
	             obs_step=10,obs_feat=['power','action','l','power_diff']):
		# init_state = [ 'optimal','random','non_optimal']
		# obs_type =['final_power','stacks_power','final_pulse','stacks_pulse','all']
		self.stage= stage
		self.noise_sigma = noise_sigma
		self.free_run_time = free_run_time
		self.init_state = init_state
		self.action_scale = action_scale
		self.norm_action = norm_action
		self.count=0
		self.obs_type = obs_type
		self.obs_step = obs_step
		self.obs_feat = obs_feat
		self._init_state()

	def _init_pulse(self):
		frep_MHz = 1000 # =1G
		period = 1e6 / frep_MHz  # =100 ps

		FWHM = 1.  # = 1ps,  pulse duration (ps)
		pulseWL = 1030  # pulse central wavelength (nm)
		EPP = 2e-10  # Energy per pulse (J) # 0.2 nj
		GDD = 0.0  # Group delay dispersion (ps^2)
		TOD = 0.0  # Third order dispersion (ps^3)

		#Window = 10.0  # simulation window (ps)
		Window = 2**(self.stage) * period  # simulation window (ps)
		Points = int(2**9 * Window /FWHM)
		#Points = 2 ** 13  # simulation points



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
		self.FWHM = FWHM
		self.pulseWL_mm = pulseWL * 1e-6

	def _init_stackstage(self):
		init_state = self.init_state
		noise_type = 'gauss'
		noise_mean = 0
		stacks_list = []

		l0_list = init_PZM_l0(self.stage,self.frep_MHz)

		noise_sigma = self.noise_sigma * self.pulseWL_mm
		init_nois_scale = 0.1* self.pulseWL_mm


		for ii in range(1,self.stage+1):
			fold = 2 * 2**((ii-2)//2) if ii >1 else 2
			optim_l0 = l0_list[ii]/fold
			l0 = optim_l0
			if init_state =='random':
				l0 += np.random.uniform(low=-init_nois_scale, high=init_nois_scale,size=1)
			elif init_state =='non_optimal':
				l0 += (-1)**ii * init_nois_scale
			ss = StackStage(PZM_fold=fold, PZM_l0=l0,optim_l0=optim_l0,noise_type=noise_type,noise_mean=noise_mean,
			                noise_sigma=noise_sigma, name='s'+str(ii))
			stacks_list.append(ss)

		self.stacks_list = stacks_list

	def _init_state(self):
		self._init_pulse()
		self.pm = PhaseModulator(stage=self.stage)
		self._init_stackstage()
		self.pulse_stacking = MultiStack(stacks_list=self.stacks_list, free_run_time=self.free_run_time)

		self.orig_trains = self.pm.infer(self.pulse)

		ideal_pulse=self.pulse.create_cloned_pulse()
		for _ in range(self.stage):
			new_at = math.sqrt(1 / 2) * (ideal_pulse.AT + ideal_pulse.AT)
			ideal_pulse.set_AT(new_at)
		self.ideal_pulse = ideal_pulse
		self.ideal_power = round(pulse_f2_power(ideal_pulse),2)

		action_scale = self.action_scale*self.pulseWL_mm

		self.action_space = Box(low=-action_scale, high=action_scale, shape=(self.stage,), dtype=np.float32)

		obs_dim = 0
		obs_low = []
		obs_high =[]
		if 'power' in self.obs_feat:
			if self.obs_type == 'final':
				obs_dim += 1
				obs_low +=[0]
				obs_high += [math.ceil(self.ideal_power)]
			else:
				obs_dim += self.stage
				obs_low +=[0]*self.stage
				obs_high += [math.ceil(self.ideal_power)]*self.stage
		if 'action' in self.obs_feat:
			obs_dim += self.stage
			obs_low += list(self.action_space.low)
			obs_high += list(self.action_space.high)
		if 'l' in self.obs_feat:
			obs_dim += self.stage
			for stk in self.stacks_list:
				obs_low.append(stk.l0 - self.action_space.high[0]*1000)
				obs_high.append(stk.l0 + self.action_space.high[0]*1000)
		if 'power_diff' in self.obs_feat:
			if self.obs_type == 'final':
				obs_dim += 1
				obs_low +=[-math.ceil(self.ideal_power)]
				obs_high += [math.ceil(self.ideal_power)]
			else:
				obs_dim += self.stage
				obs_low +=[-math.ceil(self.ideal_power)]*self.stage
				obs_high += [math.ceil(self.ideal_power)]*self.stage

		curr_observation_space = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
		self.norm_act_env = NormalizedEnv(action_space=self.action_space)
		self.norm_obs_env = NormalizedEnv(action_space=curr_observation_space)

		shape = (obs_dim * self.obs_step,)
		obs_low = obs_low* self.obs_step
		obs_high = obs_high * self.obs_step
		self.observation_space = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)

		self.action = np.zeros(shape=self.action_space.shape)
		self.hist_observation = shape[0]* [0]
		self.last_power = None

	def _extract_observation(self, info):
		stacks_power = []
		stacks_pulse = []
		stacks_l = []
		for ss in info[1:]:
			pulse = ss[0]['pulse']
			power = round(pulse_f2_power(pulse),2)
			stacks_power.append(power)
			stacks_pulse.append(pulse)
			stacks_l.append(ss[0]['displacement'])
		action = list(self.action)
		self.total_obs = {'power':stacks_power,'pulse':stacks_pulse,'action':action,'l':stacks_l}
		last_pow = self.last_power
		if last_pow is None:
			last_pow = stacks_power
		self.last_power = stacks_power

		cur_obs = []
		cur_obs_dict={}
		if 'power' in self.obs_feat:
			cur_obs += stacks_power[-1:] if self.obs_type == 'final' else stacks_power
			cur_obs_dict['power'] = stacks_power
		if 'action' in self.obs_feat:
			cur_obs += action
			cur_obs_dict['action'] = action
		if 'l' in self.obs_feat:
			cur_obs += stacks_l
			cur_obs_dict['l'] = stacks_l
		if 'power_diff' in self.obs_feat:
			diff_power = [a-b for a,b in zip(stacks_power,last_pow)]
			cur_obs += diff_power[-1:] if self.obs_type == 'final' else diff_power
			cur_obs_dict['power_diff'] = diff_power
		if self.norm_action:
			cur_obs = self.norm_obs_env._reverse_action(np.array(cur_obs))
		final_power = stacks_power[-1]

		self.hist_observation = self.hist_observation[len(cur_obs):] + list(cur_obs)
		observation = np.array(self.hist_observation)

		return observation, final_power, cur_obs,cur_obs_dict

	def _check_action(self,action):
		if action is None:
			action = self.random_action()
		else:
			if self.norm_action:
				action = self.norm_act_env._action(action)
			action = np.clip(action,self.action_space.low[0],self.action_space.high[0])
		return action

	def reset(self):
		self.count = 0
		self._init_stackstage()
		self.pulse_stacking = MultiStack(stacks_list=self.stacks_list, free_run_time=self.free_run_time)

		self.action = np.zeros(shape=self.action_space.shape)
		self.hist_observation = len(self.hist_observation) * [0]
		self.last_power = None
		info = self.pulse_stacking.infer(self.orig_trains)
		observation,_,cur_obs,cur_obs_dict = self._extract_observation(info)
		self.hist_observation = self.hist_observation[-len(cur_obs):] * (len(self.hist_observation)//len(cur_obs))
		observation = np.array(self.hist_observation)
		return observation

	def random_action(self):
		act = np.random.uniform(low=self.action_space.low, high=self.action_space.high,size=self.stage)
		return act

	def free_run(self):
		self.pulse_stacking.free_run()

	def measure(self,add_noise=True):
		if add_noise:
			self.free_run()
		info = self.pulse_stacking.infer(self.orig_trains)

		observation,final_power,cur_obs,cur_obs_dict = self._extract_observation(info)
		return observation, info,final_power,cur_obs_dict

	def update(self,action):
		action = self._check_action(action)
		self.action = action
		self.pulse_stacking.feedback(action)

	def perturb(self,loc=0.01):
		loc = loc * self.pulseWL_mm
		action = np.random.uniform(low=-loc, high=loc,size=self.stage)
		self.pulse_stacking.feedback(action)
		return action

	def cal_fake_grad(self,loc=0.001,add_noise=True):
		observation_0, info_0,final_power,cur_obs_dict_0 = self.measure(add_noise)
		y_0 = final_power
		delta_x = self.perturb(loc)
		observation_1, info_1,final_power,cur_obs_dict_1 = self.measure(add_noise)
		y_1 = final_power

		grad =(y_1 - y_0)*delta_x

		ret_dict={'observation_0':cur_obs_dict_0,
		          'observation_1':cur_obs_dict_1,
		          'y_0':y_0,'y_1':y_1,'delta_x':delta_x,'grad':grad}

		return ret_dict

	def step(self,action):
		self.count += 1
		self.update(action)

		observation, info,final_power,cur_obs_dict = self.measure(add_noise=True)
		reward = - (final_power - self.ideal_power)**2 / (self.ideal_power)
		if final_power >= 0.999*self.ideal_power:
			done=True
		else:
			done=False
		return observation, reward, done, info