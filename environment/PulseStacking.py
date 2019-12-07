
from scipy import constants
import numpy as np
from functools import partial
import math

polarization_mark={'p':'^ (up)','-p':'v (down)','s':'-> (in)','-s':'<- (out)'}
_C_mmps = constants.value('speed of light in vacuum') * 1e3 / 1e12 # light speed of  mm/ps


class PhotoDetector:
	def intensity(self,num):
		return abs(num) ** 2
	def dB(self,num):
		return 10 * np.log10(num)
	def f2_amp(self,num):
		return num * num.conjugate()
	def max_num(self,num):
		return np.max(np.abs(num))
	def integrate_num(self,num):
		return np.trapz(num)
	def avg_power(self,num):
		return self.integrate_num(self.intensity(num))
	def avg_f2_power(self,num):
		return self.integrate_num(self.intensity(self.f2_amp(num)))

avg_f2_power = PhotoDetector().avg_f2_power

def pulse_f2_power(pulse):
	return 1e9 * pulse.dT_mks * avg_f2_power(pulse.AT)

def init_PZM_l0(stage=3,frep_MHZ=1000):
	period = 1e6/frep_MHZ # ps
	l0_list=[0]
	for ii in range(1,stage+1):
		delay = 2**(ii-1)*period/2
		l = delay * _C_mmps
		l0_list.append(l)
	return l0_list

class PhaseModulator:
	def __init__(self,stage=3, p_s_delay=1/2):
		self.stage=stage
		self.ps_delay=p_s_delay
		self.num_pulse = 2**stage

	def _phase_preset(self,out_p):
		if out_p == 'p':
			return 'p', 's'
		if out_p == '-p':
			return '-p', '-s'
		if out_p == 's':
			return '-p', 's'
		if out_p == '-s':
			return 'p', '-s'

	def _cal_polar(self,stage=None):
		if stage is None:
			stage = self.stage
		phases = ['-s']
		for ii in range(stage):
			temp = []
			for ph in phases:
				a, b = self._phase_preset(ph)
				temp += [a, b]
			phases = temp.copy()
		return phases

	def infer(self,pulse):
		frep = pulse._frep_MHz
		period = 1e6/frep # ps
		new_frep = frep/self.num_pulse
		phases = self._cal_polar()
		p_s_delay = self.ps_delay *period
		pulse_train=[]
		for ind in range(self.num_pulse):
			new_p = pulse.create_cloned_pulse()
			new_phase = phases[ind]
			if ind%2==0:
				new_delay = -(ind*period/2)
			else:
				new_delay = -((ind-1)*period/2 + p_s_delay)
			new_p.set_frep_MHz(new_frep)
			new_p.add_time_offset(new_delay)
			pulse_info = {'pulse':new_p,'phase':new_phase,'displacement':0,'delay_avg':new_delay,'delay_diff':0,'name':'orig_'+str(ind)}
			pulse_train.append(pulse_info)
		return pulse_train

class StackStage:
	def __init__(self,PZM_fold=1, PZM_l0=0,optim_l0=0 ,noise_type='gauss',noise_mean=0., noise_sigma=1.,seed=None,name='s1'):
		self.fold = PZM_fold
		self.l0 = PZM_l0
		self.noise_sigma = noise_sigma
		self.optim_l0 = optim_l0
		if seed is not None:
			np.random.seed(seed)
		if noise_type=='gauss':
			self.noise = partial(np.random.normal,loc=noise_mean,scale=noise_sigma,size=1)
		else:
			self.noise = partial(np.random.uniform, low=noise_mean-noise_sigma, high=noise_mean+noise_sigma, size=1)

		self.name = name
		self.L = self.l0

	def _phase_postset(self,p1,p2):
		inp_ps=set([p1,p2])
		if inp_ps==set(['s','p']):
			return 'p'
		if inp_ps==set(['-s','-p']):
			return '-p'
		if inp_ps == set(['s', '-p']):
			return 's'
		if inp_ps == set(['-s', 'p']):
			return '-s'

	def cal_displacement(self):
		return self.fold*self.L

	def feedback(self,delta_l):
		self.L += delta_l

	def free_run(self,count=1):
		for ii in range(count):
			raw_noise = self.noise()[0]
			if raw_noise<0:
				raw_noise = max(raw_noise, -3*self.noise_sigma)
			else:
				raw_noise = min(raw_noise, 3 * self.noise_sigma)
			self.L += raw_noise

	def infer(self,pulse_train):
		d = self.cal_displacement()
		offset_ps = d / _C_mmps
		new_pulse_train = []
		n = len(pulse_train)
		for ind in range(0,n,2):
			pulse_info1,pulse_info2 = pulse_train[ind],pulse_train[ind+1]
			pu1 = pulse_info1['pulse']; ph1 = pulse_info1['phase']; de1 = pulse_info1['delay_avg']; na1 =pulse_info1['name']
			pu2 = pulse_info2['pulse']; ph2 = pulse_info2['phase']; de2 = pulse_info2['delay_avg']; na2 = pulse_info2['name']
			pu1 = pu1.create_cloned_pulse()
			if ph1 in ['s','-s']:
				pu1.add_time_offset(offset_ps)
				de1 += offset_ps
			if ph2 in ['s','-s']:
				pu2 = pu2.create_cloned_pulse()
				pu2.add_time_offset(offset_ps)
				de2 += offset_ps
			#pu3 = pu1.create_cloned_pulse()
			new_at = math.sqrt(1 / 2) * (pu1.AT + pu2.AT)
			pu1.set_AT(new_at)
			new_phase = self._phase_postset(ph1,ph2)
			new_name = self.name + '_' + (na1.split('_')[-1]+ '&' +na2.split('_')[-1])
			pulse_info = {'pulse':pu1,'phase':new_phase,'displacement':self.L,'delay_avg':(de1+de2)/2,'delay_diff':de1-de2,'name':new_name}
			new_pulse_train.append(pulse_info)

		return new_pulse_train

class MultiStack:
	def __init__(self,stacks_list,free_run_time=1):
		self.stacks_list = stacks_list
		self.stage = len(stacks_list)
		self.free_run_time = free_run_time

	def feedback(self,delta_l_list):
		for ii in range(self.stage):
			self.stacks_list[ii].feedback(delta_l_list[ii])

	def free_run(self,count=None):
		if count is None:
			count = self.free_run_time
		for ii in range(self.stage):
			self.stacks_list[ii].free_run(count)

	def infer(self, trains):
		trains_hist=[trains]
		for ii,stagestack in enumerate(self.stacks_list):
			trains = stagestack.infer(trains)
			trains_hist.append(trains)

		return trains_hist




