import copy
import os
import math
import numpy as np
import pynlo
import joblib
import seaborn as sns
import  pandas as pd
import matplotlib.pyplot as plt
from environment import PulseStacking


class Stacking_Power:
	def __init__(self, stage=2):
		self.stage = stage
		self._init_pulse()
		self.lim=1

	def _init_pulse(self):
		frep_MHz = 1000

		self.FWHM = 0.1  # pulse duration (ps)
		pulseWL = 1030  # pulse central wavelength (nm)
		EPP = 1.5e-10  # Energy per pulse (J) # 0.15 nj
		GDD = 0.0  # Group delay dispersion (ps^2)
		TOD = 0.0  # Third order dispersion (ps^3)

		# Window = 10.0  # simulation window (ps)
		Window = self.FWHM * 100  # simulation window (ps)
		Points = 2 ** 12
		# Points = 2 ** 13  # simulation points

		# create the pulse!
		self.pulse = pynlo.light.DerivedPulses.SechPulse(power=1,  # Power will be scaled by set_epp
		                                                 T0_ps=self.FWHM / 1.76,
		                                                 center_wavelength_nm=pulseWL,
		                                                 time_window_ps=Window,
		                                                 GDD=GDD, TOD=TOD,
		                                                 NPTS=Points,
		                                                 frep_MHz=frep_MHz,
		                                                 power_is_avg=False)

		# set the pulse energy!
		self.pulse.set_epp(EPP)


	def combine_pulse(self, spulse, offset_ps):
		spulse_2 = spulse.create_cloned_pulse()
		spulse_2.add_time_offset(offset_ps)
		new_at = math.sqrt(1/2) * (spulse.AT + spulse_2.AT)
		spulse_2.set_AT(new_at)
		return spulse_2

	def multi_combine_pulse(self,  offset_list):
		stage = len(offset_list)
		spulse = self.pulse.create_cloned_pulse()
		for ii in range(stage):
			spulse = self.combine_pulse(spulse, offset_list[ii])
		return spulse

	def delay_power_relation_1d(self, ts):

		offsets = []
		powers = []

		cnt=0
		for t in ts:
			ofs = [t]
			pp = self.multi_combine_pulse(ofs)
			pow = PulseStacking.pulse_f2_power(pp)
			offsets.append(ofs)
			powers.append(pow)
			cnt+=1
			if cnt%100==0:
				print('delay_power_relation_1d:',cnt)

		return np.array(offsets), np.array(powers)

	def delay_power_relation_2d(self, ts):

		offsets = []
		powers = []

		cnt=0
		for t1 in ts:
			for t2 in ts:
				ofs = [t1,t2]
				pp = self.multi_combine_pulse(ofs)
				pow = PulseStacking.pulse_f2_power(pp)
				offsets.append(ofs)
				powers.append(pow)
				cnt += 1
				if cnt%100==0:
					print('delay_power_relation_2d:', cnt)

		return np.array(offsets), np.array(powers)

	def delay_power_relation_3d(self, ts):

		offsets = []
		powers = []

		cnt=0
		for t1 in ts:
			for t2 in ts:
				for t3 in ts:
					ofs = [t1,t2,t3]
					pp = self.multi_combine_pulse(ofs)
					pow = PulseStacking.pulse_f2_power(pp)
					offsets.append(ofs)
					powers.append(pow)
					cnt += 1
					if cnt%100==0:
						print('delay_power_relation_3d:', cnt)

		return np.array(offsets), np.array(powers)

	def get_delay_power_relation(self,d=1,dir='demo/',num=100):

		if not os.path.exists(dir):
			os.mkdir(dir)

		pkl_name = dir + str(d)+'_dp.pkl'
		ts = np.round(np.linspace(-self.lim * self.FWHM, self.lim * self.FWHM, num=2*num+1),3)
		if d==3:
			offsets, powers = self.delay_power_relation_3d(ts)
		elif d==2:
			offsets, powers = self.delay_power_relation_2d(ts)
		else:
			offsets, powers = self.delay_power_relation_1d(ts)

		ret_dict = {'offsets':offsets,'powers':powers,'ts':ts}
		joblib.dump(ret_dict,pkl_name)

		fig_name = dir + str(d) + '_dp_pow.png'
		fig, ax = plt.subplots(figsize=(20,20))
		if d==1:
			power_df = pd.DataFrame(np.tile(powers.reshape((1,2*num+1)),(2*num+1,1)), columns=ts)
			sns.heatmap(power_df, xticklabels= True, yticklabels= False, )
			# sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
			#            square=True, cmap="YlGnBu")
			ax.set_title('timeoffset_power_heatmap_1stage', fontsize=30)
			#ax.set_ylabel('EPP (nj)', fontsize=20)
			ax.set_xlabel('1st stage time offset (ps)', fontsize=30)
			plt.savefig(fig_name)

			fig, ax = plt.subplots(figsize=(20, 20))
			plt.plot(offsets[:,0],powers)
			ax.set_title('timeoffset_power_relation_1stage', fontsize=30)
			ax.set_ylabel('EPP (nj)', fontsize=30)
			ax.set_xlabel('timeoffset (ps)', fontsize=30)
			plt.savefig(dir +  '1_curve_dp_pow.png')
			#plt.show()

		if d==2:
			power_df = pd.DataFrame(powers.reshape((2*num+1,2*num+1)), columns=ts, index=ts)
			sns.heatmap(power_df,   )
			# sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
			#            square=True, cmap="YlGnBu")
			ax.set_title('timeoffset_power_heatmap_2stage', fontsize=30)
			ax.set_ylabel('2nd stage time offset (ps)', fontsize=30)
			ax.set_xlabel('1st stage time offset (ps)', fontsize=30)
			plt.savefig(fig_name)
			#plt.show()


		return ret_dict



def main(d=2):
	sp = Stacking_Power()
	sp.get_delay_power_relation(d)

if __name__=='__main__':
	main()
