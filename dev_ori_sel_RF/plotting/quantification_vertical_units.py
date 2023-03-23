import numpy as np
import pickle
import os
import sys
from scipy import linalg

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bettina.modeling.ori_dev_model import data_dir,image_dir,inputs,\
connectivity
from bettina.modeling.ori_dev_model.tools import misc,analysis_tools


def relation_between_connectivity_and_phase(Version,habanero=True):
	## load parameters
	# if habanero:
	# 	params = pickle.load(open(data_dir + "layer4/habanero/v{v}/config_v{v}.p".format(\
	# 		v=Version),"rb"))
	# else:
	params = pickle.load(open(data_dir + "layer4/v{v}/config_v{v}.p".format(\
		v=Version),"rb"))

	sc = params["Wret_to_lgn_params"]["sigma"]
	sr = params["W4to4_params"]["sigma_factor"]
	N4 = params["N4"]
	Nret = params["Nret"]
	Nlgn = params["Nlgn"]
	try:
		Nvert = params["Nvert"]
	except:
		Nvert = 1
	rA = int(params["Wlgn_to4_params"]["r_A"] * N4)
	DA = 2*rA + 5
	inp_params = params["Inp_params"]
	T_pd = params["Inp_params"]["pattern_duration"]
	if "random_seed" in params.keys():
		random_seed = params["random_seed"]
	else:
		random_seed = Version
	if "dt" in params.keys():
		dt = params["dt"]
	else:
		dt = 0.05
	if "simulate_activity" in params["Inp_params"].keys():
		simulate_activity = params["Inp_params"]["simulate_activity"]
	else:
		simulate_activity = True
	if not "onoff_corr_factor" in params["Inp_params"].keys():
		params["Inp_params"].update({"onoff_corr_factor" : 1.})

	## lgn input
	if "profile" in params["Wret_to_lgn_params"]:
		Wret_profile = params["Wret_to_lgn_params"]["profile"]
	else:
		Wret_profile = "Gaussian"
	Wret_to_lgn = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
		random_seed=random_seed,Nvert=1).create_matrix(\
		params["Wret_to_lgn_params"],Wret_profile) * params["Wret_to_lgn_params"]["gamma_ret"]

	if params["Wret_to_lgn_params"]["profile"]=="Gaussian_broadOFF":
		Wret_to_lgn_OFF = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
		random_seed=random_seed,Nvert=1).create_matrix(\
		{"ampl" : params["Wret_to_lgn_params"]["ampl"],\
		"noise" : params["Wret_to_lgn_params"]["noise"],\
		"sigma" : 2*params["Wret_to_lgn_params"]["sigma"]},\
		"Gaussian") * params["Wret_to_lgn_params"]["gamma_ret"]
		Wret_to_lgn = np.stack([Wret_to_lgn,Wret_to_lgn_OFF])


	# ## arbor + ff conn
	# Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
	#  random_seed=random_seed,Nvert=(1,Nvert))
	# arbor = Wlgn4.create_arbor(radius=params["Wlgn_to4_params"]["r_A"],\
	#  profile=params["Wlgn_to4_params"]["arbor_profile"])
	# arbor_bool = arbor>0
	# Won_to_4 = Wlgn4.create_matrix(params["Wlgn_to4_params"], "random_delta",\
	# 	r_A=params["Wlgn_to4_params"]["r_A"],profile_A="heaviside")
	# Wof_to_4 = Wlgn4.create_matrix(params["Wlgn_to4_params"], "random_delta",\
	# 	r_A=params["Wlgn_to4_params"]["r_A"],profile_A="heaviside")
	# Wlgn_to_4_init = np.stack([Won_to_4,Wof_to_4])
	

	## create folder
	image_dir_param = image_dir +\
	 "layer4/tmp/v{}_rA{:.1f}_srec{:.2f}_scc{:.2f}_{:s}_AVG_test/".format(\
	 	Version,\
		rA,	sr, sc,
		params["normalisation_mode"])
	print("image_dir_param",image_dir_param);sys.stdout.flush()
	if not os.path.exists(image_dir_param):
		os.makedirs(image_dir_param)
	
	## load data
	try:
		yt = np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version))
		print("yt.files",yt.files)
		timesteps = yt["Wt"].shape[0]
		Wlgn_to_4 = yt["Wt"][-1,:].reshape(2,N4**2*Nvert,Nlgn**2)
		## by default look at excitatory activity
		if "l4t" in yt.files:
			l4 = yt["l4t"][-1,:N4**2*Nvert]
			l4_t = yt["l4t"][:,:N4**2*Nvert]
		Wlgn_to_4_t = yt["Wt"].reshape(timesteps,2,N4*N4*Nvert,Nlgn*Nlgn)
		keys = sorted(yt.files)
		print("check shape Wlgn_to_4_t",Wlgn_to_4_t.shape)
	except:
		misc.PrintException()
		return None

	## Layer 4
	W4 = connectivity.Connectivity((N4,N4), (N4,N4),random_seed=random_seed,Nvert=Nvert)
	x_I = np.clip(params["W4to4_params"]["sigma"]*2,0.1,N4/2.)
	if "max_ew" not in params["W4to4_params"].keys():
		params["W4to4_params"]["max_ew"] = 0.8
	if "2pop" in params["W4to4_params"]["Wrec_mode"]:
		W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),\
			random_seed=random_seed,Nvert=Nvert)
		W4to4 = W4.create_matrix_2pop(params["W4to4_params"],params["W4to4_params"]["Wrec_mode"])
	else:
		W4to4 = W4.create_matrix(params["W4to4_params"], params["W4to4_params"]["Wrec_mode"],\
		 r_A=x_I)



	Wee = W4to4[:N4*N4*Nvert,:N4*N4*Nvert] 
	sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
	sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
	RFsd,_,_,_ = analysis_tools.get_RF_form(sd,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="diff_only")
	opm,Rn,pref_phase,gabors = analysis_tools.get_response(sd,DA,Nvert=Nvert)


	pref_phase = pref_phase.reshape(N4*N4,Nvert)
	Wee = Wee.reshape(N4*N4*Nvert,N4*N4,Nvert)
	ncluster = params["W4to4_params"]["ncluster"]
	rng = np.random.RandomState(random_seed*90)
	phase_field = rng.uniform(0,np.pi,np.prod(N4*N4)*Nvert)
	phase_field_binned = np.searchsorted(np.linspace(0,np.pi,ncluster,endpoint=False),\
		phase_field,side="right")
	phase_field_binned = phase_field_binned.reshape(N4*N4,Nvert)
	print("phase_field_binned",phase_field_binned.shape,np.unique(phase_field_binned))
	# above_thr = diff_phase>threshold
	# conn_matrix[above_thr] = 0.0

	pp = PdfPages(image_dir_param + "phases_vert_units.pdf")
	bins_uni = np.unique(phase_field_binned)
	phasemean_same_bin_total,phasemean_diff_bin_total = [],[]
	for subset_idx in bins_uni:
		phase_same_bin = pref_phase[phase_field_binned==subset_idx]
		if len(phase_same_bin)>1:
			phase_diff = analysis_tools.difference_in_phase(phase_same_bin,phase_same_bin,\
															in_deg=False)
			phasemean_same_bin_total.append(\
				np.nanmean(phase_diff[np.tri(len(phase_same_bin),k=-1,dtype=bool)]))
		phase_diff_bin = pref_phase[np.logical_not(phase_field_binned==subset_idx)]
		if (len(phase_diff_bin)>1 and len(phase_same_bin)>1):
			phase_diff = analysis_tools.difference_in_phase(phase_diff_bin,phase_same_bin,\
															in_deg=False)
			phasemean_diff_bin_total.append(\
				np.nanmean(phase_diff[np.tri(len(phase_diff_bin),len(phase_same_bin),\
							k=-1,dtype=bool)]))
	phasemean_same_bin_total = np.array(phasemean_same_bin_total)
	phasemean_diff_bin_total = np.array(phasemean_diff_bin_total)
	phasemean_same_bin_total = np.nanmean(phasemean_same_bin_total)
	phasemean_diff_bin_total = np.nanmean(phasemean_diff_bin_total)

	## calculate phase difference across vertical units
	phasemean_same_bin,phasemean_diff_bin = [],[]
	for iloc in range(N4*N4):
		for subset_idx in bins_uni:
			phase_same_bin = pref_phase[iloc,phase_field_binned[iloc,:]==subset_idx]
			if len(phase_same_bin)>1:
				phase_diff = analysis_tools.difference_in_phase(phase_same_bin,phase_same_bin,\
															in_deg=False)
				phasemean_same_bin.append(\
					np.nanmean(phase_diff[np.tri(len(phase_same_bin),k=-1,dtype=bool)]))
			phase_diff_bin = pref_phase[iloc,np.logical_not(phase_field_binned[iloc,:]==subset_idx)]
			if (len(phase_diff_bin)>1 and len(phase_same_bin)>1):
				phase_diff = analysis_tools.difference_in_phase(phase_diff_bin,phase_same_bin,\
															in_deg=False)
				phasemean_diff_bin.append(\
					np.nanmean(phase_diff[np.tri(len(phase_diff_bin),len(phase_same_bin),\
								k=-1,dtype=bool)]))
	phasemean_same_bin = np.array(phasemean_same_bin)
	phasemean_diff_bin = np.array(phasemean_diff_bin)

	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(121)
	ax.set_title("Phase diff between units of same subgroup")
	ax.hist(phasemean_same_bin,histtype="bar",\
		label="Vert. units")
	ax.axvline(x=phasemean_same_bin_total,ymin=0,ymax=1,c="k",label="All units")
	ax.legend(loc="best")
	# ax.hist(phase_same_bin)
	ax = fig.add_subplot(122)
	ax.set_title("Phase diff between units of diff subgroup")
	ax.hist(phasemean_diff_bin,histtype="bar")
	ax.axvline(x=phasemean_same_bin_total,ymin=0,ymax=1,c="k")
	# ax.hist(phase_diff_bin)
	pp.savefig(fig,dpi=300,bbox_inches='tight')
	plt.close(fig)
	pp.close()


if __name__=="__main__":
	import argparse

	current_user = os.environ["USER"]
	on_habanero = current_user=="bh2757"
	Version_default = misc.get_version(data_dir + "layer4/",version=None) - 1

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Plot results from simulation.")
	parser.add_argument("--index", dest='idx', help="specifying run number (either\
	 individual runs as x1 x2 x3 or endpoints of successive numbers of runs as x1:xn)",\
	 required=False,  nargs="*", default=Version_default)
	args = parser.parse_args()

	Versions = np.array([],dtype=int)
	## no arguments given, index falls back to default value
	if isinstance(args.idx,int):
		Versions = np.concatenate([Versions,np.array([args.idx])])
	## arguments given, either 
	else:
		for item in args.idx:
			if ":" in item:
				version_list = item.split(":")
				assert version_list[0]!="",\
				 "Error: start and end value expected for list of indices"
				assert version_list[1]!="",\
				 "Error: start and end value expected for list of indices"
				Versions = np.concatenate([Versions,np.arange(int(version_list[0]),\
					int(version_list[1])+1,1)])
			elif "-" in item:
				version_list = item.split("-")
				assert version_list[0]!="",\
				 "Error: start and end value expected for list of indices"
				assert version_list[1]!="",\
				 "Error: start and end value expected for list of indices"
				Versions = np.concatenate([Versions,np.arange(int(version_list[0]),\
					int(version_list[1])+1,1)])
			else:
				assert isinstance(int(item),int), "Error: int value expected for index"
				Versions = np.concatenate([Versions,np.array([int(item)])])
			


	for Version in np.unique(Versions):
		print("Plotting data from run {}".format(Version))
		try:
			relation_between_connectivity_and_phase(Version,habanero=on_habanero)
		except:
			misc.PrintException()
			print("Ignoring version {}".format(Version))
			pass
