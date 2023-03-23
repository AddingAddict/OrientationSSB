#!/usr/bin/python

"""
script creates plot per simulation showing
- Connectivity (ff and rec)
- temporal plasticity behaviour and activity
- LGN input (correlations and patterns)
- examples of activity patterns in L4 and LGN 
- rec fields (RF)
- opm + RF properties
"""


import numpy as np
import pickle
import os
import sys
from scipy import linalg
import h5py

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from bettina.modeling.ori_dev_model import data_dir,image_dir,inputs,\
connectivity,network
from bettina.modeling.ori_dev_model.tools import plot_functions,analysis_tools,misc,\
update_params_dict,get_experim_observables
from bettina.modeling.ori_dev_model.data import data_lee



def plotting_routines(Version,load_external_from=""):
	## load parameters
	if load_external_from=="habanero":
		file_dir =\
		 "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/data/" +\
		 "layer4/habanero/v{v}/".format(v=Version)
	elif load_external_from=="aws":
		file_dir = data_dir + "layer4/aws/v{v}/".format(v=Version)
	else:
		file_dir = data_dir + "layer4/v{v}/".format(v=Version)
	params = pickle.load(open(file_dir + "config_v{v}.p".format(v=Version),"rb"))
	update_params_dict.update_params(params)
	print("params",params.keys())

	## ======================== NETWORK PARAMETERS ========================
	sc = params["Wret_to_lgn_params"]["sigma"]
	sr = params["W4to4_params"]["sigma_factor"]
	N4 = params["N4"]
	Nret = params["Nret"]
	Nlgn = params["Nlgn"]
	Nvert = params["Nvert"]
	rA_on = int(params["Wlgn_to4_params"]["r_A_on"] * N4)
	rA_off = int(params["Wlgn_to4_params"]["r_A_off"] * N4)
	DA_on = 2*rA_on + 5
	DA_off = 2*rA_off + 5
	inp_params = params["Inp_params"]
	random_seed = params["random_seed"]

	num_lgn_paths = params["num_lgn_paths"]
	avg_no_inp = params["Inp_params"]["avg_no_inp"]
	gamma_lgn = params["gamma_lgn"]

	Nvert = params["Nvert"]
	dt = params["dt"]

	## number of input patterns needed
	print("# of stimuli: {}".format(params["Inp_params"]["Nsur"]));sys.stdout.flush()

	last_timestep = params["runtime"]/dt
	params.update({
					"last_timestep" : last_timestep,
					"RF_mode" : "initialize",
					"system" : "one_layer",
					"Version" : Version
					})
	n = network.Network(Version,params)
	lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
							same_EI_input=True)
	Wret_to_lgn,Wlgn_to_4_init,arbor_on,arbor_off,arbor2,_,W4to4 = n.system

	Wlgn_to_4_init = Wlgn_to_4_init.reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	init_norm_alpha = np.nanmean(Wlgn_to_4_init,axis=(0,2))
	init_norm_x = np.nanmean(Wlgn_to_4_init,axis=1)


	## ======================== create image folder ========================
	if load_external_from=="habanero":
		image_dir_param = image_dir + "layer4/habanero/v{v}/".format(v=Version)
	elif load_external_from=="aws":
		image_dir_param = image_dir + "layer4/aws/v{v}/".format(v=Version)
	else:
		image_dir_param = image_dir + "layer4/tmp/v{v}/".format(v=Version)
	if not os.path.exists(image_dir_param):
		os.makedirs(image_dir_param)
	
	## ======================== load data ========================
	Wlgn_to_4_t,Wlgn_to_4,l4_t,l4I_t,l4,timesteps,keys = misc.load_data(Version,file_dir,params)
	print("check shape Wlgn_to_4_t",Wlgn_to_4_t.shape,Wlgn_to_4.shape,keys)
	lgn = lgn[:,:,::params["saving_stepsize"]]
	lgn = lgn[:,:,:Wlgn_to_4_t.shape[0]-1]

	## ======================== LAYER 4 CONNECTIVITY ========================
	W4 = connectivity.Connectivity((N4,N4), (N4,N4),random_seed=random_seed,Nvert=Nvert,\
									full_output=True)
	x_I = np.clip(params["W4to4_params"]["sigma"]*2,0.1,N4/2.)
	if "max_ew" not in params["W4to4_params"].keys():
		params["W4to4_params"]["max_ew"] = 0.8
	if "2pop" in params["W4to4_params"]["Wrec_mode"]:
		W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),\
			 random_seed=random_seed,Nvert=Nvert,full_output=True)
		W4to4,output_dict,_ = W4.create_matrix_2pop(params["W4to4_params"],\
							params["W4to4_params"]["Wrec_mode"])
	else:
		W4to4,output_dict,_ = W4.create_matrix(params["W4to4_params"],\
							 params["W4to4_params"]["Wrec_mode"],r_A=x_I)
		# if not simulate_activity:
		# 	W4to4 = np.linalg.inv(np.diagflat(np.ones(N4*N4*Nvert)) - W4to4)



	if (l4_t is None and not params["Inp_params"]["simulate_activity"]):
		l4_t = []
		W4to4_EE = W4to4[:N4*N4*Nvert,:N4*N4*Nvert]
		for i in range(lgn.shape[2]):
			iWff = i#int(np.floor(i/params["Inp_params"]["avg_no_inp"]))
			if (iWff)<Wlgn_to_4_t.shape[0]:
				l4 = np.dot(W4to4_EE, np.dot(Wlgn_to_4_t[iWff,0,:,:],lgn[0,:,i])) +\
					 np.dot(W4to4_EE, np.dot(Wlgn_to_4_t[iWff,1,:,:],lgn[1,:,i]))
				l4_t.append(l4 * gamma_lgn)
		l4_t = np.array(l4_t)
		l4 = l4_t[-1,:]
		print("l4_t={}, Wlgn_to_4={}, lgn={}, avg_no_inp={}".format(l4_t.shape,\
			Wlgn_to_4_t.shape,lgn.shape,avg_no_inp))


	#################################################################################
	############################# FIGURES ###########################################

	filename_list = ["experim_observables_E","experim_observables_I"]
	DA = max([DA_on,DA_off])
	nbins = 10
	bins_dict = {
					"Envelope width" : np.linspace(1,6,nbins,endpoint=True),
					"Orientation_fit" : np.linspace(0,np.pi,nbins,endpoint=True),
					"Orientation_FT" : np.linspace(0,np.pi,nbins,endpoint=True),
					"Relative phase" : np.linspace(0,2*np.pi,nbins,endpoint=True),
					"# half cycles" : np.linspace(1,5,nbins,endpoint=True),
					"Log aspect ratio" : np.linspace(-0.6,0.6,nbins,endpoint=True),
					"Fitted Gabor" : None,
					"Fit cost" : None,
					"Distance ON to center" : None,
					"Distance OFF to center" : None,
					"ON-OFF Distance to center" : None,
					"Center value RF" : None,
					"Average RF" : None,
					"ONOFF segregation" : np.linspace(0,1,nbins,endpoint=True),
					"ONOFF ratio" : np.linspace(0,1,nbins,endpoint=True),
					}
	for j in range(num_lgn_paths//2):
		pp = PdfPages(image_dir_param + "{}.pdf".format(filename_list[j]))
		sf = Wlgn_to_4[j*2:j*2+2,...]
		print("sf",np.nanmin(sf),np.nanmax(sf),np.sum(np.isfinite(sf)),sf.size)
		sf = sf.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
		RF,PF,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
		observables =\
		 get_experim_observables.compute_experimental_observables(Wlgn_to_4,RF,N4=N4,\
		 														  DA=DA,Nvert=Nvert,Nlgn=Nlgn)


		for key,value in sorted(observables.items()):
			# print(key,value.shape)
			ncol,nrow = 2,1
			fig = plt.figure(figsize=(6*ncol,5*nrow))
			fig.suptitle(key)

			if key in ("Orientation_FT",):
				ncol,nrow = 2,3
				fig = plt.figure(figsize=(6*ncol,5*nrow))
				fig.suptitle(key + "cardina/oblique")

				value2d = value.reshape(N4,N4*Nvert)
				tuning = np.angle(value2d)
				car = np.real(np.exp(1j*2*tuning))
				obl = np.imag(np.exp(1j*2*tuning))

				for l,lmap in enumerate([car,obl]):
					ax = fig.add_subplot(nrow,ncol,1+l*2)
					im=ax.imshow(lmap,interpolation="nearest",cmap="binary")
					plt.colorbar(im,ax=ax)

					ax = fig.add_subplot(nrow,ncol,2+l*2)
					ft_map = np.abs(np.fft.fftshift(np.fft.fft2(lmap-np.nanmean(lmap))))
					im=ax.imshow(ft_map,interpolation="nearest",cmap="binary")
					plt.colorbar(im,ax=ax)

				ax = fig.add_subplot(nrow,ncol,5)
				im=ax.imshow(misc.plot_complex_map(value2d),interpolation="nearest",cmap="hsv")
				plt.colorbar(im,ax=ax,orientation="horizontal")

				ax = fig.add_subplot(nrow,ncol,6)
				ft_map = np.abs(np.fft.fftshift(np.fft.fft2(tuning-np.nanmean(tuning))))
				im=ax.imshow(ft_map,interpolation="nearest",cmap="binary")
				plt.colorbar(im,ax=ax)

			elif key in ("Distance ON to center","Distance OFF to center",\
					   "ON-OFF Distance to center","Center value RF"):
				
				ax = fig.add_subplot(nrow,ncol,1)
				im=ax.imshow(value.reshape(N4,N4*Nvert),interpolation="nearest",cmap="binary")
				plt.colorbar(im,ax=ax)

				ax = fig.add_subplot(nrow,ncol,2)
				x = value[np.isfinite(value)]
				ax.plot(np.sort(x),np.linspace(0,1,x.size),"-k")
				ax.set_xlabel(key)
				ax.set_ylabel("Cumulative distrib.")
			
			elif key in ("ONOFF ratio","ONOFF segregation","Envelope width","Orientation_fit",\
						 "Relative phase","# half cycles","Log aspect ratio"):
				ncol,nrow = 2,1
				value2d = value.reshape(N4,N4*Nvert)
				cmap = "binary"
				if key=="Orientation_fit":
					cmap = "hsv"
					value2d = (np.angle(value2d)) % (2 * np.pi)
				fig = plt.figure(figsize=(6*ncol,5*nrow))
				fig.suptitle(key)
				ax = fig.add_subplot(nrow,ncol,1)
				im=ax.imshow(value2d,interpolation="nearest",cmap=cmap,\
							 vmin=bins_dict[key][0],vmax=bins_dict[key][-1])
				plt.colorbar(im,ax=ax)

				ax = fig.add_subplot(nrow,ncol,2)
				_,bins,_=ax.hist(value[np.isfinite(value)],bins=bins_dict[key],density=True)
				ax.set_xlabel(key)
				ax.set_ylabel("Number of units")

				if key in data_lee.exp_data.keys():
					x = data_lee.exp_data[key][:,0]
					bin_diff = x[1] - x[0]
					y = data_lee.exp_data[key][:,1]
					ax.plot(x,y/np.sum(y)/bin_diff,'--',c="k")
				# ax = fig.add_subplot(nrow,ncol,3)
				# ft_val = np.abs(np.fft.fftshift(np.fft.fft2(value2d-np.nanmean(value2d))))
				# im=ax.imshow(ft_val,interpolation="nearest",cmap="binary",\
				# 			 vmin=bins_dict[key][0],vmax=bins_dict[key][-1])
				# plt.colorbar(im,ax=ax)

			elif key=="Fitted Gabor":
				## imshow + comparison to original RF
				# observables["fitted_gabor"]
				ncol,nrow = 3,1
				fig = plt.figure(figsize=(6*ncol,5*nrow))
				ax = fig.add_subplot(nrow,ncol,1)
				ax.set_title("Fitted gabors")
				im=ax.imshow(observables["Fitted Gabor"][0,:,:],interpolation="nearest",\
							 cmap="RdBu_r")
				plt.colorbar(im,ax=ax)
				ax = fig.add_subplot(nrow,ncol,2)
				ax.set_title("Simulated RF")
				im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
				plt.colorbar(im,ax=ax)
				ax = fig.add_subplot(nrow,ncol,3)
				ax.set_title("Average RF")
				im=ax.imshow(observables["Average RF"],interpolation="nearest",cmap="RdBu_r")
				plt.colorbar(im,ax=ax)

			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

		pp.close()

		## save data into hdf5 file
		if (load_external_from=="" and os.environ["USER"]=="bettina"):
			cluster_name = "local"
		elif load_external_from!="":
			cluster_name = load_external_from
		elif os.environ["USER"]=="bh2757":
			cluster_name = "habanero"
		elif os.environ=="ubuntu":
			cluster_name = "aws"
		print("cluster_name",cluster_name)
		observables.update({"RF" : RF})
		print("observables",observables.keys())
		if j==0:
			filename = data_dir+"layer4/observables/observables_v{}.hdf5".format(Version)
			misc.write_to_hdf5(observables,cluster_name,Version,filename)



if __name__=="__main__":
	import argparse


	current_user = os.environ["USER"]
	on_habanero = current_user=="bh2757"
	Version_default = misc.get_version(data_dir + "layer4/",version=None) - 1

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Plot results from simulation.")
	parser.add_argument("--index", dest='idx', help="specifying run number (either\
						 individual runs as x1 x2 x3 or endpoints of successive numbers\
						 of runs as x1:xn)", required=False,  nargs="*",\
						 default=Version_default)
	parser.add_argument("--cl", dest='load_external_from', help="specify where data lies\
						(e.g. None, aws, habanero)", required=False,  nargs="*", default=[""])
	args = parser.parse_args()

	load_external_from = args.load_external_from[0]

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
		plotting_routines(Version,load_external_from=load_external_from)
		# try:
		# 	plotting_routines(Version,load_external_from=load_external_from)
		# except Exception as e:
		# 	misc.PrintException()
		# 	print("Ignoring version {}".format(Version))
		# 	pass
