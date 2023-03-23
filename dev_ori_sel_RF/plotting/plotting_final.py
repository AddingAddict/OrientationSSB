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
connectivity,system_generation,network
from bettina.modeling.ori_dev_model.tools import plot_functions,analysis_tools,misc,\
update_params_dict

def circvar(angles,high=2*np.pi,axis=None):
	angles = 1.*angles/high*2*np.pi
	if axis is not None:
		return 1-np.abs(np.nanmean(np.exp(1j*angles),axis=axis))
	else:
		return 1-np.abs(np.nanmean(np.exp(1j*angles)))




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
	params = update_params_dict.update_params(params)

	## ======================== NETWORK PARAMETERS ========================
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
	if not "onoff_rel_weight" in params["Inp_params"].keys():
		params["Inp_params"]["onoff_rel_weight"] = 1.
	if not "connectivity_type" in params["Wlgn_to4_params"].keys():
		params["Wlgn_to4_params"]["connectivity_type"] = "E"
		params["num_lgn_paths"] = 2
	num_lgn_paths = params["num_lgn_paths"]


	# ## Parameters
	Nvert = params["Nvert"]
	dt = params["dt"]

	last_timestep = params["runtime"]/dt
	## number of input patterns needed
	print("# of stimuli: {}".format(params["Inp_params"]["Nsur"]));sys.stdout.flush()

	params.update({
				"last_timestep" : last_timestep,
				#     "RF_mode" : "load_from_external",
				"RF_mode" : "initialize",
				"system" : "one_layer",
				"Version" : Version,
				"full_lgn_output" :  True
				})
	n=network.Network(Version,params)
	_,Wlgn_to_4,arbor,arbor2,_,W4to4 = n.system
	lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
								same_EI_input=True)

	# network_kwargs = {
	# 				"T_pd" : T_pd,
	# 				"last_timestep" : last_timestep,
	# 				"T_exp" : T_exp,
	# 				"RF_mode" : "initialize",
	# 				"system" : "one_layer",
	# 				"Version" : Version,
	# 				"full_lgn_output" :  False,
	# 				}
	# lgn,Wret_to_lgn = system_generation.generate_inputs(params,**network_kwargs)
	# system = system_generation.generate_network(params,**network_kwargs)
	# _,arbor,arbor2,init_weights,W4to4 = system
	# print("CHECK SHAPE",Wret_to_lgn.shape,arbor.shape,init_weights.shape,W4to4.shape)
	

	## ======================== create image folder ========================
	if load_external_from=="habanero":
		image_dir_param = image_dir + "layer4/habanero/v{v}/".format(v=Version)
	elif load_external_from=="aws":
		image_dir_param = image_dir + "layer4/aws/v{v}/".format(v=Version)
	else:
		image_dir_param = image_dir + "layer4/tmp/v{v}/".format(v=Version)
	print("image_dir_param",image_dir_param);sys.stdout.flush()
	if not os.path.exists(image_dir_param):
		os.makedirs(image_dir_param)
	
	## ======================== load data ========================

	y = np.load(file_dir + "y_v{v}.npz".format(v=Version))
	keys = y.files
	if "Wlgn_to_4" in y.files:
		Wlgn_to_4 = y["Wlgn_to_4"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	else:
		Wlgn_to_4 = y["W"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	print("check shape Wlgn_to_4_t",Wlgn_to_4.shape,keys)	


	#################################################################################
	############################# FIGURES ###########################################
	if True:#not os.path.exists(image_dir_param + "connectivity.pdf"):
		pp = PdfPages(image_dir_param + "connectivity.pdf")
		fig_list = plot_functions.plot_connectivity(W4to4,N4=N4,Nvert=Nvert,\
					Wrec_mode=params["W4to4_params"]["Wrec_mode"])
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()

	## change_weights
	xcoord1 = [6,5] #selective for 360 and 332
	xcoord2 = [3,7] #unselective/off for 360, sel for 332
	xcoords = [[6,5],[3,7],[0,0]]
	ncoords = 3

	if not os.path.exists(image_dir_param + "lgn_input.pdf"):
		pp = PdfPages(image_dir_param + "lgn_input.pdf")
		fig_list = plot_functions.plot_LGN_input_corr(lgn,Nlgn=Nlgn)
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()


	## receptive and projection fields
	## visualization of SD = S_on - S_off
	if True:#not os.path.exists(image_dir_param + "rec_field_final.pdf"):
		filename_list = ["rec_field_final","rec_field_final_I"]
		for j in range(num_lgn_paths//2):
			pp = PdfPages(image_dir_param + "{}.pdf".format(filename_list[j]))
			sf = Wlgn_to_4[j*2:j*2+2,...]
			sf = sf.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
			RF,PF,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)

			fig = plot_functions.plot_RF(RF,PF,N4=N4,Nvert=Nvert,DA=DA)
			fig.suptitle("RF")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			sf = sf.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
			fig = plot_functions.plot_RF_visual_space(sf[:,N4//2:N4//2+1,:,:,:],Nlgn=Nlgn,\
														N4=1,Nvert=N4)
			# fig.suptitle("RF in visual field")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			fig = plot_functions.plot_RF_visual_space(sf[:,N4//4*3:N4//4*3+1,:,:,:],\
														Nlgn=Nlgn,N4=1,Nvert=N4)
			# fig.suptitle("Section of RF in visual field")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
			fig = plot_functions.plot_RF_visual_space(sf,Nlgn=Nlgn,N4=N4,\
														Nvert=Nvert)
			# fig.suptitle("Section of RF in visual field")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## quantify on and off RF position scatter in LGN space
			RF_array = (Wlgn_to_4[j*2,...] - Wlgn_to_4[j*2+1,...]).reshape(N4**2*Nvert,Nlgn,Nlgn)
			centroids_on_off = analysis_tools.get_center_of_mass_subfields(RF_array)
			centroids_on_off = centroids_on_off.reshape(N4**2*Nvert,2,2)
			## pairwise distances between on/off subfield centers
			pairwise_distance_on_x = analysis_tools.distance_in_space_pbc(\
				centroids_on_off[None,:,0,0],centroids_on_off[:,None,0,0],border_value=Nlgn)
			pairwise_distance_on_y = analysis_tools.distance_in_space_pbc(\
				centroids_on_off[None,:,1,0],centroids_on_off[:,None,1,0],border_value=Nlgn)
			pairwise_distance_on =\
				np.sqrt(pairwise_distance_on_x**2 + pairwise_distance_on_y**2)
			pairwise_distance_on = pairwise_distance_on[np.tri(N4**2*Nvert,k=-1,dtype=bool)]

			pairwise_distance_off_x = analysis_tools.distance_in_space_pbc(\
				centroids_on_off[None,:,0,1],centroids_on_off[:,None,0,1],border_value=Nlgn)
			pairwise_distance_off_y = analysis_tools.distance_in_space_pbc(\
				centroids_on_off[None,:,1,1],centroids_on_off[:,None,1,1],border_value=Nlgn)
			pairwise_distance_off =\
				np.sqrt(pairwise_distance_off_x**2 + pairwise_distance_off_y**2)
			pairwise_distance_off = pairwise_distance_off[np.tri(N4**2*Nvert,k=-1,dtype=bool)]
			fig = plt.figure(figsize=(12,5))
			ax = fig.add_subplot(121)
			ax.plot(np.sort(pairwise_distance_on),np.linspace(0,1,len(pairwise_distance_on)),\
					"-k",label="on")
			ax.plot(np.sort(pairwise_distance_off),np.linspace(0,1,len(pairwise_distance_off)),\
					"-m",label="off")
			ax.set_xlabel("Pairwise distance")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(122)
			ax.plot([0,1],[np.nanmean(pairwise_distance_on),np.nanmean(pairwise_distance_off)],\
					"s")
			ax.set_xticks([0,1])
			ax.set_xticklabels(["on","off"])
			ax.set_xlim(-0.2,1.2)
			ax.set_ylim(bottom=0)
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## - relation cortical position vs rec field position
			x,y = np.meshgrid(np.arange(0,N4*Nvert,1),np.arange(0,N4,1))
			x = x.reshape(N4**2*Nvert)
			y = y.reshape(N4**2*Nvert)
			fig = plt.figure(figsize=(6*4,5))
			# fig.suptitle("")
			ax = fig.add_subplot(141)
			ax.plot(centroids_on_off[:,0,0],x,"ok",alpha=0.4,rasterized=True)
			ax.set_xlabel("On subfield vis field x location")
			ax.set_ylabel("Cortical x location")
			ax = fig.add_subplot(142)
			ax.plot(centroids_on_off[:,1,0],y,"ok",alpha=0.4,rasterized=True)
			ax.set_xlabel("On subfield vis field y location")
			ax.set_ylabel("Cortical y location")
			ax = fig.add_subplot(143)
			ax.plot(centroids_on_off[:,0,1],x,"ok",alpha=0.4,rasterized=True)
			ax.set_xlabel("Off subfield vis field x location")
			ax.set_ylabel("Cortical x location")
			ax = fig.add_subplot(144)
			ax.plot(centroids_on_off[:,1,1],y,"ok",alpha=0.4,rasterized=True)
			ax.set_xlabel("Off subfield vis field y location")
			ax.set_ylabel("Cortical y location")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## quantifiying how much off rec fields are anchored
			## - distance to center of RF (absolute and relative to on distance to center)
			RF_array = np.empty((N4,N4*Nvert,DA,DA))*np.nan
			for k in range(N4):
				for l in range(N4*Nvert):
					RF_array[k,l,:,:] = RF[0,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
			RF_array = RF_array.reshape(N4**2*Nvert,DA,DA)
			centroids_on_off_RF = analysis_tools.get_center_of_mass_subfields(RF_array)
			centroids_on_off_RF = centroids_on_off_RF.reshape(N4**2*Nvert,2,2)
			dist_on_center = np.sqrt((DA//2 - centroids_on_off_RF[:,0,0])**2 +\
			 					(DA//2 - centroids_on_off_RF[:,1,0])**2)
			dist_off_center = np.sqrt((DA//2 - centroids_on_off_RF[:,0,1])**2 +\
			 					(DA//2 - centroids_on_off_RF[:,1,1])**2)
			dist_on_minus_off = dist_on_center - dist_off_center
			dist_on_minus_off = dist_on_minus_off[np.isfinite(dist_on_minus_off)]
			## - RF value at center
			rf_center_value = RF_array[:,DA//2,DA//2]
			ncol,nrow = 4,1
			fig = plt.figure(figsize=(ncol*6,nrow*5))
			fig.suptitle("Distance between ON/OFF subfield center")
			ax = fig.add_subplot(nrow,ncol,1)
			ax.plot(np.sort(dist_on_center),np.linspace(0,1,len(dist_on_center)),\
					"-k",label="on")
			ax.plot(np.sort(dist_off_center),np.linspace(0,1,len(dist_off_center)),\
					"-m",label="off")
			ax.set_xlabel("On distance to center")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(nrow,ncol,2)
			ax.plot(np.sort(dist_on_minus_off),np.linspace(0,1,len(dist_on_minus_off)),\
					"-c",label="on-off")
			ax.set_xlabel("On-Off distance to center")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax.set_xlim(-5,5)
			ax.set_ylim(0,1)
			ax = fig.add_subplot(nrow,ncol,3)
			ax.plot(np.sort(rf_center_value),np.linspace(0,1,len(rf_center_value)),\
					"-k",label="RF center")
			ax.set_xlabel("Center value of RF")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(nrow,ncol,4)
			ax.errorbar([0],[np.nanmean(dist_on_center)],\
						yerr=np.nanstd(dist_on_center)/np.sqrt(len(dist_on_center)),fmt="k",\
						c="k",ecolor="k")
			ax.errorbar([1.],[np.nanmean(dist_off_center)],\
						yerr=np.nanstd(dist_off_center)/np.sqrt(len(dist_off_center)),fmt="k",\
						c="k",ecolor="k")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## quantify size of ON/OFF subfield (simple vs single sign RFs)
			RF_array = RF_array.reshape(N4,N4*Nvert,DA,DA)
			size_on = np.nansum(RF_array>0,axis=(2,3)).flatten()
			size_off = np.nansum(RF_array<0,axis=(2,3)).flatten()
			total_activation = np.empty((2,N4,N4*Nvert))*np.nan
			for k in range(N4):
				for l in range(N4*Nvert):
					total_activation[0,k,l] = np.nansum(RF_array[k,l,RF_array[k,l,:,:]>0])
					total_activation[1,k,l] = -np.nansum(RF_array[k,l,RF_array[k,l,:,:]<0])
			total_activation = total_activation.reshape(2,-1)
			fig = plt.figure(figsize=(3*6,5))
			fig.suptitle("ON/OFF subfield size")
			ax = fig.add_subplot(131)
			ax.plot(np.sort(size_on),np.linspace(0,1,N4*N4*Nvert),'-k',label="on")
			ax.plot(np.sort(size_off),np.linspace(0,1,N4*N4*Nvert),'-m',label="off")
			ax.set_xlabel("Size of subfield")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(132)
			ax.plot(np.sort(total_activation[0,:]),np.linspace(0,1,N4*N4*Nvert),'-k',label="on")
			ax.plot(np.sort(total_activation[1,:]),np.linspace(0,1,N4*N4*Nvert),'-m',label="off")
			ax.set_xlabel("Total weight of subfield")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(133)
			ax.plot(np.squeeze(np.sort(np.diff(total_activation,axis=0))),\
					np.linspace(0,1,N4*N4*Nvert),'-c')
			ax.set_xlabel("Weight of on-off subfield")
			ax.set_ylabel("Cumulative distribution")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## ARF (rel phase, # half cycles, log aspect ratio via gabor fitting)
			# sigma : width of envelope in x direction
			# theta : orientation of gabor in rad
			# Lambda : wavelength of gabor
			# psi : phase of gabor in rad
			## Kuo et al 2016:
			## log aspect ratio: log_10(sig_y/sig_x)
			## # half cycles: 8*f*sig_x
			sd = Wlgn_to_4[j*2,...] - Wlgn_to_4[j*2+1,...]
			sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
			_,Rn,_,_ = analysis_tools.get_response(sd,DA,Nvert=Nvert)
			fit_params,fitted_gabor,fit_cost,xmax,ymax = analysis_tools.fit_gabor_to_RF(\
															RF[0,...],DA=DA,\
															Nvert=Nvert,N4=N4,Rn=Rn)
			## fit parameters: sigma_x, theta, psi, Lambda=1/f, gamma
			fit_params = fit_params.reshape(N4**2*Nvert,-1)
			reasonable_fits = fit_params[:,0]>1.
			labels = ["Envelope width","Orientation","Relative phase","# half-cycles",\
						"Log Aspect ratio","Wavelength"]
			cmaps = ["binary","hsv","hsv","binary","binary","binary"]
			ax_xlims = [[0,12],[0,np.pi],[0,2*np.pi],[1,8],[-0.6,0.6],[3,30]]
			ax_ylims = [[0,None],[0,None],[0,130],[0,175],[0,120],[0,None]]
			gabor_params = np.empty((fit_params.shape[0],6))*np.nan
			# num_halfcycles = np.nanmean([xmax,ymax])*4/fit_params[:,3]
			num_halfcycles = 8*fit_params[:,0]/fit_params[:,3]
			gabor_params[:,:3] = fit_params[:,:3]
			gabor_params[:,3] = num_halfcycles
			gabor_params[:,4] = np.log10(1./fit_params[:,4])
			gabor_params[:,5] = fit_params[:,3]

			ncols,nrows = gabor_params.shape[-1],3
			fig = plt.figure(figsize=(6*ncols,nrows*5))
			fig.suptitle("Gabor fit parameters")
			for k in range(ncols):
				ax = fig.add_subplot(nrows,ncols,k+1)
				hbins = np.linspace(ax_xlims[k][0],ax_xlims[k][1],20)
				n,bins,_=ax.hist(gabor_params[reasonable_fits,k],bins=hbins)
				ax.set_xlabel(labels[k])
				ax = fig.add_subplot(nrows,ncols,k+1+ncols)
				im=ax.imshow(gabor_params[:,k].reshape(N4,N4*Nvert),interpolation="nearest",\
							cmap=cmaps[k])
				plt.colorbar(im,ax=ax)
				ax.set_xlabel(labels[k])
				ax = fig.add_subplot(nrows,ncols,k+1+2*ncols)
				binm = [item + (bins[1]-bins[0])*0.5 for item in bins[:-1]]
				ax.plot(binm,n,"-k")
				ax.set_xlabel(labels[k])
				ax.set_ylabel("Cumulative dist")
				ax.set_xlim(ax_xlims[k][0],ax_xlims[k][1])
				ax.set_ylim(ax_ylims[k][0],ax_ylims[k][1])
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			fig = plt.figure(figsize=(6,5))
			ax = fig.add_subplot(111)
			ax.plot(gabor_params[:,3].flatten(),gabor_params[:,4].flatten(),"ok",\
					rasterized=True)
			ax.set_xlabel("# half-cycles")
			ax.set_ylabel("Log Aspect ratio")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)


			ARF = 0
			for k in range(N4):
				for l in range(N4*Nvert):
						ARF += RF[0,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
			ARF /= N4**2*Nvert
			fig = plt.figure(figsize=(6*4,5))
			ax = fig.add_subplot(141)
			ax.set_title("Fitted Gabors")
			im=ax.imshow(fitted_gabor[0,:,:],interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(142)
			ax.set_title("Interpolated RFs")
			im=ax.imshow(fitted_gabor[1,:,:],interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(143)
			ax.set_title("Original RFs")
			im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(144)
			ax.set_title("Average RF")
			im=ax.imshow(ARF,interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
			pp.close()

	## receptive and projection fields for symmetrized weights t=W/sqrt(A)
	## visualization of SD = S_on - S_off
	if not os.path.exists(image_dir_param + "rec_field_additional.pdf"):
		filename_list = ["rec_field_additional","rec_field_additional_I"]
		Wlgn_to_4 = Wlgn_to_4.reshape(num_lgn_paths,N4*N4*Nvert,Nlgn*Nlgn)
		for j in range(num_lgn_paths//2):
			pp = PdfPages(image_dir_param + "{}.pdf".format(filename_list[j]))
			tf = Wlgn_to_4[j*2:j*2+2,...]/np.sqrt(arbor2[j*2:j*2+2,...])
			print("tf",tf.shape)
			tf[np.logical_not(arbor2[j*2:j*2+2,...]>0)] = 0.0
			tf = tf.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
			RF,PF,_,_ = analysis_tools.get_RF_form(tf,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)

			fig = plot_functions.plot_RF(RF,PF,N4=N4,Nvert=Nvert,DA=DA)
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## check whether sum of RF and PF decay to zero
			fig = plt.figure(figsize=(12,5))
			ax = fig.add_subplot(121)
			after_weights = np.sum(Wlgn_to_4[j*2:j*2+2,...],axis=1)
			ax.plot(after_weights.flatten(),'-k')
			ax.set_title("# of RF whose sum is 0: {}".format(np.sum(after_weights==0)))
			ax = fig.add_subplot(122)
			after_weights_a = np.sum(Wlgn_to_4[j*2:j*2+2,...],axis=2)
			ax.plot(after_weights_a.flatten(),'-k')
			ax.set_title("# of PF whose sum is 0: {}".format(np.sum(after_weights_a==0)))
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
			pp.close()


	## ============= OPM ==========================================
	if True:#not os.path.exists(image_dir_param + "opm.pdf"):
		filename_list = ["opm","opm_I"]
		for j in range(num_lgn_paths//2):
			pp = PdfPages(image_dir_param + "{}.pdf".format(filename_list[j]))

			sd = Wlgn_to_4[2*j,...] - Wlgn_to_4[2*j+1,...]
			sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
			RFsd,_,_,_ = analysis_tools.get_RF_form(sd,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,\
													mode="diff_only")
			opm,Rn,pref_phase,gabors = analysis_tools.get_response(sd,DA,Nvert=Nvert)
			gabors = np.swapaxes(gabors,1,2)
			gabors = gabors.reshape(DA*N4*2,DA*2*N4*Nvert)
			pref_ori = 0.5*np.angle(opm,deg=True)
			pref_ori = pref_ori - (np.sign(pref_ori)-1)*0.5*180
			sel = np.abs(opm)
			pref_phase /= 2.*np.pi/360

			fig = plt.figure(figsize=(6,5))
			ax = fig.add_subplot(111)
			ax.plot(np.sort(sel.flatten()),np.linspace(0,1,N4**2*Nvert),"-")
			ax.set_xlim(0,0.4)
			ax.set_ylim(0,1)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			fig = plt.figure(figsize=(12,10))
			ax = fig.add_subplot(221)
			ax.set_title("Preferred orientation")
			im=ax.imshow(pref_ori,interpolation="nearest",cmap='hsv')
			plt.colorbar(im,ax=ax,orientation="horizontal")
			if Nvert>1:
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
			ax = fig.add_subplot(222)
			ax.set_title("Selectivity")
			im=ax.imshow(np.abs(opm),interpolation="nearest",cmap='binary')
			plt.colorbar(im,ax=ax,orientation="horizontal")
			if Nvert>1:
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
			ax = fig.add_subplot(223)
			ax.set_title("Orientation preference map")
			im=ax.imshow(misc.plot_complex_map(opm),interpolation="nearest")
			plt.colorbar(im,ax=ax,orientation="horizontal")
			if Nvert>1:
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
			ax = fig.add_subplot(224)
			ax.set_title("Preferred phase")
			im=ax.imshow(pref_phase,interpolation="nearest",cmap="hsv",vmin=0,vmax=360)
			plt.colorbar(im,ax=ax,orientation="horizontal")
			if Nvert>1:
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
			## =============================================================================

			## === relationship between phase difference and connection weight =============
			pref_phase = pref_phase.reshape(N4*N4*Nvert)
			pairwise_phase_diff = analysis_tools.difference_in_phase(pref_phase,pref_phase)
			W4 = connectivity.Connectivity((N4,N4),(N4,N4),random_seed=random_seed,Nvert=Nvert)
			pairwise_distance = W4.create_matrix(params["W4to4_params"],"linear")
			pairs_within_rAdistance = pairwise_distance < (rA * 1./N4)
			nearest_neighbours = pairwise_distance <= (np.sqrt(2.) * 1./N4)
			lower_tri = np.tri(N4**2*Nvert,N4**2*Nvert,k=-1,dtype=bool)
			WEE = W4to4[:N4**2*Nvert,:N4**2*Nvert]

		
			lower_WEE = WEE[lower_tri]
			lower_phase_diff = pairwise_phase_diff[lower_tri]
			NN_low_WEE = WEE[np.logical_and(lower_tri,nearest_neighbours)]
			NN_low_phase_diff = pairwise_phase_diff[np.logical_and(lower_tri,nearest_neighbours)]
			rA_low_WEE = WEE[np.logical_and(lower_tri,pairs_within_rAdistance)]
			rA_low_phase_diff = pairwise_phase_diff[np.logical_and(lower_tri,pairs_within_rAdistance)]

			cc = analysis_tools.correlate(lower_WEE,lower_phase_diff)
			NN_cc = analysis_tools.correlate(NN_low_WEE,NN_low_phase_diff)
			rA_cc = analysis_tools.correlate(rA_low_WEE,rA_low_phase_diff)

			fig = plt.figure(figsize=(3*6,2*5))
			ax = fig.add_subplot(231)
			ax.set_title("Corr coeff = {:.2f}".format(cc))
			ax.plot(lower_WEE,lower_phase_diff,"o",alpha=0.4,rasterized=True)
			ax.set_xlabel("Rec connectivity weight")
			ax.set_ylabel("Phase diff (deg)")
			ax = fig.add_subplot(232)
			ax.set_title("Corr coeff = {:.2f}".format(NN_cc))
			ax.plot(NN_low_WEE,NN_low_phase_diff,"o",alpha=0.4,rasterized=True)
			ax.set_xlabel("Rec connectivity weight NN")
			ax.set_ylabel("Phase diff NN (deg)")
			ax = fig.add_subplot(233)
			ax.set_title("Corr coeff = {:.2f}".format(rA_cc))
			ax.plot(rA_low_WEE,rA_low_phase_diff,"o",alpha=0.4,rasterized=True)
			ax.set_xlabel("Rec connectivity weight within rA")
			ax.set_ylabel("Phase diff within rA (deg)")
			ax = fig.add_subplot(234)
			im=ax.imshow(WEE,interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
			ax.set_title("Rec connectivity weight")
			ax = fig.add_subplot(235)
			im=ax.imshow(pairwise_phase_diff,interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
			ax.set_title("Pairwise Phase diff")
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
			## =============================================================================

			fig = plt.figure(figsize=(12,5))
			ax = fig.add_subplot(121)
			ax.set_title("Fitted gabors")
			im=ax.imshow(gabors,interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(122)
			ax.set_title("RFs")
			im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			if Nvert>1:
				fig = plt.figure(figsize=(2*6,4*5))
				fig.suptitle("Variability in phase and orientation between vertical units")
				ax = fig.add_subplot(421)
				ax.set_title("RF")
				im=ax.imshow(RFsd,interpolation="nearest",cmap="RdBu_r")
				plt.colorbar(im,ax=ax,orientation="horizontal")
				for iN in range(N4-1):
					ax.axvline(Nvert*DA*(iN+1),ls="--",c="k",lw=1)
				ax = fig.add_subplot(422)
				ax.set_title("Pref phase (deg)")
				im=ax.imshow(pref_phase.reshape(N4,N4*Nvert),interpolation="nearest",\
							cmap="hsv",vmin=0,vmax=360)
				plt.colorbar(im,ax=ax,orientation="horizontal")
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)

				pref_phase = pref_phase.reshape(N4,N4,Nvert)
				cv_phase = circvar(pref_phase,high=360,axis=2)
				pref_ori = pref_ori.reshape(N4,N4,Nvert)
				cv_ori = circvar(2*pref_ori,high=360,axis=2)*0.5
				sel = sel.reshape(N4,N4,Nvert)
				sd_sel = np.nanstd(sel,axis=2)/np.nanmean(sel,axis=2)

				ax = fig.add_subplot(423)
				ax.set_title("CV phase")
				im=ax.imshow(cv_phase,interpolation="nearest",cmap="binary",vmin=0,vmax=1)
				plt.colorbar(im,ax=ax)
				ax = fig.add_subplot(424)
				ax.hist(cv_phase[np.isfinite(cv_phase)])
				ax.set_xlim(0,1)

				ax = fig.add_subplot(425)
				ax.set_title("CV Pref Ori")
				im=ax.imshow(cv_ori,interpolation="nearest",cmap="binary",vmin=0,vmax=1)
				plt.colorbar(im,ax=ax)
				ax = fig.add_subplot(426)
				ax.hist(cv_ori[np.isfinite(cv_ori)])
				ax.set_xlim(0,1)

				ax = fig.add_subplot(427)
				ax.set_title("Rel SD sel")
				im=ax.imshow(sd_sel,interpolation="nearest",cmap="binary",vmin=0,\
					vmax=np.max([1,np.nanmax(sd_sel)]))
				plt.colorbar(im,ax=ax)
				ax = fig.add_subplot(428)
				ax.hist(sd_sel[np.isfinite(sd_sel)])
				ax.set_xlim(0,np.max([1,np.nanmax(sd_sel)]))
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

				cv_phase_80perc = np.percentile(cv_phase.flatten(),80)
				a = 0
				idx_80perc = np.argmin(np.abs(cv_phase.flatten()-cv_phase_80perc))
				yidx = idx_80perc//N4
				xidx = idx_80perc%N4
				fig = plt.figure(figsize=(12,10))
				fig.suptitle("Example (80perc circ var phase)")
				ax = fig.add_subplot(211)
				ax.imshow(RFsd[yidx*DA:(yidx+1)*DA,xidx*DA:(xidx+Nvert)*DA],\
					interpolation="nearest",cmap="RdBu_r")
				ax = fig.add_subplot(212)
				ax.imshow(gabors[yidx*DA*2:(yidx+1)*DA*2,xidx*DA*2:(xidx+Nvert)*DA*2],\
					interpolation="nearest",cmap="RdBu_r")
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

				fig2 = plt.figure(figsize=(18,6))
				ax = fig2.add_subplot(231)
				ax.set_title("Orientation preference map")
				im=ax.imshow(misc.plot_complex_map(opm),interpolation="nearest")
				plt.colorbar(im,ax=ax,orientation="horizontal")
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
				ax = fig2.add_subplot(234)
				ax.set_title("Pref phase (deg)")
				im=ax.imshow(pref_phase.reshape(N4,N4*Nvert),interpolation="nearest",\
							cmap="hsv",vmin=0,vmax=360)
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
				plt.colorbar(im,ax=ax,orientation="horizontal")
				ax = fig2.add_subplot(132)
				ax.hist(cv_phase[np.isfinite(cv_phase)])
				ax.set_xlim(0,1)
				ax.set_xlabel("CV phase")
				ax = fig2.add_subplot(133)
				ax.hist(cv_ori[np.isfinite(cv_ori)])
				ax.set_xlim(0,1)
				ax.set_xlabel("CV pref ori")
				pp.savefig(fig2,dpi=300,bbox_inches='tight')
				plt.close(fig2)

			elif Nvert==1:
				pref_phase = pref_phase.reshape(N4*N4)
				pref_ori = pref_ori.reshape(N4*N4)

				W4 = connectivity.Connectivity((N4,N4),(N4,N4),random_seed=random_seed)
				pairwise_distance = W4.create_matrix(params["W4to4_params"],"linear")
				pairs_within_rAdistance = pairwise_distance < (rA * 1./N4)
				pairs_within_rAdistance[np.tri(N4**2,N4**2,k=0,dtype=bool)] = 0
				pairwise_phase_diff = analysis_tools.difference_in_phase(pref_phase,pref_phase)
				pairwise_ori_diff = analysis_tools.difference_in_pref_ori(pref_ori,pref_ori)

				phase_diff_rA = pairwise_phase_diff[pairs_within_rAdistance]
				ori_diff_rA = pairwise_ori_diff[pairs_within_rAdistance]

				not_nearest_neighbours = pairwise_distance > (np.sqrt(2.) * 1./N4)
				not_nearest_neighbours[np.tri(N4**2,N4**2,k=0,dtype=bool)] = 1
				pairwise_phase_diff[not_nearest_neighbours] = np.nan
				pairwise_ori_diff[not_nearest_neighbours] = np.nan

				fig = plt.figure(figsize=(6*2,10))
				ax = fig.add_subplot(221)
				try:
					ax.hist([ori_diff_rA,pairwise_ori_diff[np.isfinite(pairwise_phase_diff)]],\
							color=["b","m"],label=["within rA", "NN"],density=True)
				except:
					ax.hist([ori_diff_rA,pairwise_ori_diff[np.isfinite(pairwise_phase_diff)]],\
							color=["b","m"],label=["within rA", "NN"],normed=True)
				ax.set_xlabel("Pairwise pref ori difference NN")
				ax.legend(loc="best")
				ax = fig.add_subplot(222)
				try:
					ax.hist([phase_diff_rA,pairwise_phase_diff[np.isfinite(pairwise_phase_diff)]],\
							bins=7,color=["b","m"],label=["within rA", "NN"],density=True)
				except:
					ax.hist([phase_diff_rA,pairwise_phase_diff[np.isfinite(pairwise_phase_diff)]],\
							bins=7,color=["b","m"],label=["within rA", "NN"],normed=True)
				ax.set_xlabel("Pairwise phase difference NN")
				ax.legend(loc="best")
				plot_pairw_ori_diff = pairwise_ori_diff[np.logical_not(not_nearest_neighbours)]
				plot_pairw_phase_diff = pairwise_phase_diff[np.logical_not(not_nearest_neighbours)]
				ax = fig.add_subplot(223)
				ax.plot(plot_pairw_ori_diff,plot_pairw_phase_diff,"ok",\
						rasterized=True,alpha=0.4)
				ax.set_xlabel("Pairwise pref ori difference NN")
				ax.set_ylabel("Pairwise phase difference NN")
				ax = fig.add_subplot(224)
				try:
					ax.hist([plot_pairw_phase_diff[plot_pairw_ori_diff<20]],bins=7,color=["orange"],
							label=[r"$\Delta$PO<20deg"],density=True)
				except:
					ax.hist([plot_pairw_phase_diff[plot_pairw_ori_diff<20]],bins=7,color=["orange"],
							label=[r"$\Delta$PO<20deg"],normed=True)
				# ax.axvline(np.nanmedian(plot_pairw_phase_diff[plot_pairw_ori_diff>70]),c="g")
				ax.axvline(np.nanmedian(plot_pairw_phase_diff[plot_pairw_ori_diff<20]),c="orange")
				ax.legend(loc="best")
				ax.set_xlabel("Pairwise phase difference NN")
				ax.set_ylabel("Frequency")
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

			pp.close()

	try:
		del Wlgn_to_4
	except:
		pass

	return None



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
		try:
			plotting_routines(Version,load_external_from=load_external_from)
		except Exception as e:
			misc.PrintException()
			print("Ignoring version {}".format(Version))
			pass
