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
from skimage.filters import threshold_otsu

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcol

from bettina.modeling.ori_dev_model import data_dir,image_dir,inputs,\
connectivity,network
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
	update_params_dict.update_params(params)

	# params["Inp_params"]["off_bias_strength"] = 1.

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

	if not "onoff_corr_factor" in params["Inp_params"].keys():
		params["Inp_params"].update({"onoff_corr_factor" : 1.})
	if not "onoff_rel_weight" in params["Inp_params"].keys():
		params["Inp_params"]["onoff_rel_weight"] = 1.
	if not "connectivity_type" in params["Wlgn_to4_params"].keys():
		params["Wlgn_to4_params"]["connectivity_type"] = "E"
		params["num_lgn_paths"] = 2
	if not "saving_stepsize" in params.keys():
		params["saving_stepsize"] = 1
	if not "multiplicative_normalisation" in params.keys():
		params["multiplicative_normalisation"] = "x"
	num_lgn_paths = params["num_lgn_paths"]
	avg_no_inp = params["Inp_params"]["avg_no_inp"]
	gamma_lgn = params["gamma_lgn"]

	Nvert = params["Nvert"]
	dt = params["dt"]

	## number of input patterns needed
	print("# of stimuli: {}".format(params["Inp_params"]["Nsur"]));sys.stdout.flush()

	# params["W4to4_params"]["sigma_factor"] = 0.25

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
	print("lgn",lgn.shape)
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
		W4to4,output_dict = W4.create_matrix(params["W4to4_params"],\
							params["W4to4_params"]["Wrec_mode"])#,r_A=x_I)

		output_dict,_ = output_dict



	# if (l4_t is None and not params["Inp_params"]["simulate_activity"]):
	# 	I_crt = np.linalg.inv(np.diagflat(np.ones(N4*N4*2*Nvert)) - W4to4)
	# 	l4_t = np.dot(I_crt[:,:N4**2*Nvert], np.dot(Wlgn_to_4[0,:,:],lgn[0,:,:]) +\
	# 	  						 			 np.dot(Wlgn_to_4[1,:,:],lgn[1,:,:]))

	# 	if num_lgn_paths==4:
	# 		l4_toI = np.dot(I_crt[:,N4**2*Nvert:], np.dot(Wlgn_to_4[2,:,:],lgn[2,:,:]) +\
	# 		  						 			   np.dot(Wlgn_to_4[3,:,:],lgn[3,:,:]))
	# 		l4_t += l4_toI
	# 	l4_t *= gamma_lgn
	# 	l4I_t = l4_t[:N4**2,:]
	# 	l4_t = l4_t[N4**2:,:]

	# 	l4 = l4_t[-1,:]
	# 	print("l4_t={}, Wlgn_to_4={}, lgn={}, avg_no_inp={}".format(l4_t.shape,\
	# 			Wlgn_to_4_t.shape,lgn.shape,avg_no_inp))


	#################################################################################
	############################# FIGURES ###########################################
	try:
		norm = mcol.TwoSlopeNorm(vcenter=0)
	except:
		norm = None

	if True:#not os.path.exists(image_dir_param + "connectivity.pdf"):
		pp = PdfPages(image_dir_param + "connectivity.pdf")
		fig_list = plot_functions.plot_connectivity(W4to4,N4=N4,Nvert=Nvert,\
					output_dict=output_dict,Wrec_mode=params["W4to4_params"]["Wrec_mode"])
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()

	## change_weights
	xcoord1 = [6,5] #selective for 360 and 332
	xcoord2 = [3,7] #unselective/off for 360, sel for 332
	xcoords = [[6,5],[3,7],[0,0]]
	ncoords = 3
	if ("Wt" in keys and (not os.path.exists(image_dir_param + "change_Weights.pdf"))):
		pp = PdfPages(image_dir_param + "change_Weights.pdf")
		fig = plt.figure(figsize=(24,5))
		
		deltaW = Wlgn_to_4_t[:-1,...] - Wlgn_to_4_t[1:,...]
		print("deltaW",deltaW.shape)
		avg_deltaW_t = np.nanmean(deltaW[500:1000,...],axis=(0))
		avg_deltaW_xa = np.nanmean(deltaW,axis=(1,2,3))
		sf = avg_deltaW_t.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		for k,DA in enumerate([DA_on,DA_off]):
			delta_RF = np.zeros((DA*N4,DA*N4*Nvert))
			for i in range(N4*Nvert):
				for j in range(N4):
					son_ij = np.roll(np.roll(sf[k,j,i,:,:],shift=Nlgn//2-int(1.*j/N4 * Nlgn),\
									axis=0),shift=Nlgn//2-int(1.*i/N4/Nvert * Nlgn),axis=1)
					delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]

			ax = fig.add_subplot(1,4,k+1)
			im=ax.imshow(delta_RF,interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
			ax.set_ylabel("avg change in Won/off")
		ax = fig.add_subplot(143)
		ax.plot(avg_deltaW_xa,"-ok")
		ax.set_ylabel("avg cov between change in Won/off")
		ax = fig.add_subplot(144)
		ax.plot(np.nanmean(np.abs(deltaW[:,0,:,:]),axis=(1,2)),"-ok")
		ax.plot(np.nanmean(np.abs(deltaW[:,1,:,:]),axis=(1,2)),"-om")
		ax.set_ylabel("avg abs change in W_on/off")
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		nrow,ncol = 3,3
		fig = plt.figure(figsize=(50,50))
		for irow in range(nrow):
			for jcol in range(ncol):
					if deltaW.shape[0]>(jcol+irow*ncol):
						sf = deltaW[jcol+irow*ncol,:,:,:].reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
						DA = max([DA_on,DA_off])
						delta_RF = np.zeros((DA*N4,DA*N4*Nvert))
						for i in range(N4*Nvert):
							for j in range(N4):
								son_ij = np.roll(np.roll(sf[0,j,i,:,:]-sf[1,j,i,:,:],\
												shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
								delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
								 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
						ax = fig.add_subplot(nrow,ncol,jcol+irow*ncol+1)
						ax.set_title("t={}".format(jcol+irow*ncol))
						im=ax.imshow(delta_RF,interpolation="nearest",cmap="RdBu_r")
						plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		nrow,ncol = 3,3
		fig = plt.figure(figsize=(50,50))
		for irow in range(nrow):
			for jcol in range(ncol):
					if deltaW.shape[0]>(jcol+irow*ncol):
						sf =  Wlgn_to_4_t[jcol+irow*ncol,:,:,:].reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
						DA = max([DA_on,DA_off])
						delta_RF = np.zeros((DA*N4,DA*N4*Nvert))
						for i in range(N4*Nvert):
							for j in range(N4):
								son_ij = np.roll(np.roll(sf[0,j,i,:,:],\
												shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
								delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
								 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
						ax = fig.add_subplot(nrow,ncol,jcol+irow*ncol+1)
						ax.set_title("t={}".format(jcol+irow*ncol))
						im=ax.imshow(delta_RF,interpolation="nearest",cmap="RdBu_r")
						plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		nrow,ncol = 3,3
		fig = plt.figure(figsize=(50,50))
		for irow in range(nrow):
			for jcol in range(ncol):
					if deltaW.shape[0]>(jcol+irow*ncol):
						sf =  Wlgn_to_4_t[jcol+irow*ncol,:,:,:].reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
						DA = max([DA_on,DA_off])
						delta_RF = np.zeros((DA*N4,DA*N4*Nvert))
						for i in range(N4*Nvert):
							for j in range(N4):
								son_ij = np.roll(np.roll(sf[1,j,i,:,:],\
												shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
								delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
								 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
						ax = fig.add_subplot(nrow,ncol,jcol+irow*ncol+1)
						ax.set_title("t={}".format(jcol+irow*ncol))
						im=ax.imshow(delta_RF,interpolation="nearest",cmap="RdBu_r")
						plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		## plot dvelopment for two chosen RFs at cortical locations: 
		timesteps = np.min([1000,deltaW.shape[0]])#deltaW.shape[0]
		# for timestep in range(timesteps-600,timesteps,20):
		# 	fig = plt.figure(figsize=(6*ncoords,5))
		# 	sf = deltaW[timestep,:,:,:].reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		# 	for jc in range(ncoords):
		# 		ax1 = fig.add_subplot(1,ncoords,1+jc)
		# 		dsf = sf[0,xcoords[jc][1],xcoords[jc][0],:,:]-sf[1,xcoords[jc][1],\
		# 		xcoords[jc][0],:,:]
		# 		dsf = np.roll(np.roll(dsf,shift=Nlgn//2-xcoords[jc][1],axis=0),\
		# 							shift=Nlgn//2-xcoords[jc][0],axis=1)
		# 		im=ax1.imshow(dsf[5:15,5:15],interpolation="nearest",cmap="RdBu_r",\
		# 		vmin=-0.015,vmax=0.015)
		# 		plt.colorbar(im,ax=ax1)
		# 	pp.savefig(fig,dpi=300,bbox_inches='tight')
		# 	plt.close(fig)

		# fig = plt.figure(figsize=(12,5))
		# fig.suptitle("Avg change")
		# sf = np.nanmean(deltaW,axis=0).reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		# for jc in range(ncoords):
		# 	ax1 = fig.add_subplot(1,ncoords,1+jc)
		# 	dsf = sf[0,xcoords[jc][1],xcoords[jc][0],:,:]-sf[1,xcoords[jc][1],\
		# 	xcoords[jc][0],:,:]
		# 	dsf = np.roll(np.roll(dsf,shift=Nlgn//2-xcoords[jc][1],axis=0),\
		# 						shift=Nlgn//2-xcoords[jc][0],axis=1)
		# 	im=ax1.imshow(dsf[5:15,5:15],interpolation="nearest",cmap="RdBu_r",\
		# 	vmin=-0.015,vmax=0.015)
		# 	plt.colorbar(im,ax=ax1)

		# pp.savefig(fig,dpi=300,bbox_inches='tight')
		# plt.close(fig)
		pp.close()
	

	##  delta W, change_rec_field
	if ("Wt" in keys and not os.path.exists(image_dir_param + "change_rec_field.pdf")):
		pp = PdfPages(image_dir_param + "change_rec_field.pdf")
		dW = Wlgn_to_4_t[-1,...] - Wlgn_to_4_t[-2,...]
		dW = dW.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		DA = max([DA_on,DA_off])
		dRF,dPF,avgdRF,avgdPF = analysis_tools.get_RF_form(dW[:2,...],N4,Nlgn,DA,calc_PF=True,\
															Nvert=Nvert)
		fig = plot_functions.plot_RF(dRF,dPF,add_PF_row=True,avgRF=avgdRF,\
									 avgdPF=avgdPF,N4=N4,Nvert=Nvert,DA=DA)
		fig.suptitle("Change in last timestep (E)")
		pp.savefig(fig,dpi=300,bbox_inches="tight")
		plt.close(fig)
		if num_lgn_paths==4:
			dRF,dPF,avgdRF,avgdPF = analysis_tools.get_RF_form(dW[2:,...],N4,Nlgn,DA,\
																calc_PF=True,Nvert=Nvert)
			fig = plot_functions.plot_RF(dRF,dPF,add_PF_row=True,avgRF=avgdRF,\
										 avgdPF=avgdPF,N4=N4,Nvert=Nvert,DA=DA)
			fig.suptitle("Change in last timestep (I)")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()
	

	## visualize ff W and activity, IO_test
	if ("lgn_inp" in keys and l4_t is not None and\
	 (not os.path.exists(image_dir_param + "IO_test.pdf"))):
		fig = plt.figure()
		ax = fig.add_subplot(331)
		ax.set_title("input on(t=-11)")
		im=ax.imshow(lgn[0,:,0].reshape((Nlgn,Nlgn)),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(332)
		ax.set_title("input off(t=-11)")
		im=ax.imshow(lgn[1,:,0].reshape((Nlgn,Nlgn)),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(333)
		ax.set_title("FF conn (t=-1)")
		im=ax.imshow(Wlgn_to_4[1,:,Nlgn*Nlgn//2].reshape((N4,N4)),\
			interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(334)
		ax.set_title("Avg lgn input on")
		im=ax.imshow(np.nanmean(lgn[0,:,:],axis=1).reshape((Nlgn,Nlgn)),\
			interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(335)
		ax.set_title("Avg lgn input off")
		im=ax.imshow(np.nanmean(lgn[1,:,:],axis=1).reshape((Nlgn,Nlgn)),\
			interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(337)
		ax.set_title("L23 act (t=10)")
		im=ax.imshow(l4_t[1,:].reshape(N4,N4*Nvert),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(338)
		ax.set_title("L23 act (t=-1)")
		im=ax.imshow(l4_t[-1,:].reshape(N4,N4*Nvert),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(339)
		ax.set_title("Rec conn (t=-1)")
		im=ax.imshow(W4to4[:N4**2,N4].reshape((N4,N4*Nvert)),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		fig.savefig(image_dir_param + "IO_test.pdf")
		plt.close(fig)


	## temporal behaviour, dyn_test
	if True:#("Wt" in keys and (not os.path.exists(image_dir_param + "dyn_test.pdf")) and\
	 	# l4_t is not None):
		l4_t = l4_t.reshape(-1,N4**2*Nvert)
		pp = PdfPages(image_dir_param + "dyn_test.pdf")
		ncol,nrow = 4,2
		figt = plt.figure(figsize=(ncol*5,nrow*5))
		ax = figt.add_subplot(nrow,ncol,1)
		# ax.plot(l4[:,::100],"-")
		im=ax.imshow(l4_t,interpolation="nearest",cmap="binary",\
					 aspect=1.*l4_t.shape[1]/l4_t.shape[0])
		plt.colorbar(im,ax=ax)
		ax.set_ylabel("Timesteps")
		ax.set_xlabel("Act l4")
		ax = figt.add_subplot(nrow,ncol,2)
		ax.plot(np.nanmean(l4_t[1::avg_no_inp,],axis=1),'-k',label="mean l4")
		l4_t = l4_t.reshape(-1,N4,N4*Nvert)
		for i in range(3):
			ax.plot(l4_t[1::avg_no_inp,xcoords[i][1],xcoords[i][0]],'-',\
					label="l4,{}".format(i+1),alpha=0.5)
		# ax.plot(l4_t[1::2,N4//2*(N4+1)],'-c', label="l4, pxl2")
		ax.set_ylabel("mean/indiv activity l4")
		ax.set_xlabel("Timesteps")
		ax.legend(loc="best")
		l4_t = l4_t.reshape(-1,N4*N4*Nvert)
		if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
			l4I_t = l4I_t.reshape(-1,N4*N4*Nvert)
			ax = figt.add_subplot(nrow,ncol,3)
			ax.plot(np.nanmean(l4I_t[1::avg_no_inp,:],axis=1),'-k',label="mean l4 I")
			l4I_t = l4I_t.reshape(-1,N4,N4*Nvert)
			for i in range(3):
				ax.plot(l4I_t[1::avg_no_inp,xcoords[i][1],xcoords[i][0]],'-',\
						label="l4 I,{}".format(i+1),alpha=0.5)
			ax.set_ylabel("mean/indiv activity l4 I")
			ax.set_xlabel("Timesteps")
			ax.legend(loc="best")
			l4I_t = l4I_t.reshape(-1,N4*N4*Nvert)

		Wlgn_to_4_t = Wlgn_to_4_t.reshape(Wlgn_to_4_t.shape[0],num_lgn_paths,N4,N4*Nvert,-1)
		ax = figt.add_subplot(nrow,ncol,4)
		arbor2 = arbor2.reshape(num_lgn_paths,N4,N4*Nvert,-1)
		WE = Wlgn_to_4_t[:,:2,...]
		Wlim = params["Wlgn_to4_params"]["Wlim"]
		frozen = np.logical_or(WE[:,arbor2[:2,...]>0]<=0,WE[:,arbor2[:2,...]>0]>=Wlim)
		ax.plot(np.nanmean(frozen,axis=1), '-k')
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Fraction of frozen weights")
		ax.set_ylim(0,1)

		ax = figt.add_subplot(nrow,ncol,5)
		ax.plot(Wlgn_to_4_t[:,0,xcoords[0][1],xcoords[0][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[0][1],xcoords[0][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")
		ax = figt.add_subplot(nrow,ncol,6)
		ax.plot(Wlgn_to_4_t[:,0,xcoords[1][1],xcoords[1][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[1][1],xcoords[1][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")
		ax = figt.add_subplot(nrow,ncol,7)
		ax.plot(Wlgn_to_4_t[:,0,xcoords[2][1],xcoords[2][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[2][1],xcoords[2][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")
		ax = figt.add_subplot(nrow,ncol,8)
		WE = Wlgn_to_4_t[:,:2,...]
		arbor2 = arbor2.reshape(num_lgn_paths,N4,N4*Nvert,-1)
		ax.plot(np.nanmean(np.abs(WE[:,arbor2[:2,...].astype(bool)]),axis=1),"-k",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Avg weights")
		pp.savefig(figt,dpi=300,bbox_inches='tight')
		plt.close(figt)		

		if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
			ncol,nrow = 4,1
			fig = plt.figure(figsize=(6*ncol,5*nrow))
			ax = fig.add_subplot(nrow,ncol,1)
			ax.plot(Wlgn_to_4_t[:,0,xcoords[0][1],xcoords[0][0],:],"-k",alpha=0.5)
			ax.plot(Wlgn_to_4_t[:,1,xcoords[0][1],xcoords[0][0],:],"--r",alpha=0.5)
			ax.set_xlabel("Timesteps")
			ax.set_ylabel("Won,off (k=on,r=off)")
			ax = fig.add_subplot(nrow,ncol,2)
			ax.plot(Wlgn_to_4_t[:,0,xcoords[1][1],xcoords[1][0],:],"-k",alpha=0.5)
			ax.plot(Wlgn_to_4_t[:,1,xcoords[1][1],xcoords[1][0],:],"--r",alpha=0.5)
			ax.set_xlabel("Timesteps")
			ax.set_ylabel("Won,off (k=on,r=off)")
			ax = fig.add_subplot(nrow,ncol,3)
			ax.plot(Wlgn_to_4_t[:,0,xcoords[2][1],xcoords[2][0],:],"-k",alpha=0.5)
			ax.plot(Wlgn_to_4_t[:,1,xcoords[2][1],xcoords[2][0],:],"--r",alpha=0.5)
			ax.set_xlabel("Timesteps")
			ax.set_ylabel("Won,off (k=on,r=off)")
			ax = fig.add_subplot(nrow,ncol,4)
			WI = Wlgn_to_4_t[:,2:,...]
			ax.plot(np.nanmean(np.abs(WI[:,arbor2[2:,...].astype(bool)]),axis=1),"-k",alpha=0.5)
			ax.set_xlabel("Timesteps")
			ax.set_ylabel("Avg weights")
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			fig = plt.figure(figsize=(6,5))
			ax = fig.add_subplot(111)
			y1 = np.isfinite(np.nanmean(np.abs(WE[1:,arbor2[2:,...].astype(bool)]),axis=1))
			y2 = np.isfinite(np.nanmean(np.abs(WI[1:,arbor2[2:,...].astype(bool)]),axis=1))
			y3 = np.nanmean(l4_t[1::2,:],axis=1)
			y4 = np.nanmean(l4I_t[1::2,:],axis=1)
			print("y",y1.shape,y2.shape,y3.shape,y4.shape)
			ax.plot(y1.astype(float)*2,'-+k',label="WE")
			ax.plot(y2.astype(float),'-sm',label="WI")
			ax.plot(np.clip(y3/np.nanmean(y3),0,2),'--oc',label="rE")
			ax.plot(np.clip(y4/np.nanmean(y4),0,3),'--og',label="rI")
			ax.legend(loc="best")
			ax.set_xlim(110,120)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
		pp.close()
		Wlgn_to_4_t = Wlgn_to_4_t.reshape(Wlgn_to_4_t.shape[0],num_lgn_paths,N4*N4*Nvert,-1)
		arbor2 = arbor2.reshape(num_lgn_paths,N4*N4*Nvert,-1)
		try:
			del WE
			del WI
		except:
			pass
	

	## correlation between cortical layer and lgn input, corr_ff_rec
	if True:#("Wt" in keys and not os.path.exists(image_dir_param + "corr_ff_rec.pdf")):
		pp = PdfPages(image_dir_param + "corr_ff_rec.pdf")
		if "cct" in keys:
			cc_recff = yt["cct"]
			fig = plt.figure(figsize=(12,5))
			fig.suptitle("cc_recff")
			ax = fig.add_subplot(121)
			ax.plot(cc_recff[:,0],'-o')
			ax.plot(cc_recff[:,1],'-o')
			ax = fig.add_subplot(122)
			im=ax.imshow(l4.reshape(N4,N4*Nvert),interpolation="nearest",\
				cmap="binary")
			plt.colorbar(im,ax=ax)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

		
		inp_on,inp_of = [],[]
		for it in range(lgn.shape[2]):
		# for it in range(lgn.shape[2]-1,lgn.shape[2]):
			Wtime = it//avg_no_inp + 1
			try:
				inp_on.append(np.dot(Wlgn_to_4_t[Wtime,0,...],lgn[0,:,it]).reshape(N4,N4*Nvert))
				inp_of.append(np.dot(Wlgn_to_4_t[Wtime,1,...],lgn[1,:,it]).reshape(N4,N4*Nvert))
			except:
				pass
		inp_on = np.array(inp_on)
		inp_of = np.array(inp_of)

		ncol,nrow = 4,2
		fig = plt.figure(figsize=(ncol*6,nrow*5))
		fig.suptitle("Avg Wlgn_to_4_t * lgn")
		ax = fig.add_subplot(nrow,ncol,1)
		ax.set_title("on avg")
		im=ax.imshow(np.nanmean(inp_on,axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,2)
		ax.set_title("off avg")
		im=ax.imshow(np.nanmean(inp_of,axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,3)
		ax.set_title("on std")
		im=ax.imshow(np.nanstd(inp_on,axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,4)
		ax.set_title("off std")
		im=ax.imshow(np.nanstd(inp_of,axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		
		nmax = 200
		ax = fig.add_subplot(nrow,ncol,5)
		ax.set_title("on, avg first {} patterns".format(nmax))
		im=ax.imshow(np.nanmean(inp_on[:nmax,:],axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,6)
		ax.set_title("off, avg first {} patterns".format(nmax))
		im=ax.imshow(np.nanmean(inp_of[:nmax,:],axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,7)
		ax.set_title("on, std first {} patterns".format(nmax))
		im=ax.imshow(np.nanstd(inp_on[:nmax,:],axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,8)
		ax.set_title("off, std first {} patterns".format(nmax))
		im=ax.imshow(np.nanstd(inp_of[:nmax,:],axis=0),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		
		ncol,nrow = 4,1
		fig = plt.figure(figsize=(ncol*6,nrow*5))
		ax = fig.add_subplot(nrow,ncol,1)
		ax.set_title("Avg ON, SD={:.2f}".format(np.nanstd(np.nanmean(lgn[0,:,:],axis=1))))
		im=ax.imshow(np.nanmean(lgn[0,:,:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,2)
		ax.set_title("Avg OFF, SD={:.2f}".format(np.nanstd(np.nanmean(lgn[1,:,:],axis=1))))
		im=ax.imshow(np.nanmean(lgn[1,:,:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,3)
		ax.set_title("SD ON")
		im=ax.imshow(np.nanstd(lgn[0,:,:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,4)
		ax.set_title("SD OFF")
		im=ax.imshow(np.nanstd(lgn[1,:,:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		ncol,nrow = 2,1
		fig = plt.figure(figsize=(ncol*6,nrow*5))
		ax = fig.add_subplot(nrow,ncol,1)
		im=ax.imshow(inp_on[0,:,:] + inp_of[0,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,2)
		im=ax.imshow(inp_on[-1,:,:] + inp_of[-1,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		
		## LAYER 4 ACTIVITY
		if l4_t is not None:
			l4_t = l4_t.reshape(-1,N4,N4*Nvert)
			fig = plt.figure(figsize=(25,10))
			fig.suptitle("l4 E act")
			n = 60
			l4_first_npatterns = l4_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
			n = 10
			l4_last_npatterns = l4_t[-n*avg_no_inp+(l4_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
			l4_last_npatterns = l4_last_npatterns[::-1,...]
			l4_to_show = np.concatenate([l4_first_npatterns,l4_last_npatterns])
			print("l4_to_show",l4_to_show.shape,l4_last_npatterns.shape,l4_t.shape)
			fig,_,_=plot_functions.grid_plot_twolayer(l4_to_show,fig,ncol=10,nrow=7)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
				l4I_t = l4I_t.reshape(-1,N4,N4*Nvert)
				fig = plt.figure(figsize=(25,10))
				fig.suptitle("l4 I act")
				n = 10
				l4I_first_npatterns = l4I_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
				l4I_last_npatterns = l4I_t[-n*avg_no_inp+(l4I_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
				l4I_last_npatterns = l4I_last_npatterns[::-1,...]
				l4I_to_show = np.concatenate([l4I_first_npatterns,l4I_last_npatterns])
				fig,_,_=plot_functions.grid_plot_twolayer(l4I_to_show,fig,ncol=n,nrow=2)
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

		if "2pop" not in params["W4to4_params"]["Wrec_mode"]:
			l4_filt = []
			for it in range(lgn.shape[2]):
				Wtime = it//avg_no_inp + 1
				l4_filt.append( np.dot(W4to4, np.dot(Wlgn_to_4_t[Wtime,0,...],lgn[0,:,it])) +\
								np.dot(W4to4, np.dot(Wlgn_to_4_t[Wtime,1,...],lgn[1,:,it])))
			l4_filt = np.array(l4_filt) * gamma_lgn

			fig = plt.figure(figsize=(25,25))
			fig.suptitle("l4 act=filtered lgn inp")
			for i in range(25):
				if l4_filt.shape[0]>(1+i):
					ax = fig.add_subplot(5,5,i+1)
					ax.set_title(str(i))
					im=ax.imshow(l4_filt[1+i,:].reshape(N4,N4*Nvert),interpolation="nearest",\
								 cmap="binary")
					plt.colorbar(im,ax=ax)
					ax.set_xticks([])
					ax.set_yticks([])
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
			
		## LGN INPUT
		labels = ["ON LGN input to E","OFF LGN input to E"]
		n = np.min([10,lgn.shape[-1]])
		timesteps_to_plot = np.linspace(0,n,n).astype(int)
		lgn_first_patterns_on,lgn_first_patterns_of = [],[]
		for istep in range(n):
			iplot = timesteps_to_plot[istep]
			if (iplot<Wlgn_to_4_t.shape[0] and iplot<lgn.shape[2]):
				dotproduct_on = gamma_lgn * (np.dot(Wlgn_to_4_t[iplot,0,...],lgn[0,:,iplot]))
				dotproduct_of = gamma_lgn * (np.dot(Wlgn_to_4_t[iplot,1,...],lgn[1,:,iplot]))
				lgn_first_patterns_on.append(dotproduct_on)
				lgn_first_patterns_of.append(dotproduct_of)
		# for istep in range(n):
		# 	dotproduct_on = gamma_lgn * (np.dot(Wlgn_to_4_t[-n+istep-1,0,...],lgn[0,:,-n+istep]))
		# 	dotproduct_of = gamma_lgn * (np.dot(Wlgn_to_4_t[-n+istep-1,1,...],lgn[1,:,-n+istep]))
		# 	lgn_first_patterns_on.append(dotproduct_on)
		# 	lgn_first_patterns_of.append(dotproduct_of)
		lgn_first_patterns_on = np.array(lgn_first_patterns_on).reshape(-1,N4,N4*Nvert)
		lgn_first_patterns_of = np.array(lgn_first_patterns_of).reshape(-1,N4,N4*Nvert)

		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns_on,None,ncol=5,nrow=2)
		fig.suptitle(labels[0])
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns_of,None,ncol=5,nrow=2)
		fig.suptitle(labels[1])
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)


		## CUMULATIVE LGN INPUT
		labels = ["Cumul. ON LGN input to E","Cumul OFF LGN input to E"]
		n = np.min([10,lgn.shape[-1]])
		timesteps_to_plot = np.linspace(0,n,n).astype(int)
		lgn_first_patterns_on,lgn_first_patterns_of = [],[]
		dotproduct_on,dotproduct_of = 0,0
		for istep in range(n):
			iplot = timesteps_to_plot[istep]
			if (iplot<lgn.shape[2] and iplot<Wlgn_to_4_t.shape[0]):
				dotproduct_on += gamma_lgn * (np.dot(Wlgn_to_4_t[iplot,0,...],lgn[0,:,iplot]))
				dotproduct_of += gamma_lgn * (np.dot(Wlgn_to_4_t[iplot,1,...],lgn[1,:,iplot]))
				lgn_first_patterns_on.append(dotproduct_on/(istep+1))
				lgn_first_patterns_of.append(dotproduct_of/(istep+1))
		# for istep in range(n):
		# 	dotproduct_on = gamma_lgn * (np.dot(Wlgn_to_4_t[-n+istep-1,0,...],lgn[0,:,-n+istep]))
		# 	dotproduct_of = gamma_lgn * (np.dot(Wlgn_to_4_t[-n+istep-1,1,...],lgn[1,:,-n+istep]))
		# 	lgn_first_patterns_on.append(dotproduct_on)
		# 	lgn_first_patterns_of.append(dotproduct_of)
		lgn_first_patterns_on = np.array(lgn_first_patterns_on).reshape(-1,N4,N4*Nvert)
		lgn_first_patterns_of = np.array(lgn_first_patterns_of).reshape(-1,N4,N4*Nvert)

		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns_on,None,ncol=5,nrow=2)
		fig.suptitle(labels[0])
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns_of,None,ncol=5,nrow=2)
		fig.suptitle(labels[1])
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)


		## AVERAGE ACTIVITY/INPUT
		nrow,ncol = num_lgn_paths//2,4
		fig = plt.figure(figsize=(ncol*6,5*nrow))
		if ("l4t" in keys and "Wt" in keys):
			ax = fig.add_subplot(nrow,ncol,1)
			ax.set_title("l4 avg E")
			im=ax.imshow(np.nanmean(l4_t[1::avg_no_inp,:,:],axis=0),interpolation="nearest",\
						cmap="binary")
			plt.colorbar(im,ax=ax)
			if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
				ax = fig.add_subplot(nrow,ncol,ncol+1)
				ax.set_title("l4 avg I")
				im=ax.imshow(np.nanmean(l4I_t[1::avg_no_inp,:,:],axis=0),interpolation="nearest",\
							cmap="binary")
				plt.colorbar(im,ax=ax)

		labels = ["Avg ON LGN to E","Avg OFF LGN to E","Avg ON LGN to I","Avg OFF LGN to I"]
		for j in range(2):
			ax = fig.add_subplot(nrow,ncol,2+j)
			ax.set_title(labels[j])
			im=ax.imshow(np.nanmean(lgn[j,:,:],axis=1).reshape(Nlgn,Nlgn),interpolation="nearest",\
						cmap="binary")
			plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(nrow,ncol,ncol)
		ax.set_title("lgn on + off avg")
		im=ax.imshow(np.nanmean(lgn[:2,...],axis=(0,2)).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)

		if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
			for j in range(2):
				ax = fig.add_subplot(nrow,ncol,ncol+2+j)
				ax.set_title(labels[2+j])
				im=ax.imshow(np.nanmean(lgn[j+2,:,:],axis=1).reshape(Nlgn,Nlgn),\
							interpolation="nearest",cmap="binary")
				plt.colorbar(im,ax=ax)

			ax = fig.add_subplot(nrow,ncol,ncol*2)
			ax.set_title("lgn on + off avg")
			im=ax.imshow(np.nanmean(lgn[2:,...],axis=(0,2)).reshape(Nlgn,Nlgn),\
						interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)


		if False:#(params["Inp_params"]["Nsur"]<=10 and ("l4t" in keys and "Wt" in keys)):
			constraint_vec = np.load(data_dir + "layer4/P_orth/N4{}_Nlgn{}{}_rA{}_{}.npy".format(\
							N4,Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
							np.around(params["Wlgn_to4_params"]["r_A"],2),\
							params["Wlgn_to4_params"]["arbor_profile_on"]))
			c_orth = constraint_vec[:constraint_vec.shape[0]//2,:]
			s_orth = constraint_vec[constraint_vec.shape[0]//2:,:]

			arbor2 = np.stack([arbor_bool,arbor_bool])
			# l4const = np.ones((N4*N4))
			# lgnconst = np.ones((2,Nlgn*Nlgn))
			for i in range(lgn.shape[2]):
			# for i in range(1):
				# Wstep.append( l4_t[i*2+1,None,:,None] * lgn[:,None,:,i] )
				Wstep = params["beta_P"] * l4_t[i//avg_no_inp+1,None,:,None] *\
				 		lgn[:,None,:,i] * arbor_bool[None,:,:]
				# Wstep = params["beta_P"] * l4const[None,:,None] * lgn[:,None,:,i] *\
				#  arbor_bool[None,:,:]
				WstepA = Wstep[arbor2]

				WstepA -= np.sum(s_orth*np.dot(c_orth,WstepA)[:,None],axis=0)
				print("WstepA",i,np.nanmin(WstepA),np.nanmax(WstepA),np.nanmin(Wstep),\
					np.nanmax(Wstep))
				Wstep[arbor2] = WstepA
				# Wstep = Wstep.reshape(num_lgn_paths,N4,N4,Nlgn,Nlgn)
				Wlgn_to_4_init += 0.05*Wstep

			# print("Wstep",np.sum(Wstep,axis=(0,3,4))[13:15,12:14],np.sum(Wstep,axis=(1,2))[:,13:15,12:14])
			Wstep = Wlgn_to_4_init.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
			arbor_bool = arbor_bool.reshape(N4,N4*Nvert,Nlgn,Nlgn)
			stRF = np.zeros((3,DA*N4,DA*N4*Nvert))
			stPF = np.zeros((3,DA*N4,DA*N4*Nvert))
			avgstRF = np.zeros((N4,N4*Nvert))
			avgstPF = np.zeros((N4,N4*Nvert))
			for i in range(N4*Nvert):
				for j in range(N4):
					son_ij = np.roll(np.roll(Wstep[0,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
					sof_ij = np.roll(np.roll(Wstep[1,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
					stRF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2] -\
					 sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
					stRF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
					stRF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
					avgstRF[j,i] = np.sum(stRF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA])
			for i in range(Nlgn):
				for j in range(Nlgn):
					son_ij = np.roll(np.roll(Wstep[0,:,:,j,i],shift=N4//2-j,axis=0),shift=N4//2-i,axis=1)
					sof_ij = np.roll(np.roll(Wstep[1,:,:,j,i],shift=N4//2-j,axis=0),shift=N4//2-i,axis=1)
					arb_ij = np.roll(np.roll(arbor_bool[:,:,j,i],shift=N4//2-j,axis=0),shift=N4//2-i,axis=1)
					stPF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2] -\
					 sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]

					stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]
					# print("mean1",np.nanmean(stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]))
					# stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] -= \
					#  np.nansum(stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]) /\
					#  np.nansum(arb_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2])
					# print("mean2",np.nanmean(stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]))
					# stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = (stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]>0).astype(float)

					stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]
					# stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] -= \
					#  np.nansum(stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA]) /\
					#  np.nansum(arb_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2])
					# stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = (stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA]>0).astype(float)

					avgstPF[j,i] = np.sum(stPF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]) +\
					np.sum(stPF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA])

			fig = plt.figure(figsize=(24,10))
			## change in receptive field
			ax = fig.add_subplot(241)
			ax.set_title("S_D (PF)")
			im = ax.imshow(avgstRF,interpolation='nearest',cmap='RdBu_r')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(242)
			ax.set_title("dS_D")
			im = ax.imshow(stRF[0,:,:],interpolation='nearest',cmap='RdBu_r')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(243)
			ax.set_title("dS_on")
			im = ax.imshow(stRF[1,:,:],interpolation='nearest',cmap='binary')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(244)
			ax.set_title("dS_of")
			im = ax.imshow(stRF[2,:,:],interpolation='nearest',cmap='binary')
			plt.colorbar(im,ax=ax)

			## change in projective field
			ax = fig.add_subplot(245)
			ax.set_title("S_D (PF)")
			im = ax.imshow(avgstPF,interpolation='nearest',cmap='RdBu_r')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(246)
			ax.set_title("S_D (PF)")
			im = ax.imshow(stPF[0,:,:],interpolation='nearest',cmap='RdBu_r')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(247)
			ax.set_title("S_on (PF)")
			im = ax.imshow(stPF[1,:,:],interpolation='nearest',cmap='binary')
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(248)
			ax.set_title("S_of (PF)")
			im = ax.imshow(stPF[2,:,:],interpolation='nearest',cmap='binary')
			plt.colorbar(im,ax=ax)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
		pp.close()


	if True:#not os.path.exists(image_dir_param + "lgn_input.pdf"):
		pp = PdfPages(image_dir_param + "lgn_input.pdf")
		fig_list = plot_functions.plot_LGN_input_corr(lgn,Nlgn=Nlgn)
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()


	## receptive and projection fields
	## visualization of SD = S_on - S_off
	if True:#not os.path.exists(image_dir_param + "rec_field.pdf"):
		filename_list = ["rec_field","rec_field_I"]
		DA = max([DA_on,DA_off])
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

			## snapshots of RF development
			ncol,nrow = 5,5
			fig = plt.figure(figsize=(6*ncol,5*nrow))
			fig.suptitle("Development of RF")
			timesteps = Wlgn_to_4_t.shape[0]
			timesteps_to_plot = np.linspace(1,1*ncol*nrow,ncol*nrow).astype(int)
			print("timesteps",timesteps,timesteps_to_plot,Wlgn_to_4_t.shape)
			for iplot in range(np.min([ncol*nrow,timesteps])):
				istep = timesteps_to_plot[iplot]
				if istep<Wlgn_to_4_t.shape[0]:
					ax = fig.add_subplot(nrow,ncol,1+iplot)
					ax.set_title("Timestep={}".format(1+istep*params["saving_stepsize"]))
					idsf = Wlgn_to_4_t[istep,0,...] - Wlgn_to_4_t[istep,1,...]
					idsf = idsf.reshape(N4,N4*Nvert,Nlgn,Nlgn)
					iRF,_,_,_ = analysis_tools.get_RF_form(idsf,N4,Nlgn,DA,calc_PF=False,\
															Nvert=Nvert,mode="other")
					im=ax.imshow(iRF,interpolation="nearest",cmap="RdBu_r")
					plt.colorbar(im,ax=ax)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			## 3 snapshots of RF development
			t0,t1,t2=timesteps//4,timesteps//2,timesteps-1#timesteps//4
			print("t0,t1,t2",t0,t1,t2,timesteps)
			sf0 = Wlgn_to_4_t[t0,j*2:j*2+2,...]
			sf0 = sf0.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
			RF0,PF0,_,_ = analysis_tools.get_RF_form(sf0,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
			sf1 = Wlgn_to_4_t[t1,j*2:j*2+2,...]
			sf1 = sf1.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
			RF1,PF1,_,_ = analysis_tools.get_RF_form(sf1,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
			ncol,nrow = 3,2
			fig = plt.figure(figsize=(6*ncol,5*nrow))
			fig.suptitle("Development of RF")
			if np.sum(RF[0,:,:]>0)>0:
				vm = np.percentile(RF[0,RF[0,:,:]>0],90)
			else:
				vm = 2
			for k,(kRF,kPF,kt) in enumerate(zip([RF0,RF1,RF],[PF0,PF1,PF],[t0,t1,t2])):
				ax1 = fig.add_subplot(nrow,ncol,1+k)
				ax1.set_title("t={}".format(kt))
				ax2 = fig.add_subplot(nrow,ncol,1+k+ncol)
				m = np.nanmax(kRF[0,:,:])
				im=ax1.imshow(kRF[0,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-m,vmax=m)
				plt.colorbar(im,ax=ax1)
				im=ax2.imshow(kPF[0,:,:],interpolation="nearest",cmap="RdBu_r")
				plt.colorbar(im,ax=ax2)
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			## conservation of ff weights, across x and alpha+i
			ncol,nrow = 5,2
			fig = plt.figure(figsize=(6*ncol,5*nrow))
			fig.suptitle("conservation of ff weights, across x and alpha+i")
			for k,kt in enumerate([t0,t1,t2]):
				mean_x =\
				 np.nanmean(Wlgn_to_4_t[kt,j*2:j*2+2,...].reshape(2,N4**2*Nvert,Nlgn**2),axis=1)
				mean_alpha =\
				 np.nanmean(Wlgn_to_4_t[kt,j*2:j*2+2,...].reshape(2,N4**2*Nvert,Nlgn**2),axis=(0,2))
				ax1 = fig.add_subplot(nrow,ncol,1+k)
				ax1.set_title("t={}".format(kt))
				ax1.plot(np.copy(mean_x.flatten()),'-k')
				ax1.plot(init_norm_x.flatten()[j*2*Nlgn**2:(j+1)*2*Nlgn**2],"--",c="gray")
				ax1.set_xlabel("LGN units")
				ax1.set_ylabel("Average weight")
				ax2 = fig.add_subplot(nrow,ncol,1+k+ncol)
				ax2.plot(np.copy(mean_alpha),'-k',label="t={}".format(kt))
				ax2.plot(init_norm_alpha,"--",c="gray",label="init")
				ax2.legend(loc="best")
				ax2.set_xlabel("Cortical units")
				ax2.set_ylabel("Average weight")
			ax1 = fig.add_subplot(nrow,ncol,ncol-1)
			ax1.set_title("t={}".format(kt))
			im=ax1.imshow(np.sum(np.copy(mean_x).reshape(2,Nlgn,Nlgn),axis=0),\
							interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax1)
			ax2 = fig.add_subplot(nrow,ncol,ncol*2-1)
			im=ax2.imshow(np.copy(mean_alpha).reshape(N4,N4),\
							interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax2)

			mean_x_t =\
			 np.nanmean(Wlgn_to_4_t[:,j*2:j*2+2,...],axis=2)
			mean_alpha_t =\
			 np.nanmean(Wlgn_to_4_t[:,j*2:j*2+2,...],axis=(1,3))
			ax = fig.add_subplot(nrow,ncol,ncol)
			ax.errorbar(np.arange(timesteps),np.nanmean(mean_x_t,axis=(1,2)),\
						yerr=np.nanstd(mean_x_t,axis=(1,2)),fmt="+-")
			ax.set_xlabel("Timesteps")
			ax.set_ylabel("Avg weight summed over ON/OFF and x")
			ax = fig.add_subplot(nrow,ncol,ncol*2)
			ax.errorbar(np.arange(timesteps),np.nanmean(mean_alpha_t,axis=1),\
						yerr=np.nanstd(mean_alpha_t,axis=1),fmt="+-")
			ax.set_xlabel("Timesteps")
			ax.set_ylabel(r"Avg weight summed over ON/OFF and $\alpha$")
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
			## - RF value at center
			rf_center_value = RF_array[:,DA//2,DA//2]
			fig = plt.figure(figsize=(3*6,5))
			ax = fig.add_subplot(131)
			ax.plot(np.sort(dist_on_center),np.linspace(0,1,len(dist_on_center)),\
					"-k",label="on")
			ax.plot(np.sort(dist_off_center),np.linspace(0,1,len(dist_off_center)),\
					"-m",label="off")
			ax.set_xlabel("Subfield distance to RF center")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(132)
			ax.plot(np.sort(dist_on_minus_off),np.linspace(0,1,len(dist_on_minus_off)),\
					"-c",label="on-off")
			ax.set_xlabel("On-Off distance to RF center")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			ax = fig.add_subplot(133)
			ax.plot(np.sort(rf_center_value),np.linspace(0,1,len(rf_center_value)),\
					"-k",label="RF center")
			ax.set_xlabel("Center value of RF")
			ax.set_ylabel("Cumulative distribution")
			ax.legend(loc="best")
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

			# if params["Inp_params"]["off_bias_strength"]>0:
			# 	lgn_sd = np.nanstd(lgn,axis=2).reshape(2,Nlgn,Nlgn)

			# 	minval = np.nanmin(lgn_sd)
			# 	maxval = np.nanmax(lgn_sd)
			# 	lgn_sd_int = ((lgn_sd-minval)/(maxval-minval)*255).astype("uint8")
			# 	otsu_thr,_ = cv2.threshold(lgn_sd_int.flatten(),0,255,cv2.THRESH_OTSU)
			# 	otsu_thr = otsu_thr/255.*(maxval-minval)+minval

			# 	on_dom = lgn_sd[0,:,:] > otsu_thr
			# 	of_dom = lgn_sd[1,:,:] > otsu_thr


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
			## sigma : width of envelope in x direction
			## theta : orientation of gabor in rad
			## Lambda : wavelength of gabor
			## psi : phase of gabor in rad
			# sd = Wlgn_to_4[j*2,...] - Wlgn_to_4[j*2+1,...]
			# sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
			# _,Rn,_,_ = analysis_tools.get_response(sd,DA,Nvert=Nvert)
			# fit_params,fitted_gabor,fit_cost,xmax,ymax = analysis_tools.fit_gabor_to_RF(\
			# 														RF[0,...],DA=DA,\
			# 														Nvert=Nvert,N4=N4,Rn=Rn)
			# fit_params = fit_params.reshape(N4**2*Nvert,-1)
			# ncols = fit_params.shape[-1]
			# reasonable_fits = fit_params[:,0]>1.
			# labels = ["Envelope width","Orientation","Relative phase","# half-cycles",\
			# 			"Log Aspect ratio"]

			# gabor_params = np.empty_like(fit_params)*np.nan
			# num_halfcycles = np.nanmean([xmax,ymax])*4/fit_params[:,3]
			# gabor_params[:,:3] = fit_params[:,:3]
			# gabor_params[:,3] = num_halfcycles
			# gabor_params[:,4] = np.log10(fit_params[:,4])

			# fig = plt.figure(figsize=(6*ncols,2*5))
			# for k in range(ncols):
			# 	ax = fig.add_subplot(2,ncols,k+1)
			# 	ax.hist(gabor_params[reasonable_fits,k])
			# 	ax.set_xlabel(labels[k])
			# 	ax = fig.add_subplot(2,ncols,k+1+ncols)
			# 	im=ax.imshow(gabor_params[:,k].reshape(N4,N4*Nvert),interpolation="nearest",\
			# 				cmap="binary")
			# 	plt.colorbar(im,ax=ax)
			# 	ax.set_xlabel(labels[k])
			# pp.savefig(fig,dpi=300,bbox_inches="tight")
			# plt.close(fig)

			# ARF = 0
			# for k in range(N4):
			# 	for l in range(N4*Nvert):
			# 			ARF += RF[0,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
			# ARF /= N4**2*Nvert
			# fig = plt.figure(figsize=(6*3,5))
			# ax = fig.add_subplot(131)
			# im=ax.imshow(fitted_gabor[0,:,:],interpolation="nearest",cmap="RdBu_r")
			# plt.colorbar(im,ax=ax)
			# ax = fig.add_subplot(132)
			# im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
			# plt.colorbar(im,ax=ax)
			# ax = fig.add_subplot(133)
			# im=ax.imshow(ARF,interpolation="nearest",cmap="RdBu_r")
			# plt.colorbar(im,ax=ax)
			# pp.savefig(fig,dpi=300,bbox_inches="tight")
			# plt.close(fig)
			pp.close()

	## receptive and projection fields for symmetrized weights t=W/sqrt(A)
	## visualization of SD = S_on - S_off
	if not os.path.exists(image_dir_param + "rec_field_additional.pdf"):
		filename_list = ["rec_field_additional","rec_field_additional_I"]
		Wlgn_to_4 = Wlgn_to_4.reshape(num_lgn_paths,N4*N4*Nvert,Nlgn*Nlgn)
		DA = max([DA_on,DA_off])
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
		DA = max([DA_on,DA_off])
		for j in range(num_lgn_paths//2):
			pp = PdfPages(image_dir_param + "{}.pdf".format(filename_list[j]))

			sd = Wlgn_to_4[2*j,...] - Wlgn_to_4[2*j+1,...]
			sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
			RFsd,_,_,_ = analysis_tools.get_RF_form(sd,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,\
													mode="diff_only")
			opm,Rn = analysis_tools.get_response(sd,DA,Nvert=Nvert)
			pref_ori = 0.5*np.angle(opm,deg=False)
			pref_ori = pref_ori - (np.sign(pref_ori)-1)*0.5*np.pi

			sel = np.abs(opm)

			fig = plt.figure(figsize=(6,5))
			ax = fig.add_subplot(111)
			ax.plot(np.sort(sel.flatten()),np.linspace(0,1,N4**2*Nvert),"-")
			ax.set_xlim(0,0.5)
			ax.set_ylim(0,1)
			ax.set_xlabel("Ori Selectivity")
			ax.set_ylabel("Cumulative Dist")
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
			ax.set_title("FT pref ori")
			ft_ori = np.abs(np.fft.fftshift(np.fft.fft2(opm-np.nanmean(opm))))
			im=ax.imshow(ft_ori,interpolation="nearest",cmap="binary")#,vmin=0,vmax=360)
			plt.colorbar(im,ax=ax,orientation="horizontal")
			if Nvert>1:
				for iN in range(N4-1):
					ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
			## =============================================================================

			## === relationship between phase difference and connection weight =============
			# pref_phase = pref_phase.reshape(N4*N4*Nvert)
			# pairwise_phase_diff = analysis_tools.difference_in_phase(pref_phase,pref_phase)
			# W4 = connectivity.Connectivity((N4,N4),(N4,N4),random_seed=random_seed,Nvert=Nvert)
			# pairwise_distance,_ = W4.create_matrix(params["W4to4_params"],"linear")
			# pairs_within_rAdistance = pairwise_distance < (rA_on * 1./N4)
			# nearest_neighbours = pairwise_distance <= (np.sqrt(2.) * 1./N4)
			# lower_tri = np.tri(N4**2*Nvert,N4**2*Nvert,k=-1,dtype=bool)
			# WEE = W4to4[:N4**2*Nvert,:N4**2*Nvert]

		
			# lower_WEE = WEE[lower_tri]
			# lower_phase_diff = pairwise_phase_diff[lower_tri]
			# NN_low_WEE = WEE[np.logical_and(lower_tri,nearest_neighbours)]
			# NN_low_phase_diff = pairwise_phase_diff[np.logical_and(lower_tri,nearest_neighbours)]
			# rA_low_WEE = WEE[np.logical_and(lower_tri,pairs_within_rAdistance)]
			# rA_low_phase_diff = pairwise_phase_diff[np.logical_and(lower_tri,pairs_within_rAdistance)]

			# cc = analysis_tools.correlate(lower_WEE,lower_phase_diff)
			# NN_cc = analysis_tools.correlate(NN_low_WEE,NN_low_phase_diff)
			# rA_cc = analysis_tools.correlate(rA_low_WEE,rA_low_phase_diff)

			# fig = plt.figure(figsize=(3*6,2*5))
			# ax = fig.add_subplot(231)
			# ax.set_title("Corr coeff = {:.2f}".format(cc))
			# ax.plot(lower_WEE,lower_phase_diff,"o",alpha=0.4,rasterized=True)
			# ax.set_xlabel("Rec connectivity weight")
			# ax.set_ylabel("Phase diff (deg)")
			# ax = fig.add_subplot(232)
			# ax.set_title("Corr coeff = {:.2f}".format(NN_cc))
			# ax.plot(NN_low_WEE,NN_low_phase_diff,"o",alpha=0.4,rasterized=True)
			# ax.set_xlabel("Rec connectivity weight NN")
			# ax.set_ylabel("Phase diff NN (deg)")
			# ax = fig.add_subplot(233)
			# ax.set_title("Corr coeff = {:.2f}".format(rA_cc))
			# ax.plot(rA_low_WEE,rA_low_phase_diff,"o",alpha=0.4,rasterized=True)
			# ax.set_xlabel("Rec connectivity weight within rA")
			# ax.set_ylabel("Phase diff within rA (deg)")
			# ax = fig.add_subplot(234)
			# im=ax.imshow(WEE,interpolation="nearest",cmap="binary")
			# plt.colorbar(im,ax=ax)
			# ax.set_title("Rec connectivity weight")
			# ax = fig.add_subplot(235)
			# im=ax.imshow(pairwise_phase_diff,interpolation="nearest",cmap="binary")
			# plt.colorbar(im,ax=ax)
			# ax.set_title("Pairwise Phase diff")
			# pp.savefig(fig,dpi=300,bbox_inches='tight')
			# plt.close(fig)
			## =============================================================================

			# fig = plt.figure(figsize=(12,5))
			# ax = fig.add_subplot(121)
			# ax.set_title("Fitted gabors")
			# im=ax.imshow(gabors,interpolation="nearest",cmap="RdBu_r")
			# plt.colorbar(im,ax=ax)
			# ax = fig.add_subplot(122)
			# ax.set_title("RFs")
			# im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
			# plt.colorbar(im,ax=ax)
			# pp.savefig(fig,dpi=300,bbox_inches='tight')
			# plt.close(fig)

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

			# elif Nvert==1:
			# 	pref_phase = pref_phase.reshape(N4*N4)
			# 	pref_ori = pref_ori.reshape(N4*N4)

			# 	W4 = connectivity.Connectivity((N4,N4),(N4,N4),random_seed=random_seed)
			# 	pairwise_distance,_ = W4.create_matrix(params["W4to4_params"],"linear")
			# 	pairs_within_rAdistance = pairwise_distance < (rA_on * 1./N4)
			# 	pairs_within_rAdistance[np.tri(N4**2,N4**2,k=0,dtype=bool)] = 0
			# 	pairwise_phase_diff = analysis_tools.difference_in_phase(pref_phase,pref_phase)
			# 	pairwise_ori_diff = analysis_tools.difference_in_pref_ori(pref_ori,pref_ori)

			# 	phase_diff_rA = pairwise_phase_diff[pairs_within_rAdistance]
			# 	ori_diff_rA = pairwise_ori_diff[pairs_within_rAdistance]

			# 	not_nearest_neighbours = pairwise_distance > (np.sqrt(2.) * 1./N4)
			# 	not_nearest_neighbours[np.tri(N4**2,N4**2,k=0,dtype=bool)] = 1
			# 	pairwise_phase_diff[not_nearest_neighbours] = np.nan
			# 	pairwise_ori_diff[not_nearest_neighbours] = np.nan

			# 	fig = plt.figure(figsize=(6*2,10))
			# 	ax = fig.add_subplot(221)
			# 	try:
			# 		ax.hist([ori_diff_rA,pairwise_ori_diff[np.isfinite(pairwise_phase_diff)]],\
			# 				color=["b","m"],label=["within rA", "NN"],density=True)
			# 	except:
			# 		ax.hist([ori_diff_rA,pairwise_ori_diff[np.isfinite(pairwise_phase_diff)]],\
			# 				color=["b","m"],label=["within rA", "NN"],normed=True)
			# 	ax.set_xlabel("Pairwise pref ori difference NN")
			# 	ax.legend(loc="best")
			# 	ax = fig.add_subplot(222)
			# 	try:
			# 		ax.hist([phase_diff_rA,pairwise_phase_diff[np.isfinite(pairwise_phase_diff)]],\
			# 				bins=7,color=["b","m"],label=["within rA", "NN"],density=True)
			# 	except:
			# 		ax.hist([phase_diff_rA,pairwise_phase_diff[np.isfinite(pairwise_phase_diff)]],\
			# 				bins=7,color=["b","m"],label=["within rA", "NN"],normed=True)
			# 	ax.set_xlabel("Pairwise phase difference NN")
			# 	ax.legend(loc="best")
			# 	plot_pairw_ori_diff = pairwise_ori_diff[np.logical_not(not_nearest_neighbours)]
			# 	plot_pairw_phase_diff = pairwise_phase_diff[np.logical_not(not_nearest_neighbours)]
			# 	ax = fig.add_subplot(223)
			# 	ax.plot(plot_pairw_ori_diff,plot_pairw_phase_diff,"ok",\
			# 			rasterized=True,alpha=0.4)
			# 	ax.set_xlabel("Pairwise pref ori difference NN")
			# 	ax.set_ylabel("Pairwise phase difference NN")
			# 	ax = fig.add_subplot(224)
			# 	try:
			# 		ax.hist([plot_pairw_phase_diff[plot_pairw_ori_diff<20]],bins=7,color=["orange"],
			# 				label=[r"$\Delta$PO<20deg"],density=True)
			# 	except:
			# 		ax.hist([plot_pairw_phase_diff[plot_pairw_ori_diff<20]],bins=7,color=["orange"],
			# 				label=[r"$\Delta$PO<20deg"],normed=True)
			# 	# ax.axvline(np.nanmedian(plot_pairw_phase_diff[plot_pairw_ori_diff>70]),c="g")
			# 	ax.axvline(np.nanmedian(plot_pairw_phase_diff[plot_pairw_ori_diff<20]),c="orange")
			# 	ax.legend(loc="best")
			# 	ax.set_xlabel("Pairwise phase difference NN")
			# 	ax.set_ylabel("Frequency")
			# 	pp.savefig(fig,dpi=300,bbox_inches='tight')
			# 	plt.close(fig)

			pp.close()


	if False:#not os.path.exists(image_dir_param + "dimensionality.pdf"):
		pp = PdfPages(image_dir_param + "dimensionality.pdf")
		
		actE = l4_t[1::avg_no_inp,...]
		actI = l4I_t[1::avg_no_inp,...]
		num_events = actE.shape[0]
		num_patterns = np.min([100,num_events])
		print("actE",actE.shape)
		print("Compute dimensionality from {} response patterns,\
			   for every successive set of {} patterns.".format(num_events,num_patterns))
		sys.stdout.flush()
		print("CHECK",Wlgn_to_4_t.shape,lgn.shape)
		all_dim,all_dim_lgn,all_dim_inp = [],[],[]
		for i in range(num_events//num_patterns):
			dimE = analysis_tools.calc_dimension(actE[i*num_patterns:(i+1)*num_patterns],\
												inp="patterns",output=0)
			dimI = analysis_tools.calc_dimension(actI[i*num_patterns:(i+1)*num_patterns],\
												inp="patterns",output=0)
			all_dim.append(np.array([dimE,dimI]))

			dimOn = analysis_tools.calc_dimension(lgn[0,:,i*num_patterns:(i+1)*num_patterns].T,\
													inp="patterns",output=0)
			dimOff = analysis_tools.calc_dimension(lgn[1,:,i*num_patterns:(i+1)*num_patterns].T,\
													inp="patterns",output=0)
			all_dim_lgn.append(np.array([dimOn,dimOff]))


			input_on,input_off = [],[]
			for istep in range(i*num_patterns,(i+1)*num_patterns):
				input_on.append(np.dot(Wlgn_to_4_t[istep+1,0,...],lgn[0,:,istep]))
				input_off.append(np.dot(Wlgn_to_4_t[istep+1,1,...],lgn[1,:,istep]))

			print("input_on",np.array(input_on).shape,np.array(input_off).shape)
			dimOn = analysis_tools.calc_dimension(np.array(input_on),inp="patterns",output=0)
			dimOff = analysis_tools.calc_dimension(np.array(input_off),inp="patterns",output=0)
			all_dim_inp.append(np.array([dimOn,dimOff]))
			
		all_dim = np.array(all_dim)
		all_dim_lgn = np.array(all_dim_lgn)
		all_dim_inp = np.array(all_dim_inp)

		ncol,nrow = 3,1
		fig = plt.figure(figsize=(6*ncol,5))
		ax = fig.add_subplot(nrow,ncol,1)
		ax.plot(all_dim[:,0],'-r',label="E")
		ax.plot(all_dim[:,1],'--b',label="I")
		ax.legend(loc="best")
		ax.set_ylabel("Dim")
		ax.set_title("Activity patterns L4")
		ax.set_ylim(bottom=0)
		ax = fig.add_subplot(nrow,ncol,2)
		ax.plot(all_dim_lgn[:,0],'-k',label="On")
		ax.plot(all_dim_lgn[:,1],'--m',label="Off")
		ax.legend(loc="best")
		ax.set_ylim(bottom=0)
		ax.set_ylabel("Dim")
		ax.set_title("Activity patterns LGN")
		ax = fig.add_subplot(nrow,ncol,3)
		ax.plot(all_dim_inp[:,0],'-k',label="On")
		ax.plot(all_dim_inp[:,1],'--m',label="Off")
		ax.legend(loc="best")
		ax.set_ylim(bottom=0)
		ax.set_ylabel("Dim")
		ax.set_title("Input patterns to L4")
		pp.savefig(fig,dpi=300,bbox_inches="tight")
		plt.close(fig)
		pp.close()

	try:
		del yt
		del Wlgn_to_4
		del l4_t
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

	print("end")