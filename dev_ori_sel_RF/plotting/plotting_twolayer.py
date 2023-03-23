import numpy as np
import pickle
import os
import sys
from scipy import linalg

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bettina.modeling.ori_dev_model import data_dir,image_dir,inputs,network,\
connectivity
from bettina.modeling.ori_dev_model.tools import plot_functions,analysis_tools,misc,\
update_params_dict
from bettina.modeling.ori_dev_model.data import data_lee



def circvar(angles,high=2*np.pi,axis=None):
	angles = 1.*angles/high*2*np.pi
	if axis is not None:
		return 1-np.abs(np.nanmean(np.exp(1j*angles),axis=axis))
	else:
		return 1-np.abs(np.nanmean(np.exp(1j*angles)))



def plotting_routines(Version,load_location="local"):
	"""
	plots:
		- orientation tuning (pref ori and sel)
		- fit of gabors to RF

	"""
	## load parameters
	if load_location=="habanero":
		load_path = data_dir + "two_layer/habanero/v{v}/".format(v=Version)
	elif load_location=="aws":
		load_path = "/media/bettina/Seagate Portable Drive/physics/columbia/projects/" + \
					"ori_dev_model/data/two_layer/aws/v{v}/".format(v=Version)
		# load_path = data_dir + "two_layer/aws/v{v}/".format(v=Version)
	else:
		load_path = data_dir + "two_layer/v{v}/".format(v=Version)
	params = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
	update_params_dict.update_params(params)


	sc = params["Wret_to_lgn_params"]["sigma"]
	sr = params["W4to4_params"]["sigma_factor"]
	N4 = params["N4"]
	N23 = params["N23"]
	Nret = params["Nret"]
	Nlgn = params["Nlgn"]
	Nvert = params["Nvert"]
	rA_on = int(params["Wlgn_to4_params"]["r_A_on"] * N4)
	rA_off = int(params["Wlgn_to4_params"]["r_A_off"] * N4)
	rA = np.max([rA_off,rA_on])
	DA = np.min([2*rA + 5,N4])
	T_pd = params["Inp_params"]["pattern_duration"]
	random_seed = params["random_seed"]
	dt = params["dt"]
	num_lgn_paths = params["num_lgn_paths"]
	avg_no_inp = params["Inp_params"]["avg_no_inp"]
	gamma_lgn = params["gamma_lgn"]

	last_timestep = params["runtime"]/dt
	params.update({
					"last_timestep" : last_timestep,
					"RF_mode" : "initialize",
					"system" : "two_layer",
					"Version" : Version
					})
	n = network.Network(Version,params)
	# lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
	# 						same_EI_input=True)

	Wret_to_lgn,_,arbor_on,arbor_off,arbor2,_,W4to4,arbor4to4,\
		W23to23,arbor23,W4to23,arbor4to23,_,W23to4 = n.system

	## create folder
	if load_location=="habanero":
		image_dir_param = image_dir + "two_layer/habanero/v{}/".format(Version)
	elif load_location=="aws":
		image_dir_param = image_dir + "two_layer/aws/v{}/".format(Version)
	else:
		image_dir_param = image_dir + "two_layer/tmp/v{}/".format(Version)
	if not os.path.exists(image_dir_param):
		os.makedirs(image_dir_param)
	
	## load data
	y = np.load(load_path + "y_v{v}.npz".format(v=Version))
	keys = sorted(y.files)
	print("y.files",keys)
	Wlgn_to_4 = y["W"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	l4 = y["l4"][:N4**2*Nvert]
	l4I = y["l4"][N4**2*Nvert:]
	l23 = y["l23"][:N23**2]
	l23I = y["l23"][N23**2:]
	if params["W4to23_params"]["plastic"]:
		print("y[W4to23]",y["W4to23"].shape)
		W4to23 = y["W4to23"].reshape(2*N23**2,2*N4**2*Nvert)
	

	## check regime of system
	I = np.diagflat(np.ones(N23**2*2))
	Ir = np.linalg.inv(I - W4to4)
	Wtmp = np.dot(W4to23,Ir)
	I23 = np.dot(np.linalg.inv(I - W23to23 - np.dot(Wtmp,W23to4)),Wtmp)

	ew,_ = np.linalg.eig(W23to23 + np.dot(Wtmp,W23to4))
	print("orig max ew I23",np.nanmax(np.real(ew)))

	I = np.diagflat(np.ones(N4**2*2*Nvert))
	Ix = np.linalg.inv(I - W23to23)
	I4 = np.linalg.inv(I - W4to4 - np.dot(np.dot(W23to4,Ix),W4to23))

	ew,_ = np.linalg.eig(W4to4 + np.dot(np.dot(W23to4,Ix),W4to23))
	print("orig max ew I4",np.nanmax(np.real(ew)))


	## receptive and projection fields
	## visualization of SD = S_on - S_off
	if True:# not os.path.exists(image_dir_param + "rec_field_final.pdf"):
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
			ax.set_xlabel("Pairwise distance of ON/OFF RF center in LGN space")
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
						yerr=np.nanstd(dist_on_center)/np.sqrt(len(dist_on_center)),fmt="s",\
						c="k",ecolor="k")
			ax.errorbar([1.],[np.nanmean(dist_off_center)],\
						yerr=np.nanstd(dist_off_center)/np.sqrt(len(dist_off_center)),fmt="s",\
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

			"""
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
			print("sd.",sd.shape,Wlgn_to_4.shape,DA,Nvert)
			_,Rn,_,_ = analysis_tools.get_response(sd,DA,Nvert=Nvert)
			print("Rn",Rn.shape,RF.shape)
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
			print('h')
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
			ax.set_title("Interpolated, normalised RFs")
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
			"""
			pp.close()

	## ============= OPM ==========================================
	if True:#not os.path.exists(image_dir_param + "opm.pdf"):
		pp = PdfPages(image_dir_param + "opm.pdf")
		sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
		sd = sd.reshape(N4*N4*Nvert,Nlgn,Nlgn)

		if params["Wlgn_to4_params"].get("ret_scatter",False):
			scatter_ampl = 0.25
			if isinstance(params["Wlgn_to4_params"]["ret_scatter"],float):
				scatter_ampl = params["Wlgn_to4_params"]["ret_scatter"]
			rng = np.random.default_rng(2001)
			x_scat = scatter_ampl*rng.standard_normal(N4**2*Nvert)
			y_scat = scatter_ampl*rng.standard_normal(N4**2*Nvert)
			for ix in range(N4**2*Nvert):
				sd[ix,:,:] = np.roll(np.roll(sd[ix,:,:],shift=int(N4*y_scat[ix]),axis=0),\
								shift=int(N4*x_scat[ix]),axis=1)


		sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
		RFsd,_,_,_ = analysis_tools.get_RF_form(sd,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,\
												mode="diff_only")
		print("RFsd",RFsd.shape)
		opm,Rn = analysis_tools.get_response(sd,DA,Nvert=Nvert)

		pref_phase = np.ones_like(opm,dtype=float)
		# gabors = np.swapaxes(gabors,1,2)
		# gabors = gabors.reshape(DA*N4*2,DA*2*N4*Nvert)
		pref_ori = 0.5*np.angle(opm,deg=True)
		sel = np.abs(opm)
		pref_phase /= 2.*np.pi/360
		
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
		ax.set_title("Spectrum")
		spectrum = np.fft.fftshift(np.fft.fft2(opm-np.nanmean(opm)))
		im=ax.imshow(np.abs(spectrum),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax,orientation="horizontal")
		# if Nvert>1:
		# 	for iN in range(N4-1):
		# 		ax.axvline(Nvert*(iN+1),ls="--",c="k",lw=1)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)


		if Nvert==1:
			pref_phase = pref_phase.reshape(N4*N4)
			pref_ori = pref_ori.reshape(N4*N4)

			W4 = connectivity.Connectivity((N4,N4),(N4,N4),random_seed=random_seed)
			pairwise_distance,_ = W4.create_matrix(params["W4to4_params"],"linear")
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


	# results_dict = {
	# 				"pref_phase"	:	pref_phase,
	# 				"pref_ori"		:	pref_ori,
	# 				"l4"			:	l4,
	# 				"l23"			:	l23,
	# 				"Wlgn_to_4"		:	Wlgn_to_4,
	# 				"RF"			:	RFsd,
	# 				"sel"			:	sel,
	# 				}
	# cluster_name = load_location
	# misc.write_to_hdf5(results_dict,cluster_name,Version,data_dir+"two_layer/")

	return None


def plotting_routines_time_dependent(Version,load_location="local"):
	"""
	plots:
		- connectivity 
		- LGN input + corr
		- activity patterns of LGN input, L4 E/I, L23 E/I
		- Dimensionality of LGN input, L4 E/I, L23 E/I
		- Avg activity of L4 E/I, L23 E/I and dev of RF weights
		- dev of RF
	"""
	## load parameters
	if load_location=="habanero":
		load_path = data_dir + "two_layer/habanero/v{v}/".format(v=Version)
	elif load_location=="aws":
		load_path = data_dir + "two_layer/aws/v{v}/".format(v=Version)
	else:
		load_path = data_dir + "two_layer/v{v}/".format(v=Version)
	params = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
	update_params_dict.update_params(params)


	sc = params["Wret_to_lgn_params"]["sigma"]
	sr = params["W4to4_params"]["sigma_factor"]
	N4 = params["N4"]
	N23 = params["N23"]
	Nret = params["Nret"]
	Nlgn = params["Nlgn"]
	Nvert = params["Nvert"]
	rA_on = int(params["Wlgn_to4_params"]["r_A_on"] * N4)
	rA_off = int(params["Wlgn_to4_params"]["r_A_off"] * N4)
	rA = np.max([rA_off,rA_on])
	DA = np.min([2*rA + 5,N4])
	T_pd = params["Inp_params"]["pattern_duration"]
	random_seed = params["random_seed"]
	dt = params["dt"]
	num_lgn_paths = params["num_lgn_paths"]
	avg_no_inp = params["Inp_params"]["avg_no_inp"]
	gamma_lgn = params["gamma_lgn"]
	if "plastic" not in params["W4to23_params"].keys():
		params["W4to23_params"]["plastic"] = False
		params["W4to23_params"]["arbor_profile"] = "gaussian"
		params["W4to23_params"]["s_noise"] = 0.2
		
	last_timestep = params["runtime"]/dt
	params.update({
					"last_timestep" : last_timestep,
					"RF_mode" : "initialize",
					"system" : "two_layer",
					"Version" : Version
					})


	n = network.Network(Version,params)
	lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
							same_EI_input=True)
	lgn = lgn.reshape(num_lgn_paths,Nlgn**2,-1)
	print("lgn",lgn.shape,N4)
	print("lgn",np.nanmax(lgn[:,:,0]),np.nanmean(lgn[:,:,0]))


	Wret_to_lgn,Wlgn_to_4_init,arbor_on,arbor_off,arbor2,_,W4to4_init,arbor4to4,W23to23_init,\
		arbor23,W4to23_init,arbor4to23,_,W23to4 = n.system

	print("Wlgn_to_4_init",Wlgn_to_4_init.shape)
	Wlgn_to_4_init = Wlgn_to_4_init.reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	init_norm_alpha = np.nanmean(Wlgn_to_4_init,axis=(0,2))
	init_norm_x = np.nanmean(Wlgn_to_4_init,axis=1)

	## create folder
	if load_location=="habanero":
		image_dir_param = image_dir + "two_layer/habanero/tmp/v{}/".format(Version)
	elif load_location=="aws":
		image_dir_param = image_dir + "two_layer/aws/tmp/v{}/".format(Version)
	else:
		image_dir_param = image_dir + "two_layer/tmp/v{}/".format(Version)
	if not os.path.exists(image_dir_param):
		os.makedirs(image_dir_param)
	
	## load data
	yt = np.load(load_path + "yt_v{v}.npz".format(v=Version))
	print("yt.files",yt.files)
	timesteps = yt["Wt"].shape[0]
	print("Wlgn_to_4",yt["Wt"].shape)
	Wlgn_to_4 = yt["Wt"][-1,:].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
	Wlgn_to_4_t = yt["Wt"].reshape(timesteps,num_lgn_paths,N4*N4*Nvert,Nlgn*Nlgn)
	## by default look at excitatory activity
	l4 = yt["l4t"][-1,:N4**2*Nvert]
	l4_t = yt["l4t"][:,:N4**2*Nvert]
	l4I_t = yt["l4t"][:,N4**2*Nvert:]
	l23_t = yt["l23t"][:,:N23**2]
	l23I_t = yt["l23t"][:,N23**2:]
	if params["W4to23_params"]["plasticity_rule"]!="None":
		W4to23_t = yt["W4to23t"].reshape(-1,2*N23**2,2*N4**2*Nvert)
		W4to23 = yt["W4to23t"][-1,...].reshape(2*N23**2,2*N4**2*Nvert)
	else:
		W4to23_t = W4to23_init.reshape(1,2*N23**2,2*N4**2*Nvert)
		W4to23 = W4to23_init
	if params["W4to4_params"]["plasticity_rule"]!="None":
		W4to4_t = yt["W4to4t"].reshape(-1,2*N4**2*Nvert,2*N4**2*Nvert)
		W4to4 = yt["W4to4"]
	else:
		W4to4_t = W4to4_init.reshape(1,2*N4**2*Nvert,2*N4**2*Nvert)
		W4to4 = W4to4_init
	if params["W23_params"]["plasticity_rule"]!="None":
		W23to23_t = yt["W23to23t"].reshape(-1,2*N23**2,2*N23**2)
		W23to23 = yt["W23to23"]
	else:
		W23to23_t = W23to23_init.reshape(1,2*N23**2,2*N23**2)
		W23to23 = W23to23_init

	print("plastic rule",params["W23_params"]["plasticity_rule"])
	print("W4to23_t",W23to23_t.shape,W4to23_t.shape,l4_t.shape,avg_no_inp)
	print("W23to23_t",W23to23_t.size,np.sum(np.isfinite(W23to23_t)))

	## save ff connections and activity of last timestep separately
	# filename = "two_layer/v{v}/y_v{v}.npz".format(v=Version)
	# data_dict = {"W" : yt["Wt"][-1,:],\
	# 			"l4" : yt["l4t"][-1,:],\
	# 			"l23" : yt["l23t"][-1,:],\
	# 			"W4to23" : yt["W4to23t"][-1,:]}
	# misc.save_data(Version, filename, data_dict)


	keys = sorted(yt.files)
	# lgn = lgn.reshape(num_lgn_paths,Nlgn**2,-1,params["Inp_params"]["expanse_time"])
	# lgn = lgn[:,:,::params["saving_stepsize"],:]
	# lgn = lgn[:,:,:Wlgn_to_4_t.shape[0]-1,:]
	# lgn = lgn.reshape(num_lgn_paths,Nlgn**2,-1)
	lgn = lgn[:,:,::params["saving_stepsize"]]

	#################################################################################
	############################# FIGURES ###########################################

	## ================ CONNECTIVITY ================================================
	if True:#not os.path.exists(image_dir_param + "connectivity.pdf"):
		pp = PdfPages(image_dir_param + "connectivity.pdf")
		fig_list = plot_functions.plot_connectivity(W4to4,N4=N4,Nvert=Nvert,\
					output_dict={},Wrec_mode=params["W4to4_params"]["Wrec_mode"])
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()

		pp = PdfPages(image_dir_param + "connectivity_L23.pdf")
		fig_list = plot_functions.plot_connectivity(W23to23,N4=N23,Nvert=1,\
					output_dict={},Wrec_mode=params["W23_params"]["Wrec_mode"])
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

		nrow,ncol = 1,3
		figt = plt.figure(figsize=(6*ncol,5*nrow))
		ax = figt.add_subplot(nrow,ncol,1)
		ax.set_title("x={}, y={} (from)".format(0,0))
		W23to23_EI_t = W23to23_t[:,:N23**2,N23**2:].reshape(W23to23_t.shape[0],N23,N23,-1)
		W23to23_II_t = W23to23_t[:,N23**2:,N23**2:].reshape(W23to23_t.shape[0],N23,N23,-1)
		ax.plot(W23to23_EI_t[:,0,0,::50],"-ok",alpha=0.5)
		ax.plot(W23to23_II_t[:,0,0,::50],"-or",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax = figt.add_subplot(nrow,ncol,2)
		ax.set_title("x={}, y={} (from)".format(5,5))
		ax.plot(W23to23_EI_t[:,5,5,::50],"-ok",alpha=0.5)
		ax.plot(W23to23_II_t[:,5,5,::50],"-or",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax = figt.add_subplot(nrow,ncol,3)
		ax.set_title("x={}, y={} (to)".format(0,0))
		W23to23_EI_t = W23to23_EI_t.reshape(W23to23_t.shape[0],N23**2,-1)
		W23to23_II_t = W23to23_II_t.reshape(W23to23_t.shape[0],N23**2,-1)
		ax.plot(W23to23_EI_t[:,::50,0],"-ok",alpha=0.5)
		ax.plot(W23to23_II_t[:,::50,0],"-or",alpha=0.5)
		ax.set_xlabel("Timesteps")
		pp.savefig(figt,dpi=300,bbox_inches="tight")
		plt.close(figt)

		pp.close()


	## ============= FEEDFORWARD INPUT ======================================
	if True:#not os.path.exists(image_dir_param + "lgn_input.pdf"):
		pp = PdfPages(image_dir_param + "lgn_input.pdf")
		fig_list = plot_functions.plot_LGN_input_corr(lgn,Nlgn=Nlgn)
		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)

		lgn_input = []
		for istep in range(lgn.shape[-1]):
			istep_Wff = istep//params["Inp_params"]["avg_no_inp"]
			if istep_Wff>=Wlgn_to_4_t.shape[0]:
				continue
			dotproduct = gamma_lgn * ( np.dot(Wlgn_to_4_t[istep_Wff,0,...],lgn[0,:,istep]) + \
						 np.dot(Wlgn_to_4_t[istep_Wff,1,...],lgn[1,:,istep]) )
			lgn_input.append(dotproduct)
		lgn_input = np.array(lgn_input)
		fig = plt.figure(figsize=(12,5))
		fig.suptitle("LGN input to L4")
		ax = fig.add_subplot(121)
		im=ax.imshow(lgn_input,interpolation="nearest",cmap="binary",\
						aspect=1.*lgn_input.shape[1]/lgn_input.shape[0])
		ax.set_xlabel("Position")
		ax.set_ylabel("plasticity  steps")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("LGN input last plasticity step")
		im=ax.imshow(lgn_input[-1,:].reshape(N4,N4*Nvert),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches="tight")
		plt.close(fig)

		pp.close()


	## ============= ACTIVITY ===============================================
	if True:#("Wt" in keys and not os.path.exists(image_dir_param + "activity.pdf")):
		pp = PdfPages(image_dir_param + "activity.pdf")
	
		inp_on,inp_of = 0,0
		for it in range(lgn.shape[2]):
		# for it in range(lgn.shape[2]-1,lgn.shape[2]):
			Wtime = it//avg_no_inp + 1
			try:
				inp_on += np.clip(np.dot(Wlgn_to_4_t[Wtime,0,...],lgn[0,:,it]).reshape(N4,N4*Nvert),0,1000)
				inp_of += np.clip(np.dot(Wlgn_to_4_t[Wtime,1,...],lgn[1,:,it]).reshape(N4,N4*Nvert),0,1000)
			except:
				pass

		fig = plt.figure(figsize=(12,5))
		fig.suptitle("Avg Wlgn_to_4_t * lgn")
		ax = fig.add_subplot(121)
		ax.set_title("on")
		im=ax.imshow(inp_on/lgn.shape[2],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("off")
		im=ax.imshow(inp_of/lgn.shape[2],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		
		fig = plt.figure(figsize=(12,5))
		fig.suptitle("avg over no_inp={}".format(avg_no_inp))
		ax = fig.add_subplot(121)
		ax.set_title("SD={:.2f}".format(np.nanstd(np.nanmean(lgn[0,:,-avg_no_inp:],axis=1))))
		im=ax.imshow(np.nanmean(lgn[0,:,-avg_no_inp:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("SD={:.2f}".format(np.nanstd(np.nanmean(lgn[1,:,-avg_no_inp:],axis=1))))
		im=ax.imshow(np.nanmean(lgn[1,:,-avg_no_inp:],axis=1).reshape(Nlgn,Nlgn),\
					interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		
		## LAYER 4 ACTIVITY
		if l4_t is not None:
			l4_t = l4_t.reshape(-1,N4,N4*Nvert)
			n = 10
			fig = plt.figure(figsize=(6*10,2*n//10*5))
			fig.suptitle("l4 E act")
			l4_first_npatterns = l4_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
			l4_last_npatterns = l4_t[-n*avg_no_inp+(l4_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
			l4_last_npatterns = l4_last_npatterns[::-1,...]
			l4_to_show = np.concatenate([l4_first_npatterns,l4_last_npatterns])
			fig,_,_=plot_functions.grid_plot_twolayer(l4_to_show,fig,ncol=10,nrow=2*n//10)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)
			
			if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
				l4I_t = l4I_t.reshape(-1,N4,N4*Nvert)
				n = 5
				fig = plt.figure(figsize=(6*10,2*n//10*5))
				fig.suptitle("l4 I act")
				l4I_first_npatterns = l4I_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
				l4I_last_npatterns = l4I_t[-n*avg_no_inp+(l4I_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
				l4I_last_npatterns = l4I_last_npatterns[::-1,...]
				l4I_to_show = np.concatenate([l4I_first_npatterns,l4I_last_npatterns])
				fig,_,_=plot_functions.grid_plot_twolayer(l4I_to_show,fig,ncol=10,nrow=2*n//10)
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

		## LAYER 23 ACTIVITY
		if l23_t is not None:
			l23_t = l23_t.reshape(-1,N23,N23)
			n = 10
			fig = plt.figure(figsize=(6*10,2*n//10*5))
			fig.suptitle("l23 E act")
			l23_first_npatterns = l23_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
			l23_last_npatterns = l23_t[-n*avg_no_inp+(l23_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
			l23_last_npatterns = l23_last_npatterns[::-1,...]
			l23_to_show = np.concatenate([l23_first_npatterns,l23_last_npatterns])
			fig,_,_=plot_functions.grid_plot_twolayer(l23_to_show,fig,ncol=10,nrow=2*n//10)
			pp.savefig(fig,dpi=300,bbox_inches='tight')
			plt.close(fig)

			if params["Wlgn_to4_params"]["connectivity_type"]=="EI":
				l23I_t = l23I_t.reshape(-1,N4,N4*Nvert)
				n = 5
				fig = plt.figure(figsize=(6*10,2*n//10*5))
				fig.suptitle("l23 I act")
				l23I_first_npatterns = l23I_t[1:(n+1)*avg_no_inp:avg_no_inp,...]
				l23I_last_npatterns = l23I_t[-n*avg_no_inp+(l23I_t.shape[0]-1)%avg_no_inp::avg_no_inp,...]
				l23I_last_npatterns = l23I_last_npatterns[::-1,...]
				l23I_to_show = np.concatenate([l23I_first_npatterns,l23I_last_npatterns])
				fig,_,_=plot_functions.grid_plot_twolayer(l23I_to_show,fig,ncol=10,nrow=2*n//10)
				pp.savefig(fig,dpi=300,bbox_inches='tight')
				plt.close(fig)

		# L4 INPUT TO L23
		labels = ["EL4 input to E"]
		fig = plt.figure(figsize=(25,10))
		fig.suptitle(labels[0])
		n = np.min([l4_t.shape[0],5])
		lgn_first_patterns = []
		for istep in range(n):
			dotproduct = np.dot(W4to23[:N23**2,:N4**2*Nvert],l4_t[1+istep*avg_no_inp,:,:].reshape(N4**2))
			lgn_first_patterns.append(dotproduct)
		lgn_first_patterns = np.array(lgn_first_patterns)
		lgn_first_patterns = lgn_first_patterns.reshape(n,N4,N4*Nvert)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns,fig,ncol=n,nrow=1)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)


		## LGN INPUT
		labels = ["ON LGN activity"]
		fig = plt.figure(figsize=(25,10))
		fig.suptitle(labels[0])
		n = np.min([lgn.shape[2],5])
		lgn_first_patterns = (lgn[0,:,:n].T).reshape(n,N4,N4*Nvert)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns,fig,ncol=n,nrow=1)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		labels = ["OFF LGN activity"]
		fig = plt.figure(figsize=(25,10))
		fig.suptitle(labels[0])
		n = np.min([lgn.shape[2],5])
		lgn_first_patterns = (lgn[1,:,:n].T).reshape(n,N4,N4*Nvert)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns,fig,ncol=n,nrow=1)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		## LGN INPUT CONVOLVED WITH FF CONN
		labels = ["LGN input to E"]
		fig = plt.figure(figsize=(25,10))
		fig.suptitle(labels[0])
		n = np.min([lgn.shape[2],n])
		lgn_first_patterns = []
		for istep in range(n):
			istep_Wff = istep//params["Inp_params"]["avg_no_inp"]
			dotproduct = ( np.dot(Wlgn_to_4_t[istep_Wff,0,...],gamma_lgn * lgn[0,:,istep]) + \
						 np.dot(Wlgn_to_4_t[istep_Wff,1,...],gamma_lgn * lgn[1,:,istep]) )
			lgn_first_patterns.append(dotproduct)
		for istep in range(n):
			istep_Wff = istep//params["Inp_params"]["avg_no_inp"]
			dotproduct = ( np.dot(Wlgn_to_4_t[-n+istep_Wff-1,0,...],gamma_lgn * lgn[0,:,-n+istep]) + \
						 np.dot(Wlgn_to_4_t[-n+istep_Wff-1,1,...],gamma_lgn * lgn[1,:,-n+istep]) )
			lgn_first_patterns.append(dotproduct)
		lgn_first_patterns = np.array(lgn_first_patterns)
		lgn_first_patterns = lgn_first_patterns.reshape(2*n,N4,N4*Nvert)
		fig,_,_=plot_functions.grid_plot_twolayer(lgn_first_patterns,fig,ncol=n,nrow=2)
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

		labels = ["Avg ON LGN input to E","Avg OFF LGN input to E","Avg ON LGN input to I",\
					"Avg OFF LGN input to I"]
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
				im=ax.imshow(np.nanmean(lgn[j+2,:,:],axis=1).reshape(Nlgn,Nlgn),interpolation="nearest",\
							cmap="binary")
				plt.colorbar(im,ax=ax)

			ax = fig.add_subplot(nrow,ncol,ncol*2)
			ax.set_title("lgn on + off avg")
			im=ax.imshow(np.nanmean(lgn[2:,...],axis=(0,2)).reshape(Nlgn,Nlgn),\
						interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		l4_t = l4_t.reshape(-1,N4*N4*Nvert)
		l23_t = l23_t.reshape(-1,N23*N23)
		pp.close()


	## ============= Dimensionality ====================================
	if False:#not os.path.exists(image_dir_param + "dimensionality.pdf"):
		pp = PdfPages(image_dir_param + "dimensionality.pdf")
		
		actEL4 = l4_t[1::avg_no_inp,...]
		actIL4 = l4I_t[1::avg_no_inp,...]
		actEL23 = l23_t[1::avg_no_inp,...]
		actIL23 = l23I_t[1::avg_no_inp,...]
		num_events = actEL4.shape[0]
		num_patterns = np.min([50,num_events])
		print("Compute dimensionality from {} response patterns,\
			  for every successive set of {} patterns.".format(num_events,num_patterns))
		sys.stdout.flush()
		all_dim,all_dim_lgn = [],[]
		for i in range(num_events//num_patterns):
			for j,act in enumerate([actEL4,actIL4,actEL23,actIL23]):
				print("act",j,act.shape)
				dim = analysis_tools.calc_dimension(act[i*num_patterns:(i+1)*num_patterns],\
													inp="patterns",output=0)
				all_dim.append(dim)

			dimOn = analysis_tools.calc_dimension(lgn[0,:,i*num_patterns:(i+1)*num_patterns].T,\
													inp="patterns",output=0)
			dimOff = analysis_tools.calc_dimension(lgn[1,:,i*num_patterns:(i+1)*num_patterns].T,\
													inp="patterns",output=0)
			all_dim_lgn.append(np.array([dimOn,dimOff]))
		all_dim = np.array(all_dim).reshape(-1,4)
		all_dim_lgn = np.array(all_dim_lgn)
		
		fig = plt.figure(figsize=(12,5))
		ax = fig.add_subplot(121)
		ax.plot(all_dim[:,0],'-or',label="L4 E")
		ax.plot(all_dim[:,1],'--sb',label="L4 I")
		ax.plot(all_dim[:,2],'-om',label="L23 E")
		ax.plot(all_dim[:,3],'--sc',label="L23 I")
		ax.legend(loc="best")
		ax.set_ylabel("Dim")
		ax.set_ylim(bottom=0)
		ax = fig.add_subplot(122)
		ax.plot(all_dim_lgn[:,0],'-k',label="On")
		ax.plot(all_dim_lgn[:,1],'--m',label="Off")
		ax.legend(loc="best")
		ax.set_ylim(bottom=0)
		ax.set_ylabel("Dim")
		pp.savefig(fig,dpi=300,bbox_inches="tight")
		plt.close(fig)
		pp.close()


	## ============= DYNAMICS FF WEIGHTS ====================================
	if True:#not os.path.exists(image_dir_param + "dyn_test.pdf"):
		xcoords = [[6,5],[3,7],[0,0]]
		xcoords = [[2,1],[3,1],[0,0]]
		pp = PdfPages(image_dir_param + "dyn_test.pdf")
		ncol,nrow = 4,2
		figt = plt.figure(figsize=(6*ncol,5*nrow))
		labels = ["L4","L23"]
		Ntot_x = [N4*Nvert,N23]
		Ntot_y = [N4,N23]
		for i,act in enumerate([l4_t,l23_t]):
			ax = figt.add_subplot(nrow,ncol,1+i*2)
			im=ax.imshow(act,interpolation="nearest",cmap="binary",\
						 aspect=1.*act.shape[1]/act.shape[0])
			plt.colorbar(im,ax=ax)
			ax.set_ylabel("Timesteps")
			ax.set_xlabel("Act {}".format(labels[i]))
			ax = figt.add_subplot(nrow,ncol,2+i*2)
			ax.plot(np.nanmean(act[1:,:],axis=1),'-ok',label="mean {}".format(labels[i]))
			act = act.reshape(-1,Ntot_y[i],Ntot_x[i])
			for j in range(3):
				ax.plot(act[1:,xcoords[j][1],xcoords[j][0]],'-',\
						label="{},{}".format(labels[i],j+1),alpha=0.5)
			ax.set_ylabel("mean, indiv activity {}".format(labels[i]))
			ax.set_xlabel("Timesteps")
			ax.legend(loc="best")
			act = act.reshape(-1,Ntot_y[i]*Ntot_x[i])

		ax = figt.add_subplot(nrow,ncol,ncol+1)
		ax.set_title("x={}, y={}".format(xcoords[0][0],xcoords[0][1]))
		Wlgn_to_4_t = Wlgn_to_4_t.reshape(Wlgn_to_4_t.shape[0],num_lgn_paths,N4,N4*Nvert,-1)
		ax.plot(Wlgn_to_4_t[:,0,xcoords[0][1],xcoords[0][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[0][1],xcoords[0][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")
		ax = figt.add_subplot(nrow,ncol,ncol+2)
		ax.set_title("x={}, y={}".format(xcoords[1][0],xcoords[1][1]))
		ax.plot(Wlgn_to_4_t[:,0,xcoords[1][1],xcoords[1][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[1][1],xcoords[1][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")
		ax = figt.add_subplot(nrow,ncol,ncol+3)
		ax.set_title("x={}, y={}".format(xcoords[2][0],xcoords[2][1]))
		ax.plot(Wlgn_to_4_t[:,0,xcoords[2][1],xcoords[2][0],:],"-k",alpha=0.5)
		ax.plot(Wlgn_to_4_t[:,1,xcoords[2][1],xcoords[2][0],:],"--r",alpha=0.5)
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Won,off (k=on,r=off)")

		Wlgn_to_4_t = Wlgn_to_4_t.reshape(Wlgn_to_4_t.shape[0],num_lgn_paths,N4,N4*Nvert,-1)
		arbor2 = arbor2.reshape(num_lgn_paths,N4,N4*Nvert,-1)
		WE = Wlgn_to_4_t[:,:2,...]
		Wlim = params["Wlgn_to4_params"]["Wlim"]
		frozen = np.logical_or(WE[:,arbor2[:2,...]>0]<=0,WE[:,arbor2[:2,...]>0]>=Wlim)
		ax = figt.add_subplot(nrow,ncol,ncol+4)
		ax.plot(np.nanmean(frozen,axis=1), '-k')
		ax.set_xlabel("Timesteps")
		ax.set_ylabel("Fraction of frozen weights")
		ax.set_ylim(0,1)

		pp.savefig(figt,dpi=300,bbox_inches='tight')
		plt.close(figt)

		pp.close()
		arbor2 = arbor2.reshape(num_lgn_paths,N4**2*Nvert,-1)
		Wlgn_to_4_t = Wlgn_to_4_t.reshape(Wlgn_to_4_t.shape[0],num_lgn_paths,N4*N4*Nvert,-1)
		print("dynamics done")

	## ============= CHANGE FF WEIGHTS ======================================
	if False:
		pp = PdfPages(image_dir_param + "change_Weights.pdf")
		fig = plt.figure(figsize=(24,5))
		deltaW = Wlgn_to_4_t[:-1,...] - Wlgn_to_4_t[1:,...]
		print("Wlgn_to_4_t",Wlgn_to_4_t.shape,deltaW.shape)
		avg_deltaW_t = np.nanmean(deltaW[0:10,...],axis=(0))
		avg_deltaW_xa = np.nanmean(deltaW,axis=(1,2,3))
		sf = avg_deltaW_t.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		sf = sf[:2,...]
		for k in range(2):
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
		ax.plot(avg_deltaW_xa,"-k")
		ax.set_ylabel("avg change")
		ax.set_ylabel("timesteps")
		# ax.set_xlim(0,100)
		ax = fig.add_subplot(144)
		ax.plot(np.nanmean(np.abs(deltaW[:,0,:,:]),axis=(1,2)),"-ok")
		ax.plot(np.nanmean(np.abs(deltaW[:,1,:,:]),axis=(1,2)),"-om")
		ax.set_ylabel("avg abs change in W_on/off")
		ax.set_ylabel("timesteps")
		# ax.set_xlim(0,100)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		nrow,ncol = 5,5
		fig = plt.figure(figsize=(50,50))
		for irow in range(nrow):
			for jcol in range(ncol):
					if deltaW.shape[0]>(jcol+irow*ncol):
						sf = deltaW[jcol+irow*ncol,:,:,:].reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
						delta_RF = np.zeros((DA*N4,DA*N4*Nvert))
						for i in range(N4*Nvert):
							for j in range(N4):
								son_ij = np.roll(np.roll(sf[0,j,i,:,:]-0*sf[1,j,i,:,:],shift=Nlgn//2-j,axis=0),\
									shift=Nlgn//2-i,axis=1)
								delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
								 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
						ax = fig.add_subplot(nrow,ncol,jcol+irow*ncol+1)
						ax.set_title("t={}".format(jcol+irow*ncol))
						im=ax.imshow(delta_RF,interpolation="nearest",cmap="RdBu_r")
						plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		## plot dvelopment for two chosen RFs at cortical locations: 
		xcoord1 = [6,5] #selective for 360 and 332
		xcoord2 = [3,7] #unselective/off for 360, sel for 332
		xcoords = [[6,5],[3,7],[0,0]]
		xcoords = [[2,1],[3,1],[0,0]]
		ncoords = 3
		timesteps = np.min([1000,deltaW.shape[0]])#deltaW.shape[0]
		# for timestep in range(timesteps-600,timesteps,20):
		# 	fig = plt.figure(figsize=(6*ncoords,5))
		# 	sf = deltaW[timestep,:,:,:].reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
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

		fig = plt.figure(figsize=(12,5))
		fig.suptitle("Avg change")
		sf = np.nanmean(deltaW,axis=0).reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		for jc in range(ncoords):
			ax1 = fig.add_subplot(1,ncoords,1+jc)
			dsf = sf[0,xcoords[jc][1],xcoords[jc][0],:,:]-sf[1,xcoords[jc][1],\
			 		xcoords[jc][0],:,:]
			dsf = np.roll(np.roll(dsf,shift=Nlgn//2-int(xcoords[jc][1]/N4*Nlgn),axis=0),\
					shift=Nlgn//2-int(xcoords[jc][0]/N4/Nvert*Nlgn),axis=1)
			im=ax1.imshow(dsf,interpolation="nearest",cmap="RdBu_r",\
						  vmin=-0.015,vmax=0.015)
			plt.colorbar(im,ax=ax1)

		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)
		pp.close()
		print("change done")

	## ============= RF and PF ==============================================
	if True:#not os.path.exists(image_dir_param + "rec_field.pdf"):
		pp = PdfPages(image_dir_param + "rec_field.pdf")
		## receptive and projection fields
		## visualization of SD = S_on - S_off
		sf = Wlgn_to_4_t[-1,...]
		sf = sf.reshape(num_lgn_paths,N4*N4*Nvert,Nlgn,Nlgn)

		if params["Wlgn_to4_params"].get("ret_scatter",False):
			scatter_ampl = 0.25
			if isinstance(params["Wlgn_to4_params"]["ret_scatter"],float):
				scatter_ampl = params["Wlgn_to4_params"]["ret_scatter"]
			rng = np.random.default_rng(2001)
			x_scat = scatter_ampl*rng.standard_normal(N4**2*Nvert)
			y_scat = scatter_ampl*rng.standard_normal(N4**2*Nvert)
			print("retscatter",sf.shape,DA,DA*3,N4)

			print("SHIFT 0",N4*y_scat[0],N4*x_scat[0])
			print("SHIFT 1",N4*y_scat[1],N4*x_scat[1])
			for ix in range(N4**2*Nvert):
				sf[:,ix,:,:] = np.roll(np.roll(sf[:,ix,:,:],shift=int(N4*y_scat[ix]),axis=1),shift=int(N4*x_scat[ix]),axis=2)

		sf = sf.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		RF,PF,_,_ = analysis_tools.get_RF_form(sf[:2,...],N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
		fig = plot_functions.plot_RF(RF,PF,N4=N4,Nvert=Nvert,DA=DA)
		fig.suptitle("E RF")
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		RF,PF,_,_ = analysis_tools.get_RF_form(sf[2:,...],N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
		fig = plot_functions.plot_RF(RF,PF,N4=N4,Nvert=Nvert,DA=DA)
		fig.suptitle("I RF")
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		## receptive and projection fields for symmetrized weights t=W/sqrt(A)
		## visualization of SD = S_on - S_off
		tf = Wlgn_to_4_t[-1,...]/np.sqrt(arbor2)
		tf[np.logical_not(arbor2>0)] = 0.0
		tf = sf.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
		RF,PF,_,_ = analysis_tools.get_RF_form(tf[:2,...],N4,Nlgn,DA,calc_PF=True,Nvert=Nvert)
		fig = plot_functions.plot_RF(RF,PF,N4=N4,Nvert=Nvert,DA=DA)
		fig.suptitle("symmetrized weights")
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		# development of RF, 25 snapshots
		ncol,nrow = 5,5
		fig = plt.figure(figsize=(6*ncol,5*nrow))
		fig.suptitle("Development of RF")
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


		## conservation of ff weights, across x and alpha+i
		t0,t1,t2=1,timesteps//2,timesteps-1
		ncol,nrow = 5,2
		fig = plt.figure(figsize=(6*ncol,5*nrow))
		fig.suptitle("conservation of ff weights, across x and alpha+i")
		for k,kt in enumerate([t0,t1,t2]):
			mean_x =\
			 np.nanmean(Wlgn_to_4_t[kt,:2,...].reshape(2,N4**2*Nvert,Nlgn**2),axis=1)
			mean_alpha =\
			 np.nanmean(Wlgn_to_4_t[kt,:2,...].reshape(2,N4**2*Nvert,Nlgn**2),axis=(0,2))
			ax1 = fig.add_subplot(nrow,ncol,1+k)
			ax1.set_title("t={}".format(kt))
			ax1.plot(np.copy(mean_x.flatten()),'-k')
			ax1.plot(init_norm_x.flatten()[:2*Nlgn**2],"--",c="gray",zorder=10)
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
		im=ax2.imshow(np.copy(mean_alpha).reshape(N4,N4),interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax2)

		mean_x_t =\
		 np.nanmean(Wlgn_to_4_t[:,:2,...],axis=2)
		mean_alpha_t =\
		 np.nanmean(Wlgn_to_4_t[:,:2,...],axis=(1,3))
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

		fig = plt.figure()
		ax = fig.add_subplot(221)
		ax.set_title("final on conn")
		im=ax.imshow(Wlgn_to_4_t[-1,0,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(222)
		ax.set_title("final off conn")
		im=ax.imshow(Wlgn_to_4_t[-1,1,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(223)
		ax.set_title("initial on conn")
		im=ax.imshow(Wlgn_to_4_t[0,0,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(224)
		ax.set_title("initial off conn")
		im=ax.imshow(Wlgn_to_4_t[0,1,:,:],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches="tight")
		plt.close(fig)

		pp.close()


	if False:#(not os.path.exists(image_dir_param + "W4to23_dev.pdf")):
		pp = PdfPages(image_dir_param + "W4to23_dev.pdf")
		print("W4to23",W4to23_t.shape,W4to23.shape,N4,N23)
		W4to23 = W4to23.reshape(N23**2*2,N4**2*2)
		RF4to23_EtoE,_,_,_ = analysis_tools.get_RF_form(W4to23[:N23**2,:N4**2],N23,N4,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")
		RF4to23_EtoI,_,_,_ = analysis_tools.get_RF_form(W4to23[N23**2:,:N4**2],N23,N4,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")

		fig = plt.figure(figsize=(12,5))
		fig.suptitle("Final W4to23")
		ax = fig.add_subplot(121)
		ax.set_title("RF4to23_EtoE")
		im=ax.imshow(RF4to23_EtoE,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("RF4to23_EtoI")
		im=ax.imshow(RF4to23_EtoI,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		fig_list = plot_functions.plot_dev_connectivity(W4to23_t,\
					plastic=params["W4to23_params"]["plasticity_rule"]!="None")

		for fig in fig_list:
			pp.savefig(fig,dpi=300,bbox_inches="tight")
			plt.close(fig)
		pp.close()

		pp = PdfPages(image_dir_param + "W23to23_final.pdf")
		fig = plt.figure(figsize=(12,5))
		ax = fig.add_subplot(121)
		ax.set_title("init")
		im=ax.imshow(W23to23_init,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("final")
		im=ax.imshow(W23to23,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		W23to23 = W23to23.reshape(N23**2*2,N23**2*2)
		RF23to23_EtoE,_,_,_ = analysis_tools.get_RF_form(W23to23[:N23**2,:N23**2],N23,N23,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")
		RF23to23_EtoI,_,_,_ = analysis_tools.get_RF_form(W23to23[N23**2:,:N23**2],N23,N23,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")
		RF23to23_ItoE,_,_,_ = analysis_tools.get_RF_form(W23to23[:N23**2,N23**2:],N23,N23,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")
		RF23to23_ItoI,_,_,_ = analysis_tools.get_RF_form(W23to23[N23**2:,N23**2:],N23,N23,\
								int(N23*0.4+5),calc_PF=False,Nvert=1,mode="other")

		fig = plt.figure(figsize=(24,5))
		fig.suptitle("Final W4to23")
		ax = fig.add_subplot(141)
		ax.set_title("RF23to23_EtoE")
		im=ax.imshow(RF23to23_EtoE,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(142)
		ax.set_title("RF23to23_EtoI")
		im=ax.imshow(RF23to23_EtoI,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		fig.suptitle("Final W4to23")
		ax = fig.add_subplot(143)
		ax.set_title("RF23to23_ItoE")
		im=ax.imshow(RF23to23_ItoE,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(144)
		ax.set_title("RF23to23_ItoI")
		im=ax.imshow(RF23to23_ItoI,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		pp.close()

	## ================ FIND PHASE PREFERENCE =======================================
	if False:
		pp = PdfPages(image_dir_param + "phase_response_props.pdf")
		num_phases = 2
		spat_freq = np.array([np.sqrt(0.5* 1./0.2)])
		orientations = np.linspace(0,np.pi,2)
		stimuli = analysis_tools.generate_grating_stimuli((Nlgn,Nlgn),num_phases,\
					spat_freq,orientations)
		responses = analysis_tools.stimulate_system(stimuli,params,Wlgn_to_4,\
					W4to4=W4to4,W23to4=W23to4,W23to23=W23to23,W4to23=W4to23)
		## take responses for preferred spat frequency here
		responses = (responses[0][:,0,:,:],responses[1][:,0,:,:])
		resp_l4_phases,resp_l23_phases,sgl_cell_props_dict =\
		 analysis_tools.find_phase_sensitivity(responses)
		num_phases = resp_l4_phases.shape[0]


		fig = plt.figure(figsize=(12,5))
		ax = fig.add_subplot(121)
		ax.set_title("Phase response L4")
		im=ax.imshow(resp_l4_phases,interpolation="nearest",cmap="binary",\
			aspect=1.*resp_l4_phases.shape[1]/resp_l4_phases.shape[0])
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(122)
		ax.set_title("Phase response L23")
		im=ax.imshow(resp_l23_phases,interpolation="nearest",cmap="binary",\
			aspect=1.*resp_l23_phases.shape[1]/resp_l23_phases.shape[0])
		plt.colorbar(im,ax=ax)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		fig = plt.figure(figsize=(12,5))
		ax = fig.add_subplot(121)
		ax.set_title("Modulation ratio L4")
		ax.hist(sgl_cell_props_dict["modulation_ratio_l4"])
		ax = fig.add_subplot(122)
		ax.set_title("Modulation ratio L23")
		ax.hist(sgl_cell_props_dict["modulation_ratio_l23"])
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		fig = plt.figure(figsize=(12,5))
		ax = fig.add_subplot(121)
		ax.set_title("Phase response L4")
		for unit in range(resp_l4_phases.shape[1]):
			this_resp = resp_l4_phases[:,unit]
			this_resp = np.roll(this_resp,np.argmax(this_resp))
			ax.plot(this_resp,"-k",alpha=0.4)
		ax = fig.add_subplot(122)
		ax.set_title("Phase response L23")
		for unit in range(resp_l23_phases.shape[1]):
			this_resp = resp_l23_phases[:,unit]
			this_resp = np.roll(this_resp,np.argmax(this_resp))
			ax.plot(this_resp,"-k",alpha=0.4)
		pp.savefig(fig,dpi=300,bbox_inches='tight')
		plt.close(fig)

		pp.close()
		print("phase resp props done")


	return None




if __name__=="__main__":
	import argparse


	current_user = os.environ["USER"]
	Version_default = misc.get_version(data_dir + "layer4/",version=None) - 1

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Plot results from simulation.")
	parser.add_argument("--index", dest='idx', help="specifying run number (either\
						 individual runs as x1 x2 x3 or endpoints of successive numbers of\
						 runs as x1:xn)", required=False,  nargs="*", \
						 default=Version_default)
	parser.add_argument("--cl", dest='load_location', help="from which folder to\
						load the files (can be either local, habanero, aws)",\
					 	required=False, default="local", type=str)
	args = parser.parse_args()
	args_dict = vars(args)

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
			

	load_location = args_dict["load_location"]
	for Version in np.unique(Versions):
		print("Plotting data from run {}".format(Version))
		try:
			plotting_routines(Version,load_location=load_location)
			if os.environ["USER"]!="bettina":
				print("")
				print("Plotting time dependent data from run {}".format(Version))
			plotting_routines_time_dependent(Version,load_location=load_location)
		except:
			misc.PrintException()
			print("Ignoring version {}".format(Version))
			pass


	print("done")