"""
get comparable spatial scales of LGN input correlations and cortical actiivyt between correlation
and activity based model
"""
import numpy as np

from bettina.modeling.ori_dev_model import inputs,connectivity
from bettina.modeling.ori_dev_model.tools import analysis_tools

def determine_LGN_correlation_scale(sigma,rA,*kwargs):
	if "Wret_to_lgn_params" in kwargs.keys:
		Wret_to_lgn_params = kwargs["Wret_to_lgn_params"]
	if "Inp_params" in kwargs.keys:
		Inp_params = kwargs["Inp_params"]
		
	Wret_to_lgn_params["sigma"] = sigma*rA
	N = 25
	random_seed = 19
	Inp_params.update({"Nsur" : 200})
	Wret_to_lgn = connectivity.Connectivity((N,N),(N,N),\
				  random_seed=random_seed).create_matrix(\
				  Wret_to_lgn_params,"Gaussian") * Wret_to_lgn_params["gamma_ret"]
	lgn = []
	for i in range(Inp_params["Nsur"]):
		# rng_seed = Version*1000 + i
		rng_seed = random_seed*1000 + i
		one_stimulus = inputs.Inputs_lgn((N,N),1,rng_seed).create_lgn_input(\
						Inp_params, "white_noise_online", Wret_to_lgn)
		lgn.append( one_stimulus )
	lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
	num_lgn_path,dim,num_patterns = lgn.shape
	lgn = lgn.reshape(num_lgn_path,N,N,num_patterns)

	fit_params,fitted_gauss,avg_cc_diff =\
		 analysis_tools.determine_LGN_input_correlations_scale(lgn)
	fitted_gauss = fitted_gauss.reshape(N,N)
	return fit_params,fitted_gauss,avg_cc_diff

def determine_rec_activity_scale(sigma_factor):
	pass

if __name__=="__main__":
	from itertools import product
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages

	from bettina.modeling.ori_dev_model import Inp_params,Wret_to_lgn_params,image_dir
	
	
	pp = PdfPages(image_dir + "layer4/wavelength/LGN_spatial_scale.pdf")
	sigmas = np.arange(0.1,0.96,0.1)
	radii = [0.1,0.15,0.2,0.25]
	var_key_string = "rA/{}/sigma/{}"
	all_fit_params,all_fitted_gauss = {},{}
	for isigma,iradius in product(sigmas,radii):
		print("isigma,iradius",isigma,iradius)
		fit_params,fitted_gauss,avg_cc_diff = determine_LGN_correlation_scale(isigma,iradius)
		
		all_fitted_gauss[var_key_string.format(iradius,isigma)] = fitted_gauss
		all_fit_params[var_key_string.format(iradius,isigma)] = fit_params

		# results_dict = {"LGNcorr_wavelength" : np.nanmedian(distribution_dict["bessel"])}
		# misc.write_to_hdf5(results_dict,cluster_name,version)
	
		N = 25
		fig = plt.figure(figsize=(18,5))
		fig.suptitle("Arbor radius={:.2f}, SD={:.2f}".format(iradius,isigma))
		ax = fig.add_subplot(131)
		im=ax.imshow(avg_cc_diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
		plt.colorbar(im,ax=ax)
		ax.contour(fitted_gauss,5,colors="b")
		ax = fig.add_subplot(132)
		im=ax.imshow(fitted_gauss,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
		plt.colorbar(im,ax=ax)
		ax.contour(avg_cc_diff,5,colors="b")
		ax = fig.add_subplot(133)
		ax.plot(avg_cc_diff[N//2,:]/np.nanmax(avg_cc_diff),'b',label="Data")
		ax.plot(fitted_gauss[N//2,:],'r',label="Fit")
		ax.plot(avg_cc_diff[:,N//2]/np.nanmax(avg_cc_diff),'b--',label="Data")
		ax.plot(fitted_gauss[:,N//2],'r--',label="Fit")
		ax.legend(loc="best")
		pp.savefig(dpi=300,bbox_inches="tight")
		plt.close(fig)

	cmap = plt.get_cmap('plasma')
	cNorm = matplotlib.colors.Normalize(vmin=0,vmax=np.max([len(radii),len(sigmas)]))
	cmap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
	cmap.set_array([])

	fig = plt.figure(figsize=(12,5))
	ax1 = fig.add_subplot(121)
	ax1.set_xlabel("SD Retina to LGN connectivity")
	ax1.set_ylabel("SD LGN input correlation")
	ax1.set_ylim(bottom=0)
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel("Arbor radius")
	ax2.set_ylabel("SD LGN input correlation")
	ax2.set_ylim(bottom=0)
	for isigma,iradius in product(sigmas,radii):
		print("i",isigma,iradius,all_fit_params[var_key_string.format(iradius,isigma)])
		col = cmap.to_rgba(np.searchsorted(radii,iradius,side="left"))
		ax1.plot(isigma,all_fit_params[var_key_string.format(iradius,isigma)][0],"o",c=col)
		col = cmap.to_rgba(np.searchsorted(sigmas,isigma,side="left"))
		ax2.plot(iradius,all_fit_params[var_key_string.format(iradius,isigma)][0],"o",c=col)
	pp.savefig(dpi=300,bbox_inches="tight")
	plt.close(fig)

	pp.close()

	
