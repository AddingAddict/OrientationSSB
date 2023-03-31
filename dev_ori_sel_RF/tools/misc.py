import numpy as np
import pickle
import h5py
import os
import re
import sys
import linecache
from copy import copy
import yaml

from dev_ori_sel_RF import data_dir,normalisation_constraints as nc


def ensure_path(path):
	if not os.path.exists(path):
		os.makedirs(path)

def get_version(save_path,version=None,readonly=True):
	if version is None:
		listfiles = os.listdir(save_path)
		if len(listfiles)>0:
			matched_names = np.array(listfiles)[np.array(["v" in string for string in listfiles])]
			already_present_versions = []
			for item in matched_names:
				if re.match(r'v\d+', item):
					already_present_versions.append( int( re.findall(r'\d+', item)[-1] ) )
			if len(already_present_versions)>0:
				version = max(already_present_versions)+1
			else:
				version = 0
		else:
			version = 0
	else:
		version = version

	if not readonly:
		os.makedirs(save_path + "v{}".format(version))
	return version

def load_external_params(filename):
	""" load file with parameter settings
	input:
	filename: name of file
	"""
	current_dir = os.getcwd()
	if os.environ["USER"]=="bettina":
		file_path = os.path.join(current_dir,"bettina/modeling/ori_dev_model/data",filename+".yaml")
	elif os.environ["USER"]=="bh2757":
		file_path = os.path.join(current_dir,"code/bettina/modeling/ori_dev_model/data",filename+".yaml")
	elif os.environ["USER"]=="tuannguyen":
		file_path = os.path.join(current_dir,"../dev_ori_sel_RF/data",filename+".yaml")
	else:
		raise Exception("User not found. Can't load parameter file.")

	with open(file_path,"r") as file:
		params_dict = yaml.safe_load(file)
	print("MISC params_dict",filename,params_dict)
	return params_dict

def save_data(Version,filename,data_dict):
	# if "cct" in data.keys():
	# 	np.savez(data_dir + filename,Wt=data["Wt"],\
	# 	# lgn_inp=data["lgn_inp"],
	# 	 cct=data["cct"], l4t=data["l4t"])
	# elif "l23t" in data.keys():
	# 	np.savez(data_dir + filename,Wt=data["Wt"],\
	# 	 l23t=data["l23t"], l4t=data["l4t"])
	# else:
	# 	np.savez(data_dir + filename,Wt=data["Wt"])
	np.savez(data_dir + filename,**data_dict)
	

def save_params(Version,filename,configs):
	file_obj = open(data_dir + filename + ".txt",'w')
	keys = sorted(configs.keys(),key=str.lower)
	for ikey in keys:
		item = configs[ikey]
		if isinstance(item,dict):
			file_obj.write(ikey + "\n")
			item_keys = sorted(item.keys(),key=str.lower)
			for jkey in item_keys:
				file_obj.write("\t" + jkey + "\t" + str(item[jkey]) + "\n")
		else:
			file_obj.write(ikey + "\t" + str(configs[ikey]) + "\n")

	pickle.dump(configs, open(data_dir + filename + ".p","wb"))


def write_to_hdf5(results_dict,cluster_name,version,filename):
	# filename = filepath + "results.hdf5"
	f = h5py.File(filename,'a')
	var_key_string = "cluster/{cl}/version/{v}/".format(cl=cluster_name,v=version)
	for key,value in results_dict.items():
		if (var_key_string in f.keys() and key in f[var_key_string].keys()):
			del f[var_key_string][key]
			f[var_key_string][key] = value
		else:
			f.create_dataset(var_key_string + "/" + key, data=value)
	f.close()
	print("Results written to {}".format(filename))


def load_from_hdf5(cluster_name,version_list,filename=data_dir + "layer4/results.hdf5"):
	f = h5py.File(filename,'r')
	results = {}
	for version in version_list:
		var_key_string = "cluster/{cl}/version/{v}/".format(cl=cluster_name,v=version)
		results[version] = {}
		for key in f[var_key_string].keys():
			results[version][key] = f[var_key_string][key][()]
		# results[version] = copy(f[var_key_string])
	f.close()
	return results


def load_data(Version,file_dir,params):
	N4 = params["N4"]
	Nlgn = params["Nlgn"]
	try:
		Nvert = params["Nvert"]
	except:
		Nvert = 1
	try:
		num_lgn_paths = params["num_lgn_paths"]
	except:
		num_lgn_paths = 2
	try:
		## TEMPORAL DATA
		with np.load(file_dir + "yt_v{v}.npz".format(v=Version)) as yt:
			timesteps = yt["Wt"].shape[0]
			Wlgn_to_4 = yt["Wt"][-1,:].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
			## by default look at excitatory activity
			if "l4t" in yt.files:
				l4 = yt["l4t"][-2,:N4**2*Nvert]
				l4_t = yt["l4t"][:,:N4**2*Nvert]
				l4I_t = yt["l4t"][:,N4**2*Nvert:]			
			else:
				l4,l4_t,l4I_t = None,None,None
			print("yt[]",num_lgn_paths,N4,Nvert,Nlgn)
			Wlgn_to_4_t = yt["Wt"].reshape(timesteps,num_lgn_paths,N4*N4*Nvert,Nlgn*Nlgn)
	except:
		## STATIC DATA
		PrintException()
		yt = np.load(file_dir + "y_v{v}.npz".format(v=Version))
		timesteps = 1
		if "Wlgn_to_4" in yt.files:
			Wlgn_to_4 = yt["Wlgn_to_4"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
		else:
			Wlgn_to_4 = yt["W"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
		## by default look at excitatory activity
		if "l4" in yt.files:
			l4 = yt["l4"][:N4**2*Nvert]
		else:
			l4 = None
		Wlgn_to_4_t = None
		l4_t,l4I_t = None,None

	keys = sorted(yt.files)
	return Wlgn_to_4_t,Wlgn_to_4,l4_t,l4I_t,l4,timesteps,keys


def get_projection_operators(system_dict,arbor_dict,arbor_on,arbor_off,mode,layer):
	"""load or if file is not found compute normalisation vectors for LGN to L4 conn"""

	load_orth_vectors = False
	if (mode=="xalpha" and layer=="layer4"):
		load_orth_vectors = True
	elif (mode=="xalpha" and layer=="layer23" and\
		 system_dict["W4to23_params"]["plasticity_rule"] is not None):
		load_orth_vectors = True

	if load_orth_vectors:
		## Arbor parameters
		N4 = system_dict["N4"]
		N23 = system_dict["N23"]		
		Nlgn = system_dict["Nlgn"]
		Nvert = system_dict["Nvert"]
		
		arbor_profile_on = arbor_dict["arbor_profile_on"]
		arbor_profile_off = arbor_dict["arbor_profile_off"]
		if arbor_profile_on==arbor_profile_off:
			name_profile = "_{}".format(arbor_profile_on)
		else:
			name_profile = "_{}-{}".format(arbor_profile_on,arbor_profile_off)

		rA_on = arbor_dict["r_A_on"]
		rA_off = arbor_dict["r_A_off"]
		if rA_on==rA_off:
			name_rA = "_rA{}".format(np.around(rA_on,2))
		else:
			name_rA = "_rAon{}_rAoff{}".format(np.around(rA_on,2),np.around(rA_off,2))

		ampl_on = arbor_dict["ampl_on"]
		ampl_off = arbor_dict["ampl_off"]
		if (ampl_on!=ampl_off or ampl_on!=1):
			name_ampl = "_amplon{}_amploff{}".format(ampl_on,ampl_off)
		else:
			name_ampl = ""

		name = name_rA + name_profile + name_ampl

		try:
			if rA_on==rA_off:
				rA = rA_on
				if layer=="layer4":
					constraint_vec = np.load(data_dir +\
									 "layer4/P_orth/N4{}_Nlgn{}{}{}.npy".format(\
									 N4,Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),
									 name))
				elif layer=="layer23":
					constraint_vec = np.load(data_dir +\
									 "two_layer/P_orth/N23{}_N4{}{}{}.npy".format(\
									 N23,N4,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
									 name))
			else:
				if layer=="layer4":
					constraint_vec = np.load(data_dir +\
									 "layer4/P_orth/N4{}_Nlgn{}{}{}.npy".format(\
									 N4,Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),
									 name))
				elif layer=="layer23":
					constraint_vec = np.load(data_dir +\
									 "two_layer/P_orth/N23{}_N4{}{}{}.npy".format(\
									 N23,N4,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
									 name))
			print("constraint_vec",constraint_vec.shape)
			lim = constraint_vec.shape[0]//2
			c_orth = constraint_vec[:lim,:]
			s_orth = constraint_vec[lim:2*lim,:]
	
		except Exception as e:
			print(e)
			print("No file found for projection operator\n generating operator now...")
			sys.stdout.flush()
			if rA_on==rA_off:
				rA = rA_on
				if layer=="layer4":
					c_orth,s_orth = nc.generate_simIO_normalisation(Nlgn,N4,arbor_on,Nvert)
					np.save(data_dir + "layer4/P_orth/N4{}_Nlgn{}{}{}.npy".format(N4,\
							Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
							name),np.concatenate([c_orth,s_orth]))
				elif layer=="layer23":
					c_orth,s_orth = dynamics.generate_simIO_normalisation_oneUnittype(N4,\
										N23,arbor_on,Nvert=1)
					np.save(data_dir + "two_layer/P_orth/N23{}_N4{}{}{}.npy".format(N23,\
							N4,"" if Nvert==1 else "_Nvert{}".format(Nvert),
							name),np.concatenate([c_orth,s_orth]))
			else:
				if layer=="layer4":
					c_orth,s_orth = nc.generate_simIO_normalisation_onoff(Nlgn,N4,\
																		arbor_on,arbor_off,Nvert)
					np.save(data_dir + "layer4/P_orth/N4{}_Nlgn{}{}{}.npy".format(\
							N4,Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
							name),np.concatenate([c_orth,s_orth]))
				elif layer=="layer23":
					print("layer 23 c_orth,s_orth not implemented for different arbors")
	
	else:
		c_orth,s_orth = np.array([]),np.array([])
	
	return c_orth,s_orth




def PrintException(*args):
	exc_type, exc_obj, tb = sys.exc_info()
	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename
	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	if args:
		print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj), args)
	else:
		print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def plot_complex_map(complex_map):
	import matplotlib.colors as mcol
	h,w = complex_map.shape
	hsv_map = np.zeros((h,w,3))
	maxmax = np.nanmax(abs(complex_map))
	hsv_map[:,:,0] = ( (np.angle(complex_map)) % (2 * np.pi) ) / np.pi / 2.
	hsv_map[:,:,2] = np.clip(abs(complex_map)/maxmax/0.3,0,1)	# should be >0, and <1
	hsv_map[:,:,1] = 1	# if saturation=1, black is background color
	return mcol.hsv_to_rgb(hsv_map)


if __name__=="__main__":
	from bettina.modeling.ori_dev_model import data_dir,image_dir
	import matplotlib.pyplot as plt
	import pickle 

	## test get_response
	Version = 459
	params = pickle.load(open(data_dir + "layer4/habanero/v{v}/config_v{v}.p".format(\
		v=Version),"rb"))

	N4 = params["N4"]
	Nlgn = params["Nlgn"]
	rA = int(params["Wlgn_to4_params"]["r_A"] * N4)
	DA = 2*rA + 5
	try:
		Nvert = params["Nvert"]
	except:
		Nvert = 1
	print("DA",DA,rA,N4,Nvert,N4*Nvert//2)

	yt = np.load(data_dir + "layer4/habanero/v{v}/yt_v{v}.npz".format(v=Version))
	W = yt["Wt"][-1,:]
	W = W.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
	sd = W[0,...] - W[1,...]
	RFsd,PF,_,_ = get_RF_form(sd,N4,Nlgn,DA,calc_PF=True,Nvert=Nvert,mode="diff_only")
	
	## TEST get_RF_form function
	i = 0
	j = 2
	sd_rolled = np.roll(np.roll(sd[:,:,j,i],N4*Nvert//2-i*Nvert-Nvert//2,axis=1),N4//2-j,axis=0)
	fig = plt.figure()
	ax = fig.add_subplot(131)
	ax.imshow(sd[:,:,j,i],interpolation="nearest",cmap="RdBu_r")
	ax = fig.add_subplot(132)
	ax.imshow(sd_rolled[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2],\
		interpolation="nearest",cmap="RdBu_r")
	ax = fig.add_subplot(133)
	ax.imshow(PF[:4*DA,:4*DA],interpolation="nearest",cmap="RdBu_r")
	plt.show()

	## TEST OPM PLOTTING FUNCTION
	# opm,Rn,pref_phase,gabors = get_response(sd,DA,Nvert=Nvert)
	# print("gabors",RFsd.shape)
	# gabors = np.swapaxes(gabors,1,2)
	# gabors = gabors.reshape(DA*N4*2,DA*2*N4*Nvert)

	# fig = plt.figure()
	# ax = fig.add_subplot(141)
	# RFsd = RFsd.reshape(144,-1,2*DA)
	# im=ax.imshow(RFsd[:8*DA,0,:8*DA],interpolation="nearest",cmap="RdBu_r")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(142)
	# im=ax.imshow(plot_complex_map(opm),interpolation="nearest",cmap="RdBu_r")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(143)
	# im=ax.imshow(0.5*np.angle(opm,deg=True),interpolation="nearest",cmap='hsv')
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(144)
	# pref_phase = pref_phase.reshape(16,-1,2)
	# im=ax.imshow(pref_phase[:,0,:]/2./np.pi*360,interpolation="nearest",cmap="RdBu_r")#[:8*DA,:8*DA]
	# plt.colorbar(im,ax=ax)

	# g0 = gabor(sigma=2.,
	# 		 theta=80./360*2*np.pi,
	# 		 Lambda=2*np.pi/8.*8,
	# 		 psi=0.5, 
	# 		 gamma=1., 
	# 		 size=(18,18))
	# f0 = np.abs(np.fft.fftshift(np.fft.fft2(g0)))
	# g1 = gabor(sigma=2.,
	# 		 theta=(90-80.)/360*2*np.pi,
	# 		 Lambda=2*np.pi/8.*8,
	# 		 psi=0.5, 
	# 		 gamma=1., 
	# 		 size=(18,18))
	# fig = plt.figure()
	# ax = fig.add_subplot(121)
	# im=ax.imshow(g0,interpolation="nearest",cmap="RdBu_r")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(122)
	# im=ax.imshow(g1,interpolation="nearest",cmap="RdBu_r")
	# plt.colorbar(im,ax=ax)


	# tunign curves
	# fig = plt.figure()
	# for i in range(N4):
	# 	for j in range(N4*Nvert):
	# 		ax = fig.add_subplot(N4,N4*Nvert,j+1+i*N4*Nvert)
	# 		ax.plot(Rn[:,i,j],'-')
	plt.show()