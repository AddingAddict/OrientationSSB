"""
functions for generating one-layer or two-layer network with inputs
"""
import numpy as np
from scipy import linalg

from bettina.modeling.ori_dev_model import connectivity, inputs
from bettina.modeling.ori_dev_model import data_dir



def get_RFs(config_dict,mode,**kwargs):
	if mode=="initialize":
		Wlgn4 = kwargs["Wlgn4"]
		Won_to_4 = Wlgn4.create_matrix(config_dict["Wlgn_to4_params"], "random_delta",\
					r_A=config_dict["Wlgn_to4_params"]["r_A"],profile_A="heaviside") #* 0.5
		Wof_to_4 = Wlgn4.create_matrix(config_dict["Wlgn_to4_params"], "random_delta",\
					r_A=config_dict["Wlgn_to4_params"]["r_A"],profile_A="heaviside") #* 2.
		Wlgn_to_4 = np.stack([Won_to_4,Wof_to_4])

	elif mode=="gabor":
		Nlgn = config_dict["Nlgn"]
		N4 = config_dict["N4"]
		## smooth OPM generation
		grid = np.linspace(0,1,N4,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200205)}
		ecp,sigma = connectivity.gen_ecp(xto, yto, conn_params)
		opm = np.angle(ecp,deg=False)*0.5
		## smooth phases generation
		grid = np.linspace(0,1,N4,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200205), "kc" : 1.5, "n" : 1}
		ecp,sigma = connectivity.gen_ecp(xto, yto, conn_params)
		pref_phase = np.angle(ecp,deg=False)
		
		conn = connectivity.Connectivity((Nlgn,Nlgn),(N4,N4),random_seed=12345)
		conn_params = {"sigma" : 0.2,
						"ampl" : 1.,
						"theta" : opm,#0.3*np.ones((Nlgn,Nlgn)),
						"psi" : pref_phase,
						"freq" : 15}
		gb = conn.create_matrix(conn_params, "Gabor")
		Wlgn_to_4_on = np.copy(gb)
		Wlgn_to_4_off = np.copy(gb)
		Wlgn_to_4_on[Wlgn_to_4_on<0] = 0
		Wlgn_to_4_off[Wlgn_to_4_off>0] = 0
		Wlgn_to_4_off *= -1.
		Wlgn_to_4 = np.stack([Wlgn_to_4_on,Wlgn_to_4_off])

	elif mode=="load_from_external":
		Version = config_dict["Wlgn_to4_params"]["load_from_prev_run"]
		print(" ")
		print("Load ff connection from version {}".format(Version))
		print(" ")
		Nlgn = config_dict["Nlgn"]
		N4 = config_dict["N4"]
		Nvert = config_dict["Nvert"]
		try:
			yfile = np.load(data_dir + "layer4/habanero/y_files/y_v{v}.npz".format(v=Version))
			Wlgn_to_4 = yfile["W"].reshape(2,N4**2*Nvert,Nlgn**2)
		except:
			with np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
				Wlgn_to_4 = yt["Wt"][-1,:].reshape(2,N4**2,Nlgn**2)

	return Wlgn_to_4


def generate_network(config_dict,**kwargs):
	Nlgn = config_dict["Nlgn"]
	N4 = config_dict["N4"]
	Nvert = config_dict["Nvert"]

	## ==== Connectivity from LGN to Layer 4 E ==================
	Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
			 random_seed=config_dict["random_seed"],Nvert=(1,Nvert))
	arbor_profile = config_dict["Wlgn_to4_params"]["arbor_profile"]
	arbor = Wlgn4.create_arbor(radius=config_dict["Wlgn_to4_params"]["r_A"],\
								profile=arbor_profile)
	Wlgn_to_4 = get_RFs(config_dict,kwargs["RF_mode"],Wlgn4=Wlgn4,**kwargs)
	
	## ==== Connectivity from LGN to Layer 4 I ==================
	Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
	if (config_dict["Wlgn_to4_params"]["connectivity_type"]=="EI" and "2pop" in Wrec_mode):
		Wlgn4_I = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
			 		random_seed=config_dict["random_seed"]+1,Nvert=(1,Nvert))
		arbor_profile = config_dict["Wlgn_to4_params"]["arbor_profile"]
		arbor_I = Wlgn4_I.create_arbor(radius=config_dict["Wlgn_to4_params"]["r_A"],\
									 profile=arbor_profile)
		Wlgn_to_4_I = get_RFs(config_dict,kwargs["RF_mode"],Wlgn4=Wlgn4_I,**kwargs)
		Wlgn_to_4 = np.concatenate([Wlgn_to_4,Wlgn_to_4_I])
	arbor2 = np.stack([arbor]*config_dict["num_lgn_paths"])

	## init normalization
	## syn norm over x
	init_weights = np.sum(Wlgn_to_4,axis=1)
	## syn norm over alpha
	# init_weights = np.sum(Wlgn_to_4,axis=2)
	## ============================ Lateral conn Layer 4 ==================================
	## ====================================================================================
	W4 = connectivity.Connectivity((N4,N4), (N4,N4),random_seed=config_dict["random_seed"],\
									 Nvert=Nvert)
	Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]

	##  =========== 1pop ===========
	sigma_rec = config_dict["W4to4_params"]["sigma_factor"]
	max_ew = config_dict["W4to4_params"]["max_ew"]
	## one E population layer 4
	if "2pop" not in Wrec_mode:
		config_dict["W4to4_params"]["sigma"] *= sigma_rec
		x_I = np.clip(config_dict["W4to4_params"]["sigma"]*2,0.1,N4/2.)
		W4to4 = W4.create_matrix(config_dict["W4to4_params"], profile=Wrec_mode)#, r_A=x_I)
		ew,_ = linalg.eig(W4to4,right=True)
		W4to4 /= np.nanmax(np.real(ew))/max_ew
	##  =========== 2pop ===========
	## E and I population layer 4
	elif "2pop" in Wrec_mode:
		W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),\
			 random_seed=config_dict["random_seed"],Nvert=Nvert)
		W4to4 = W4.create_matrix_2pop(config_dict["W4to4_params"],Wrec_mode)

	if kwargs["system"]=="one_layer":
		system = (Wlgn_to_4,arbor,arbor2,init_weights,W4to4)

	elif kwargs["system"]=="two_layer":
		N23 = config_dict["N23"]
		W4 = connectivity.Connectivity_2pop((N23,N23), (N23,N23), (N23,N23), (N23,N23),\
			 random_seed=config_dict["random_seed"])
		Wrec_mode = config_dict["W23_params"]["Wrec_mode"]
		W23to23 = W4.create_matrix_2pop(config_dict["W23_params"],Wrec_mode)
		## not scaled to Nvert, because no Nvert, should be fine
		## ======================== Afferent conn L4 to L23 ===================================
		## ====================================================================================
		W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N23,N23), (N23,N23),\
			 random_seed=config_dict["random_seed"],Nvert=(Nvert,1))
		Wrec_mode = config_dict["W4to23_params"]["Wrec_mode"]
		W4to23 = W4.create_matrix_2pop(config_dict["W4to23_params"],Wrec_mode)
		## scaled by normalisation to Nvert
		# W4to23 /= float(Nvert)
		## ======================== Feedback conn L4 to L23 ===================================
		## ====================================================================================
		W4 = connectivity.Connectivity_2pop((N23,N23), (N23,N23), (N4,N4), (N4,N4),\
			 random_seed=config_dict["random_seed"],Nvert=(1,Nvert))
		Wrec_mode = config_dict["W23to4_params"]["Wrec_mode"]
		W23to4 = W4.create_matrix_2pop(config_dict["W23to4_params"],Wrec_mode)
		## not scaled to Nvert, because no Nvert in from_size
		# W23to4 /= float(Nvert)

		system = (Wlgn_to_4,arbor,arbor2,init_weights,W4to4,W23to23,W4to23,W23to4)
	return system


def generate_inputs(config_dict,**kwargs):
	Nret = config_dict["Nret"]
	Nlgn = config_dict["Nlgn"]
	Nvert = config_dict["Nvert"]
	Version = kwargs["Version"]
	## moving sinusoidal input of varying orientation and spatial frequency
	Wret_to_lgn = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
				  random_seed=config_dict["random_seed"],Nvert=1).create_matrix(\
				  config_dict["Wret_to_lgn_params"],config_dict["Wret_to_lgn_params"]["profile"])*\
				  config_dict["Wret_to_lgn_params"]["gamma_ret"]
	
	if config_dict["Wret_to_lgn_params"]["profile"]=="Gaussian_broadOFF":
		Wret_to_lgn_OFF = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
							random_seed=config_dict["random_seed"],Nvert=1).create_matrix(\
							{"ampl" : config_dict["Wret_to_lgn_params"]["ampl"],\
							"noise" : config_dict["Wret_to_lgn_params"]["noise"],\
							"sigma" : 2*config_dict["Wret_to_lgn_params"]["sigma"]},\
							"Gaussian") * config_dict["Wret_to_lgn_params"]["gamma_ret"]
		Wret_to_lgn = np.stack([Wret_to_lgn,Wret_to_lgn_OFF])

	if config_dict["Inp_params"]["input_type"]=="moving_grating_online":
		print("Wret_to_lgn as diagonal???")
		Wret_to_lgn = np.diagflat(np.ones(Wret_to_lgn.shape[0],dtype=float))

		lgn_input_on,lgn_input_off = [],[]
		num_freq = kwargs["num_freq"]#3
		num_oris = kwargs["num_oris"]#4
		Nsur = kwargs["Nsur"]#10
		spat_frequencies = kwargs["spat_frequencies"]#np.array([80,90,120])
		orientations = kwargs["orientations"]
		for spat_frequency in spat_frequencies:
			for orientation in orientations:
				config_dict["Inp_params"]["spat_frequency"] = spat_frequency # vary between 60 and 120 (3 steps?)
				config_dict["Inp_params"]["Nsur"] = Nsur
				config_dict["Inp_params"]["orientation"] = orientation # vary in 8 steps 
				lgn_input_on.append( inputs.Inputs_lgn((Nret,Nret),1,2020).create_lgn_input(\
								config_dict["Inp_params"], "moving_grating_online", Wret_to_lgn) )
				lgn_input_off.append( inputs.Inputs_lgn((Nret,Nret),1,2020).create_lgn_input(\
								config_dict["Inp_params"], "moving_grating_online", -Wret_to_lgn) )
		lgn_input_on = np.array(lgn_input_on)
		lgn_input_off = np.array(lgn_input_off)
		lgn = np.stack([lgn_input_on,np.array(lgn_input_off)])
		lgn = lgn.reshape(2,num_freq,num_oris,Nsur,-1)

	elif config_dict["Inp_params"]["input_type"]=="white_noise_online":
		lgn,lgnI = [],[]
		if kwargs["full_lgn_output"]:
			# generate only lgn input if not "online" generation of it anyways
			last_timestep = kwargs["last_timestep"]
			T_pd = kwargs["T_pd"]
			avg_no_inp = config_dict["Inp_params"]["avg_no_inp"]
			num_inputs = int(config_dict["runtime"]/config_dict["dt"]/\
							config_dict["Inp_params"]["pattern_duration"])
			num_plasticity_steps = int(num_inputs/avg_no_inp)
			num_lgn_paths = config_dict["num_lgn_paths"]
			for istep in range(num_plasticity_steps):
				for jinput in range(avg_no_inp):
					## generate LGN input per L4 simulation
					if (jinput%2)==1:
						continue
					else:
						rng_seed = config_dict["random_seed"]*1000 + jinput+istep*avg_no_inp
						lgn.append(inputs.Inputs_lgn((Nret,Nret),Version,\
								  rng_seed).create_lgn_input(\
								  config_dict["Inp_params"],\
								  config_dict["Inp_params"]["input_type"],\
								  Wret_to_lgn
								  ))
						if num_lgn_paths==4:
							lgnI.append(inputs.Inputs_lgn((Nret,Nret),Version,\
									  rng_seed+10000*0).create_lgn_input(\
									  config_dict["Inp_params"],\
									  config_dict["Inp_params"]["input_type"],\
									  Wret_to_lgn
									  ))
			lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
			if num_lgn_paths==4:
				lgnI = np.swapaxes(np.swapaxes(np.array(lgnI),0,1),1,2)
				lgn = np.concatenate([lgn,lgnI])
			# lgn = []
			# # for i in range(1,int((last_timestep+1)/config_dict["Inp_params"]["avg_no_inp"]/T_pd)):
			# for i in range(2497,2510):
			# 	for it in range(config_dict["Inp_params"]["avg_no_inp"]):
			# 		if (config_dict["Inp_params"]["simulate_activity"] and ((int(np.floor((i+1)/T_pd)) - it)%2)==0):
			# 			# print(i,it)
			# 			continue
			# 		rng_seed = config_dict["random_seed"]*1000 +\
			# 				   i*config_dict["Inp_params"]["avg_no_inp"] - 1 - it
			# 		print("i",i,rng_seed)
			# 		lgn.append( inputs.Inputs_lgn((Nret,Nret),Version,rng_seed).create_lgn_input(\
			# 			config_dict["Inp_params"], "white_noise_online", Wret_to_lgn) )
			# lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
	return lgn,Wret_to_lgn



if __name__=="__main__":
	import os
	import sys
	import numpy as np
	import matplotlib.pyplot as plt

	from bettina.modeling.ori_dev_model import Nlgn, Nret, N4, data_dir, config_dict
	from bettina.modeling.ori_dev_model.tools import analysis_tools

	Version = 1
	## Parameters
	dt = config_dict["dt"]
	t = np.arange(0,config_dict["runtime"]/dt,1).astype("float64")
	Nvert = config_dict["Nvert"]

	config_dict["Wlgn_to4_params"].update({"connectivity_type" : "E" })
	config_dict["W4to4_params"].update({"Wrec_mode" : "Mexican-hat" })

	last_timestep = config_dict["runtime"]/dt
	## number of input patterns needed
	T_pd = config_dict["Inp_params"]["pattern_duration"]
	T_exp = config_dict["Inp_params"]["expanse_time"]
	config_dict["Inp_params"].update({"Nsur" : int(np.ceil(1.*last_timestep/T_pd/(T_exp+1) ))})
	print("# of stimuli: {}".format(config_dict["Inp_params"]["Nsur"]));sys.stdout.flush()

	network_kwargs = {
				"T_pd" : T_pd,
				"last_timestep" : last_timestep,
				"T_exp" : T_exp,
				"RF_mode" : "gabor",#"initialize",
				"system" : "one_layer",
				"Version" : Version,
				"full_lgn_output"	:	True,
				}

	lgn,Wret_to_lgn = generate_inputs(config_dict,**network_kwargs)
	system = generate_network(config_dict,**network_kwargs)
	Wlgn_to_4,arbor,arbor2,init_weights,W4to4 = system
	print("Wlgn_to_4",Wlgn_to_4.shape,init_weights.shape,arbor.shape)

	N = config_dict["N4"]
	fig = plt.figure()
	ax = fig.add_subplot(121)
	im=ax.imshow(W4to4[:,(N+1)*N//2].reshape(N,N),interpolation="nearest",cmap="RdBu_r")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(122)
	ax.plot(W4to4[:,(N+1)*N//2].reshape(N,N)[N//2+1,:],"-")
	plt.show()
	exit()

	N4 = config_dict["N4"]
	Nlgn = config_dict["Nlgn"]
	rA = int(config_dict["Wlgn_to4_params"]["r_A"] * N4)
	DA = 2*rA + 5
	RF,_,_,_, = analysis_tools.get_RF_form(Wlgn_to_4,N4,Nlgn,DA,calc_PF=False,Nvert=1,\
											mode="full_onoff")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	im=ax.imshow(RF[0,:,:],interpolation="nearest",cmap="RdBu_r")
	plt.colorbar(im,ax=ax)
	plt.show()
