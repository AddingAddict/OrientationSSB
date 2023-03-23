import numpy as np
from bettina.modeling.ori_dev_model.tools import misc


def update_params_dict(orig_dict,updated_params_dict):
	if updated_params_dict["nonlinearity_l4"]=="powerlaw":
		## change to different default parameters for powerlaw nonlinearity
		orig_dict["tau"] = 0.2
		orig_dict["gamma_lgn"] = 0.05
		# orig_dict["W4to4_params"]["sigma_EE"] = 0.2
		# orig_dict["W4to4_params"]["sigma_IE"] = 0.21
		# orig_dict["W4to4_params"]["sigma_EI"] = 0.2
		# orig_dict["W4to4_params"]["sigma_II"] = 0.18
		# orig_dict["W4to4_params"]["aEE"] = 0.044
		# orig_dict["W4to4_params"]["aIE"] = 0.042
		# orig_dict["W4to4_params"]["aEI"] = 0.023
		# orig_dict["W4to4_params"]["aII"] = 0.018
		orig_dict["W4to4_params"]["max_ew"] = 0.013
		orig_dict["W4to4_params"]["beta_P"] = 0.0005#0.01
		orig_dict["W4to4_params"]["sigma_EE"] = 0.1
		orig_dict["W4to4_params"]["sigma_IE"] = 0.1
		orig_dict["W4to4_params"]["sigma_EI"] = 0.07
		orig_dict["W4to4_params"]["sigma_II"] = 0.02
		orig_dict["W4to4_params"]["aEE"] = 11.4
		orig_dict["W4to4_params"]["aIE"] = 11.
		orig_dict["W4to4_params"]["aEI"] = 10.9
		orig_dict["W4to4_params"]["aII"] = 10.2


	if updated_params_dict["r_A"] is not None:
		if len(updated_params_dict["r_A"])==2:
			r_A_on = updated_params_dict["r_A"][0]
			orig_dict["Wlgn_to4_params"]["r_A_on"] = updated_params_dict["r_A"][0]
			orig_dict["Wlgn_to4_params"]["r_A_off"] = updated_params_dict["r_A"][1]
		elif len(updated_params_dict["r_A"])==1:
			r_A_on = updated_params_dict["r_A"][0]
			orig_dict["Wlgn_to4_params"]["r_A_on"] = updated_params_dict["r_A"][0]
			orig_dict["Wlgn_to4_params"]["r_A_off"] = updated_params_dict["r_A"][0]
	else:
		r_A_on = orig_dict["Wlgn_to4_params"]["r_A_on"]

	## developing parameters
	developing_params = ["beta_P","onoff_rel_weight","off_bias_region_size","off_bias_strength"]
	for key in developing_params:
		if updated_params_dict[key] is not None:
			if len(updated_params_dict[key])>1:
				orig_dict["developing_params"].update({key : updated_params_dict[key]})
			else:
				if key in orig_dict:
					orig_dict[key] = updated_params_dict[key][0]
				elif key in orig_dict["Inp_params"]:
					orig_dict["Inp_params"][key] = updated_params_dict[key][0]
				elif key in orig_dict["Wlgn_to4_params"]:
					orig_dict["Wlgn_to4_params"][key] = updated_params_dict[key][0]

	## arbor variations
	if updated_params_dict["arbor_profile"] is not None:
		if len(updated_params_dict["arbor_profile"])==2:
			orig_dict["Wlgn_to4_params"]["arbor_profile_on"] = updated_params_dict["arbor_profile"][0]
			orig_dict["Wlgn_to4_params"]["arbor_profile_off"] = updated_params_dict["arbor_profile"][1]
		else:
			orig_dict["Wlgn_to4_params"]["arbor_profile_on"] = updated_params_dict["arbor_profile"][0]
			orig_dict["Wlgn_to4_params"]["arbor_profile_off"] = updated_params_dict["arbor_profile"][0]

	if updated_params_dict["arbor_ampl"] is not None:
		if len(updated_params_dict["arbor_ampl"])==2:
			orig_dict["Wlgn_to4_params"]["ampl_on"] = updated_params_dict["arbor_ampl"][0]
			orig_dict["Wlgn_to4_params"]["ampl_off"] = updated_params_dict["arbor_ampl"][1]
		else:
			orig_dict["Wlgn_to4_params"]["ampl_on"] = updated_params_dict["arbor_ampl"][0]
			orig_dict["Wlgn_to4_params"]["ampl_off"] = updated_params_dict["arbor_ampl"][0]


	if updated_params_dict["Wret_to_LGN_profile"] is not None:
		orig_dict["Wret_to_lgn_params"]["profile"] = updated_params_dict["Wret_to_LGN_profile"]

	if updated_params_dict["MH_strength"] is not None:
		orig_dict["Wret_to_lgn_params"]["ampl2"] = \
		orig_dict["Wret_to_lgn_params"]["ampl1"] * updated_params_dict["MH_strength"]

	if updated_params_dict["simulate_activity"] is not None:
		if updated_params_dict["simulate_activity"]==0:
			updated_params_dict["simulate_activity"]=False
		elif updated_params_dict["simulate_activity"]==1:
			updated_params_dict["simulate_activity"]=True

	## set recurrent connectitivyy schemes
	if updated_params_dict["Wrec_mode"] is not None:
		orig_dict["W4to4_params"]["Wrec_mode"] = updated_params_dict["Wrec_mode"]
	if updated_params_dict["W23to4_mode"] is not None:
		orig_dict["W23to4_params"]["Wrec_mode"] = updated_params_dict["W23to4_mode"]
	if updated_params_dict["W4to23_mode"] is not None:
		orig_dict["W4to23_params"]["Wrec_mode"] = updated_params_dict["W4to23_mode"]
	if updated_params_dict["W23to23_mode"] is not None:
		orig_dict["W23to23_params"]["Wrec_mode"] = updated_params_dict["W23to23_mode"]

	if updated_params_dict["W4to23_plastic"]:
		orig_dict["W4to23_params"]["plastic"] = True
	if updated_params_dict["w4to23_width"] is not None:
		orig_dict["W4to23_params"]["sigma_EE"] *= updated_params_dict["w4to23_width"]
		orig_dict["W4to23_params"]["sigma_EI"] *= updated_params_dict["w4to23_width"]
		orig_dict["W4to23_params"]["sigma_IE"] *= updated_params_dict["w4to23_width"]
		orig_dict["W4to23_params"]["sigma_II"] *= updated_params_dict["w4to23_width"]
	if updated_params_dict["w23to4_ampl"] is not None:
		orig_dict["W23to4_params"]["aEE"] *= updated_params_dict["w23to4_ampl"]
		orig_dict["W23to4_params"]["aEI"] *= updated_params_dict["w23to4_ampl"]
		orig_dict["W23to4_params"]["aIE"] *= updated_params_dict["w23to4_ampl"]
		orig_dict["W23to4_params"]["aII"] *= updated_params_dict["w23to4_ampl"]

	if updated_params_dict["multiplicative_normalisation_lgn"] is not None:
		orig_dict["Wlgn_to4_params"]["mult_norm"] = updated_params_dict["multiplicative_normalisation_lgn"]
	if updated_params_dict["normalisation_mode_lgn"] is not None:
		orig_dict["Wlgn_to4_params"]["constraint_mode"] = updated_params_dict["normalisation_mode_lgn"]

	## update config_dict with given arguments
	for key in updated_params_dict.keys():

		if updated_params_dict[key] is not None:

			if key=="rC":
				orig_dict["Wret_to_lgn_params"]["sigma"] = updated_params_dict[key] * r_A_on
				orig_dict["Wret_to_lgn_params"]["sigma1"] = updated_params_dict[key] * r_A_on
				orig_dict["Wret_to_lgn_params"]["sigma2"] = updated_params_dict[key] * r_A_on * 2.5
		
			elif key=="lgn_corr_het":
				orig_dict["Wret_to_lgn_params"]["heterogeneity_type"] = updated_params_dict[key]

			elif key=="connectivity_type":
				orig_dict["Wlgn_to4_params"]["connectivity_type"] = updated_params_dict[key]
				orig_dict["num_lgn_paths"] = 2 if updated_params_dict[key]=="E" else 4

			elif key=="sI":
				orig_dict["W4to4_params"]["sigma_factor"] = updated_params_dict[key]
			elif key=="noise_rec":
				orig_dict["W4to4_params"]["noise"] = updated_params_dict[key]
			elif key=="rec_conn_het":
				orig_dict["W4to4_params"]["heterogeneity_type"] = updated_params_dict[key]
			elif key=="rec_conn_mean_ecc":
				orig_dict["W4to4_params"]["mean_eccentricity"] = updated_params_dict[key]
			elif key=="w4to4_ew":
				orig_dict["W4to4_params"]["max_ew"] = updated_params_dict[key]
			
			elif key not in developing_params:

				if key in orig_dict:
					orig_dict[key] = updated_params_dict[key]
				elif key in orig_dict["Wlgn_to4_params"]:
					orig_dict["Wlgn_to4_params"][key] = updated_params_dict[key]
				elif key in orig_dict["Inp_params"]:
					orig_dict["Inp_params"][key] = updated_params_dict[key]
				elif key in orig_dict["W4to4_params"]:
					orig_dict["W4to4_params"][key] = updated_params_dict[key]

		else:
			# print("{} not found in original dictionary".format(key))
			pass

	return orig_dict




def update_arbor_params(old_dict):
	"""
	updates params_dict from versions where arbor is identical between on and off lgn units
	"""
	old_keys = sorted(old_dict.keys())
	new_dict = old_dict
	new_keys = ["r_A_on","r_A_off","arbor_profile_on","arbor_profile_off","ampl_on","ampl_off"]
	for new_key in ["r_A_on","r_A_off"]:
		if new_key not in old_keys:
			new_dict.update({new_key : old_dict["r_A"]})

	for new_key in ["arbor_profile_on","arbor_profile_off"]:
		if new_key not in old_keys:
			new_dict.update({new_key : old_dict["arbor_profile"]})

	for new_key in ["ampl_on","ampl_off"]:
		if new_key not in old_keys:
			new_dict.update({new_key : 1.})

	return new_dict



def update_params(params_dict):
	"""
	update dictionary of settings from older simulations that are missing some new keywords
	"""
	params_dict["Wlgn_to4_params"] = update_arbor_params(params_dict["Wlgn_to4_params"])


	if "plastic" not in params_dict["W4to23_params"].keys():
		params_dict["W4to23_params"]["plastic"] = False
		params_dict["W4to23_params"]["arbor_profile"] = "gaussian"
		params_dict["W4to23_params"]["s_noise"] = 0.2


	if "mult_norm" not in params_dict["Wlgn_to4_params"]:
		params_dict["Wlgn_to4_params"]["mult_norm"] = "x"
	if "mult_norm" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["mult_norm"] = "None"
	if "mult_norm" not in params_dict["W23_params"]:
		params_dict["W23_params"]["mult_norm"] = "None"
	if "mult_norm" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["mult_norm"] = "None"
	if "mult_norm" not in params_dict["W23to4_params"]:
		params_dict["W23to4_params"]["mult_norm"] = "None"

	if "beta_P" not in params_dict["Wlgn_to4_params"]:
		params_dict["Wlgn_to4_params"]["beta_P"] = 0.005
	if "learning_rate" not in params_dict["Wlgn_to4_params"]:
		params_dict["Wlgn_to4_params"]["learning_rate"] = 0.5
	if "plasticity_rule" not in params_dict["Wlgn_to4_params"]:
		params_dict["Wlgn_to4_params"]["plasticity_rule"] = "activity_based"
	if "constraint_mode" not in params_dict["Wlgn_to4_params"]:
		params_dict["Wlgn_to4_params"]["constraint_mode"] = "xalpha_approx"

	if "nu_4" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["nu_4"] = 3.
	if "beta_P" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["beta_P"] = 0.0005
	if "learning_rate" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["learning_rate"] = 0.5
	if "Wlim" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["Wlim"] = 12
	if "plasticity_rule" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["plasticity_rule"] = "None"
	if "constraint_mode" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["constraint_mode"] = "None"
	if "firing_threshold" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["firing_threshold"] = "adaptive"
	if "theta_4" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["theta_4"] = 0.9
	if "l4_avg" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["l4_avg"] = 0.0
	if "rA_E" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["rA_E"] = 0.3
	if "rA_I" not in params_dict["W4to4_params"]:
		params_dict["W4to4_params"]["rA_I"] = 0.3
		
	if "nu_23" not in params_dict["W23_params"]:
		params_dict["W23_params"]["nu_23"] = 1.
	if "beta_P" not in params_dict["W23_params"]:
		params_dict["W23_params"]["beta_P"] = 0.0005
	if "Wlim" not in params_dict["W23_params"]:
		params_dict["W23_params"]["Wlim"] = 12.
	if "learning_rate" not in params_dict["W23_params"]:
		params_dict["W23_params"]["learning_rate"] = 0.2
	if "plasticity_rule" not in params_dict["W23_params"]:
		params_dict["W23_params"]["plasticity_rule"] = "None"
	if "constraint_mode" not in params_dict["W23_params"]:
		params_dict["W23_params"]["constraint_mode"] = "None"
	if "rA_E" not in params_dict["W23_params"]:
		params_dict["W23_params"]["rA_E"] = 0.3
	if "rA_I" not in params_dict["W23_params"]:
		params_dict["W23_params"]["rA_I"] = 0.3

	if "nu_4" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["nu_4"] = 1.
	if "beta_P" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["beta_P"] = 0.1
	if "Wlim" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["Wlim"] = 12.
	if "learning_rate" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["learning_rate"] = 0.2
	if "plasticity_rule" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["plasticity_rule"] = "None"
	if "constraint_mode" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["constraint_mode"] = "None"
	if "rA_E" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["rA_E"] = 0.3
	if "rA_I" not in params_dict["W4to23_params"]:
		params_dict["W4to23_params"]["rA_I"] = 0.3

	if "nu_23" not in params_dict["W23_params"]:
		params_dict["W23_params"]["nu_23"] = 1.
	if "beta_P" not in params_dict["W23_params"]:
		params_dict["W23_params"]["beta_P"] = 0.1
	if "Wlim" not in params_dict["W23_params"]:
		params_dict["W23_params"]["Wlim"] = 12.
	if "learning_rate" not in params_dict["W23_params"]:
		params_dict["W23_params"]["learning_rate"] = 0.2
	if "plasticity_rule" not in params_dict["W23_params"]:
		params_dict["W23_params"]["plasticity_rule"] = "None"
	if "constraint_mode" not in params_dict["W23_params"]:
		params_dict["W23_params"]["constraint_mode"] = "None"


	if not "saving_stepsize" in params_dict.keys():
		params_dict["saving_stepsize"] = 1
	if not "multiplicative_normalisation" in params_dict.keys():
		params_dict["multiplicative_normalisation"] = "x"
		
	if not "onoff_corr_factor" in params_dict["Inp_params"].keys():
		params_dict["Inp_params"].update({"onoff_corr_factor" : 1.})
	if not "onoff_rel_weight" in params_dict["Inp_params"].keys():
		params_dict["Inp_params"]["onoff_rel_weight"] = 1.
	if not "connectivity_type" in params_dict["Wlgn_to4_params"].keys():
		params_dict["Wlgn_to4_params"]["connectivity_type"] = "E"
		params_dict["num_lgn_paths"] = 2

	if "Nvert" not in params_dict.keys():
		params_dict["Nvert"] = 1
	if "simulate_activity" not in params_dict["Inp_params"].keys():
		params_dict["Inp_params"]["simulate_activity"] = True

	## added sept 14, 2021 to implement off dominance in lgn
	if "off_bias_strength" not in params_dict["Inp_params"].keys():
		params_dict["Inp_params"]["off_bias_strength"] = 0.0
	if "off_bias_region_size" not in params_dict["Inp_params"].keys():
		params_dict["Inp_params"]["off_bias_region_size"] = 0.5

	## added sept 14, 2021 to implement developing parameters
	if "developing_params" not in params_dict.keys():
		params_dict["developing_params"] = {}

	# return params_dict


def update_developing_parameters(timestep,params_dict):
	"""
	update before (!) each plasticity update step all parameters
	that are changing during development, 
	e.g. beta, off_bias_strength, off_bias_region_size,onoff_rel_weight
	arbor parameters, 
	"""

	for key,value in params_dict["developing_params"].items():

		new_value = value[0] + (value[1]-value[0])*timestep/value[2]
		new_value = np.clip(new_value,min(value[:2]),max(value[:2]))
		if key=="beta_P":
			params_dict["beta_P"] = new_value
		elif key=="off_bias_strength":
			params_dict["Inp_params"]["off_bias_strength"] = new_value

		elif key=="off_bias_region_size":
			params_dict["Inp_params"]["off_bias_region_size"] = new_value

		elif key=="onoff_rel_weight":
			params_dict["Inp_params"]["onoff_rel_weight"] = new_value


