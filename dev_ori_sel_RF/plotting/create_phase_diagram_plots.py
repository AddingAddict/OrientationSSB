"""
create phase diagram along francescos work as function of spatial scale of input correlations zeta and recurrent activity patterns eta (although eta is originallly defined as width of intracortical interaction function, but not intuitive to define that in activity based model)
"""

import numpy as np

#which versions to choose? could do that via arguments similar to when running simulation

# then loop over versions, compute eta,zeta, load RF and plot


if __name__=="__main__":
	import re
	import pickle
	from os import listdir
	from copy import copy
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	from collections import defaultdict

	from bettina.modeling.ori_dev_model import config_dict,data_dir,scan_simulations,image_dir
	from bettina.modeling.ori_dev_model.tools import parse_args,update_params_dict,\
	get_spatial_wavelength,misc

	args_dict = vars(parse_args.args)
	print("args_dict",args_dict)
	print("")

	lookup_dir = parse_args.get_data_dir()
	if isinstance(args_dict["idx"],int):
		Versions = [args_dict["idx"]]
	elif args_dict["idx"] is None:
		Versions = []
		listfiles = listdir(lookup_dir)
		for item in listfiles:
			name_match = re.match("v(\d+)",item)
			if name_match:
				Versions.append(int(name_match.group(1)))
	else:
		for item in args_dict["idx"]:
			if ":" in item:
				version_list = item.split(":")
				assert version_list[0]!="",\
				 "Error: start and end value expected for list of indices"
				assert version_list[1]!="",\
				 "Error: start and end value expected for list of indices"
				Versions = np.arange(int(version_list[0]),int(version_list[1])+1,1)
			elif "-" in item:
				version_list = item.split("-")
				assert version_list[0]!="",\
				 "Error: start and end value expected for list of indices"
				assert version_list[1]!="",\
				 "Error: start and end value expected for list of indices"
				Versions = np.arange(int(version_list[0]),int(version_list[1])+1,1)
			else:
				assert isinstance(int(item),int), "Error: int value expected for index"
				Versions = [int(item)]


	search_variables = copy(args_dict)
	for key in args_dict.keys():
		if search_variables[key] is None:
			search_variables.pop(key, None)
	search_variables.pop("idx", None)
	search_variables.pop("load_external_from", None)
	search_variables.pop("not_saving_temp", None)
	print("search_variables",search_variables)
	print("Versions",Versions)
	matching_versions = scan_simulations.scan_simulation_for_params(\
						 args_dict["load_external_from"][0],Versions,**search_variables)
	print("")
	print("matching_versions",matching_versions)

	load_external_from = args_dict["load_external_from"][0]
	cluster_name = "local" if load_external_from=="" else load_external_from
	print("cluster_name",cluster_name)

	result_dict = load_from_hdf5(cluster_name,version_list)


	for version in matching_versions[:1]:
		file_dir = lookup_dir + "v{}/".format(version)
		params = pickle.load(open(file_dir + "config_v{v}.p".format(v=version),"rb"))
		_,_,l4_t,l4I_t,_,_,keys = misc.load_data(version,file_dir,params)
		l4_t = l4_t[1::2,:]
		l4I_t = l4I_t[1::2,:]
		print("check shape Wlgn_to_4_t",l4_t.shape,l4I_t.shape,keys)
		## take only every 10 pattern
		l4_t = l4_t[::10,:]
		l4I_t = l4I_t[::10,:]
		analysis_results = pickle.load(open(data_dir +\
							 "layer4/habanero/results/v{v}_results.p".format(v=VERSION),"rb"))

		eta = result_dict[version]["cortical_wavelength"]

		rA = params["Wlgn_to4_params"]["r_A"]
		sigma = params["Wret_to_lgn_params"]["sigma"]
		fit_params,_,_ = determine_LGN_correlation_scale(sigma,rA,\
												Inp_params=params["Inp_params"],\
												Wret_to_lgn_params=params["Wret_to_lgn_params"])
		zeta = fit_params[0]

		analysis_results[]






