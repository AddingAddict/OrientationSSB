import numpy as np
import os
from bettina.modeling.ori_dev_model import data_dir
from bettina.modeling.ori_dev_model.tools import misc


def delete(path):
	if os.path.exists(path):
		print("deleting {}".format(path))
		os.remove(path)


def load_ytfile_and_keep_only_last_timepoint(Version):
	data_dict = {}
	yt = np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version))
	data_dict["Wlgn_to_4"] = yt["Wt"][-1,:]
	## by default look at excitatory activity
	if "l4t" in yt.files:
		data_dict["l4"] = yt["l4t"][-2,:]
	
	if "lgn_inp" in yt.files:
		data_dict["lgn_inp"] = yt["lgn_inp"]

	if "cct" in yt.files:
		data_dict["cct"] = yt["cct"]

	print("yt.files",yt.files,data_dict.keys())
	filename = "layer4/v{v}/y_v{v}.npz".format(v=Version)
	misc.save_data(Version,filename,data_dict)

	delete(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version))


if __name__=="__main__":
	import argparse

	current_user = os.environ["USER"]
	on_habanero = current_user=="bh2757"

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Delete files with time resolved data,\
	 keep only data from last timepoint.")
	parser.add_argument("--index", dest='idx', help="specifying run number (either\
	 individual runs as x1 x2 x3 or endpoints of successive numbers of runs as x1:xn)",\
	 required=True,  nargs="*")
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



	for Version in Versions:
		try:
			load_ytfile_and_keep_only_last_timepoint(Version)
		except:
			misc.PrintException()