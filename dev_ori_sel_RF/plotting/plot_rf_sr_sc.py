import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
try:
	import PyPDF4
except:
	pass
import os
import re

from bettina.modeling.ori_dev_model import data_dir,image_dir
from bettina.modeling.ori_dev_model.tools import misc,update_params_dict


def plot_rf_sr_sc(version_list,load_external_from,show_fullfield):
	# version_list = [0]
	# version_list = np.arange(32,56)
	rA = 0.2

	N = len(version_list)
	ncol = np.ceil(np.sqrt(N))
	if ncol*(ncol-1)<N:
		nrow = ncol
	else:
		nrow = ncol - 1

	all_sc = []
	all_sr = []
	all_rf = []
	all_v = []
	for h,version in enumerate(version_list):
		if version in (118,122,126,130,134):
			continue
		try:
			## load parameters
			if load_external_from=="habanero":
				file_dir =\
				 "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/data/" +\
				 "layer4/habanero/v{v}/".format(v=version)
				if not os.path.isfile(file_dir+"config_v{v}.p".format(v=version)):
				 	file_dir = data_dir +\
					 "layer4/habanero/pickle_files/"
			elif load_external_from=="aws":
				file_dir = data_dir + "layer4/aws/v{v}/".format(v=version)
			else:
				file_dir = data_dir + "layer4/v{v}/".format(v=version)
			params = pickle.load(open(file_dir + "config_v{v}.p".format(v=version),"rb"))
			update_params_dict.update_params(params)
			
			Nlgn = params["Nlgn"]
			N4 = params["N4"]
			Nvert = params["Nvert"]
			sc = params["Wret_to_lgn_params"]["sigma"]*Nlgn
			sr = params["W4to4_params"]["sigma_factor"]*params["W4to4_params"]["sigma_EE"]*N4
			print("sr",sr,params["W4to4_params"]["sigma_factor"],params["W4to4_params"]["sigma_EE"],N4)
			DA = int(2*params["Wlgn_to4_params"]["r_A_on"] * N4) + 4
			num_lgn_paths = params["num_lgn_paths"]
			print('params',params["Wret_to_lgn_params"]["sigma"],\
				params["W4to4_params"]["sigma_EE"],params["W4to4_params"]["sigma_factor"])
			if params["Wlgn_to4_params"]["r_A_on"]!=rA:
				continue
			## load data
			try:
				y = np.load(file_dir + "y_v{v}.npz".format(v=version))
				if "Wlgn_to_4" in y.files:
					Wlgn_to_4 = y["Wlgn_to_4"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
				else:
					Wlgn_to_4 = y["W"].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
			except Exception as e:
				print("e",e)
				yt = np.load(file_dir + "yt_v{v}.npz".format(v=version))
				Wlgn_to_4 = yt["Wt"][-1,:].reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
				# yt = np.load(file_dir + "yt_v{v}.npy".format(v=version))
				# Wlgn_to_4 = yt[-1,:num_lgn_paths*N4**2*Nlgn**2].reshape(num_lgn_paths,N4*N4,-1)

			Wlgn_to_4 = Wlgn_to_4.reshape(num_lgn_paths,N4,N4*Nvert,Nlgn,Nlgn)
			RF = np.zeros((DA*N4,DA*N4))
			for i in range(N4*Nvert):
				for j in range(N4):
					son_ij = np.roll(np.roll(Wlgn_to_4[0,j,i,:,:],shift=Nlgn//2-j,axis=0),\
											shift=Nlgn//2-i,axis=1)
					sof_ij = np.roll(np.roll(Wlgn_to_4[1,j,i,:,:],shift=Nlgn//2-j,axis=0),\
											shift=Nlgn//2-i,axis=1)
					RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
					 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2] -\
					 sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]

			all_sc.append(sc)
			all_sr.append(sr)
			all_rf.append(RF)
			all_v.append(version)
		except:
			misc.PrintException()
			pass

	## order all_sc and all_sr
	sc_uni = np.unique(all_sc)
	sr_uni = np.unique(all_sr)
	ncol = len(sc_uni)
	nrow = len(sr_uni)

	fig = plt.figure(figsize=(ncol*5,nrow*5))
	gs = gridspec.GridSpec(nrows=nrow+1, ncols=ncol+1, wspace=0.3, hspace=0.2,\
							width_ratios=[0.2]+[1.0]*ncol,height_ratios=[1.0]*nrow+[0.2])


	for i in range(len(all_sr)):
		sc_idx = np.where(sc_uni==all_sc[i])[0][0]
		sr_idx = np.where(sr_uni==all_sr[i])[0][0]
		ax = fig.add_subplot(gs[sr_idx,sc_idx+1])
		ax.set_title("v{}, s_lgn={:.1f} pxl, s_EE={:.1f} pxl".format(all_v[i],all_sc[i],\
					all_sr[i]),fontsize=12)

		if show_fullfield:
			vm = np.nanmax(all_rf[i])*0.8
			im=ax.imshow(all_rf[i],interpolation="nearest",cmap="RdBu_r",vmin=-vm,vmax=vm)
		else:
			vm = np.nanmax(all_rf[i][:10*DA,:10*DA])*0.8
			im=ax.imshow(all_rf[i][:10*DA,:10*DA],interpolation="nearest",cmap="RdBu_r",\
				vmin=-vm,vmax=vm)
		plt.colorbar(im,ax=ax)
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])

	ax = fig.add_subplot(gs[-1,:])
	ax.text(0.45,0.5,"LGN correlation range rc",fontsize=15)
	ax.set_axis_off()
	ax = fig.add_subplot(gs[:,0])
	ax.text(0.5,0.65,"Cortical interaction range rI",rotation=90,fontsize=15)
	ax.set_axis_off()
	
	if load_external_from=="habanero":
		output_dir = image_dir + "layer4/habanero/"
	elif load_external_from=="aws":
		output_dir = image_dir + "layer4/aws/"
	else:
		output_dir = image_dir + "layer4/tmp/"
	plt.savefig(output_dir + "RF_{}-{}_rA{}{}.pdf".format(version_list[0],\
				version_list[-1],rA,"_ff" if show_fullfield else ""), bbox_inches='tight')
	# plt.show()


def collect_RF_plots(load_external_from,Versions):
	if load_external_from=="habanero":
		image_path = image_dir + "layer4/habanero/"
	elif load_external_from=="aws":
		image_path = image_dir + "layer4/aws/"
	else:
		image_path + "layer4/tmp/"
	listfiles = os.listdir(image_path)

	if len(listfiles)>0:
		matched_names = np.array(listfiles)[np.array(["v" in string for string in listfiles])]
		folder_names, present_versions = [],[]
		for item in matched_names:
			try:
				version_number_str = item.split("_")[0]
				version_number = int( re.findall(r"\d+",version_number_str)[0] )
				if version_number in Versions:
					folder_names.append(item)
					present_versions.append(version_number)
			except:
				misc.PrintException()
				print(item)
	else:
		return None

	present_versions = np.array(present_versions)
	folder_names = np.array(folder_names)
	sort_id = np.argsort(present_versions)
	folder_names = folder_names[sort_id]
	present_versions = present_versions[sort_id]

	pdf_writer = PyPDF4.PdfFileWriter()
	i = 0
	for Version,folder_name in zip(present_versions,folder_names):
		try:
			path = image_path + folder_name + "/rec_field.pdf"
			pdf = PyPDF4.PdfFileReader(path)
			# number_of_pages = pd.getNumPages()
			page_0 = pdf.getPage(0)
			pdf_writer.addPage(page_0)

			pdf_writer.addBookmark("v"+str(Version),i)
			# fig = plt.figure()
			# ax = fig.add_subplot(111)
			# ax.text(str(Version))
			i += 1
		except:
			# misc.PrintException()
			pass

	if load_external_from=="habanero":
		output_dir = image_dir + "layer4/habanero/"
	elif load_external_from=="aws":
		output_dir = image_dir + "layer4/aws/"
	else:
		output_dir + "layer4/tmp/"
	with open(output_dir + 'RF_{}-{}.pdf'.format(Versions[0],Versions[-1]), 'wb') as fh:
		pdf_writer.write(fh)


	pdf_writer = PyPDF4.PdfFileWriter()
	i = 0
	for Version,folder_name in zip(present_versions,folder_names):
		try:
			path = image_path + folder_name + "/opm.pdf"
			pdf = PyPDF4.PdfFileReader(path)
			# number_of_pages = pd.getNumPages()
			page_0 = pdf.getPage(-1)
			pdf_writer.addPage(page_0)

			pdf_writer.addBookmark("v"+str(Version),i)
			# fig = plt.figure()
			# ax = fig.add_subplot(111)
			# ax.text(str(Version))
			i += 1
		except:
			# misc.PrintException()
			pass

	
	with open(output_dir + 'opm_{}-{}.pdf'.format(Versions[0],Versions[-1]), 'wb') as fh:
		pdf_writer.write(fh)


if __name__=="__main__":
	import argparse
	import os
	from bettina.modeling.ori_dev_model.tools import misc

	current_user = os.environ["USER"]
	on_habanero = current_user=="bh2757"
	Version_default = misc.get_version(data_dir + "layer4/",version=None) - 1

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Plot results from simulation.")
	parser.add_argument("--index", dest='idx', help="specifying run number (either\
	 individual runs as x1 x2 x3 or endpoints of successive numbers of runs as x1:xn)",\
	 required=False,  nargs="*", default=Version_default)
	parser.add_argument("--ff", dest='fullfield', help="specifying if full grid of \
						receptive fields is shown", required=False, nargs="?", const=True,\
						 default=False, type=bool)
	parser.add_argument("--cl", dest='load_external_from', help="specify where data lies\
						(e.g. None, aws, habanero)", required=False,  nargs="*", default=[""])
	args = parser.parse_args()
	args_dict = vars(args)

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
			
	show_fullfield = args.fullfield
	print("show_fullfield:",show_fullfield)
	print("load_external_from:",load_external_from)

	# print("Plotting data from runs {}".format(str(Versions)))
	plot_rf_sr_sc(np.unique(Versions),load_external_from,show_fullfield)


	# collect_RF_plots(load_external_from,Versions)
