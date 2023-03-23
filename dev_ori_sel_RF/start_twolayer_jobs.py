import os
# import argparse
import numpy as np
# from subprocess import Popen
import time
import sys
from itertools import product

from bettina.modeling.ori_dev_model import data_dir
from bettina.modeling.ori_dev_model.tools import misc



def runjobs_oridevModel_Cluster():
	"""
		Function to be run in slurm queuing system to start list of jobs
		
	"""
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--test", required=True, type=int, default=0)
	# args = parser.parse_args()
	# if (args.test):
	#     print ("testing commands")
	
	# Vectors for iteration
	script_name = "run_twolayer"

	iterator_dict = {
					# "sI" : np.array([0.3]), #([0.2,0.5,1.0,1.4]),
					"rC" : np.array([0.2]), #([0.1,0.2,0.5,0.95]),
					"rA" : np.array([0.3]), #"0.2 0.25",,
					"betaP" : np.array([5]), ##(default: 0.005),

					# "mult_norm" : np.array(["homeostatic"]),	#default: x,
					# "norm" : np.array(["x","xalpha","alpha"]),

					"runtime" : np.array([50000]), #1000000,
					# "avg_no_inp" : np.array([20]),
					# "onoff_rel_weight" : np.array([2.]),
					# "off_bias_strength" : np.array([1.5,2.]),
					# "off_bias_region_size" : np.array([0.5]),
					
					# "sim" : np.array([0]), ##(default: 1),
					# "patt_dur" : np.array([1]), ##(default: 2500),

					## layer 4
					# "nonlin" : np.array(["powerlaw"]), ##(default: "rectifier")
					# "gamma_lgn" : gamma_lgn,
					# "wrec" : np.array(["Gaussian2pop"]), ## default: Gaussian2pop,
					# "w4to4_ew" : np.array([1.02]),	## default: 0.95
					# "density" : np.array([0.3,0.7]),
					# "noise_rec" : np.array([0.2,0.4,0.6]),
					# "ncluster" : np.array([3,5]),
					# "Nvert" : np.array([1]),	##default: 1,
					# "aEE" : np.array([8.]),
					# "rec_conn_het" : np.array(["independent"]),
					# "rec_conn_mean_ecc" : np.array([0.7]),

					## Layer 23 
					# "w23to4_ampl" : np.array([1.,2.]),
					# "w4to23_width" : np.array([0.075/0.1]),

					## FF connections
					# "Wret_to_LGN_profile" : np.array(["Gaussian"]), #default: Gaussian,
					# "MH_strength" : np.array([1.]),
					"arbor_profile" : ["heaviside"],
					# "lgn_corr_het"	: np.array(["independent"]),
					# "connectivity_type" : np.array(["EI"]),
					# "arbor_ampl" : np.array(["1. 1.2","1. 1.5","1. 2."]), #"1. 2.",

					"saving_stepsize" : np.array([20]),

					"load_params_file" : np.array(["GaussInpCorr"]) # GaussInpCorr  antolik_etal
					}


	currwd = os.getcwd()
	current_user = os.environ["USER"]
	print("currwd",currwd,current_user);sys.stdout.flush()
	if current_user=="hein":
		ofilesdir = "/home/hein/data_mpif2/code/"
		inpath = currwd + "/bettina/modeling/ori_dev_model/{}.py ".format(script_name)
	elif current_user=="bh2757":
		ofilesdir = currwd + "/projects/ori_dev_model/Ofiles/"
		# ofilesdir = "/rigel/theory/users/bh2757/columbia/projects/ori_dev_model/Ofiles/"
		# ofilesdir = "/burg/theory/users/bh2757/columbia/projects/ori_dev_model/Ofiles/"
		inpath = currwd + "/code/bettina/modeling/ori_dev_model/{}.py ".format(script_name)
		# data_dir = "/rigel/theory/users/bh2757/columbia/projects/ori_dev_model/data/"


	var_keys = sorted(iterator_dict.keys())
	n_variables = len(var_keys)
	count = 0
	var_list = []
	var_string = ""
	for key in var_keys:
		var_list.append(iterator_dict[key])
		var_string += " --{:s}".format(key) + " {}"

	count=0;
	for i,var_setting in enumerate(product(*var_list)):
		time.sleep(0.2)
		Version = 14+i#misc.get_version(data_dir + "two_layer/",version=None,readonly=False)
		print("Version={}".format(Version));sys.stdout.flush()
		c1 = "{:s} --V {:.0f}".format(inpath,Version) + var_string.format(*var_setting)

		jobname="v{}_twolayer".format(Version)
		jobnameDir = os.path.join(ofilesdir, jobname)

		text_file = open(jobnameDir, "w");
		os. system("chmod u+x "+ jobnameDir)
		text_file.write("#!/bin/sh \n")


		if current_user=="hein":
			text_file.write("#SBATCH --job-name="+jobname+ "\n")
			text_file.write("#SBATCH -t 0-12:00  \n")


			# text_file.write("#SBATCH --mem=120gb \n")
			# text_file.write("#SBATCH --gres=gpu:1 \n")

			text_file.write("#SBATCH -p sleuths \n")
			text_file.write("#SBATCH --gres gpu \n")
			text_file.write("#SBATCH --mem=60gb \n")
			text_file.write("#SBATCH --nodelist=turbine \n")
			

			text_file.write("#SBATCH -o "+ ofilesdir + "%j.o # STDOUT \n")
			text_file.write("#SBATCH -e "+ ofilesdir +"%j.e # STDERR \n")
			text_file.write("source /home/hein/data_mpif2/code/bettina/tfenv/bin/activate \n") 
			text_file.write("srun python " + c1 + " \n")
			text_file.write("echo $PATH  \n")
			text_file.write("exit 0  \n")
			text_file.close()

		elif current_user=="bh2757":
			text_file.write("#SBATCH --account=theory \n")
			text_file.write("#SBATCH --job-name="+jobname+ "\n")
			text_file.write("#SBATCH -t 0-12:00  \n")## four days for 32x32

			# text_file.write("#SBATCH --mem=120gb \n")
			# text_file.write("#SBATCH --gres=gpu:1 \n")

			text_file.write("#SBATCH -c 1 \n")
			text_file.write("#SBATCH --mem=40gb \n")## 60gb for 32x32,30gb for 25x25

			text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
			text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
			# text_file.write("source /rigel/home/bh2757/.bashrc \n")
			text_file.write("source /burg/home/bh2757/.bashrc \n")
			text_file.write("module load anaconda \n")
			# text_file.write("module load singularity \n")
			## text_file.write("singularity exec tensorflow_latest.sif python " + c1 + " \n")
			# text_file.write("singularity exec intel-optimized-tensorflow_latest.sif python " + c1 + " \n")
			text_file.write("python " + c1 + " \n")
			text_file.write("echo $PATH  \n")
			text_file.write("exit 0  \n")
			text_file.close()

		os.system("sbatch " + jobnameDir);
		#Popen(c1, shell=True)
		time.sleep(0.2)
		
		count += 1


	print("{} jobs started".format(count));sys.stdout.flush()


if __name__ == "__main__":
	runjobs_oridevModel_Cluster()