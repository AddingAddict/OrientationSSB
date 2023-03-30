import os
# import argparse
import numpy as np
# from subprocess import Popen
import time
import sys
from itertools import product

from dev_ori_sel_RF import data_dir
from dev_ori_sel_RF.tools import misc


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
	script_name = "run_onelayer"


	iterator_dict = {
					"sI" : np.array([1.]),#np.array([0.3,0.7,1.,1.3]),
					"rC" : np.array([0.25]),#np.array([0.15,0.25,0.35,0.5]),
					"rA" : np.array([0.25]), #"0.2 0.25",,
					"betaP" : np.array([0.0005]), ##(default: 0.005),

					"mult_norm" : np.array(["x"]),	#default: x,
					# "norm" : np.array(["x","xalpha","alpha"]),

					"runtime" : np.array([130000]), #1000000,
					# "avg_no_inp" : np.array([50]),
					# "onoff_rel_weight" : np.array([1.,2.]),#1.5,2., 3., 4.
					# "off_bias_strength" : np.array([1.5,2.]),
					# "off_bias_region_size" : np.array([0.5]),
					
					# "sim" : np.array(["dynamics"]), ##(default: 1),
					# "patt_dur" : np.array([1]), ##(default: 2500),

					## Recurrent layer
					# "nonlin" : np.array(["powerlaw"]), ##(default: "rectifier")
					# "gamma_lgn" : gamma_lgn,
					"wrec" : np.array(["Gaussian"]), ## default: Gaussian2pop,
					# "w4to4_ew" : np.array([1.05]),	## default: 0.95
					# "density" : np.array([0.3,0.7]),
					# "noise_rec" : np.array([0.2,0.4,0.6]),
					# "ncluster" : np.array([3,5]),
					# "Nvert" : np.array([1]),	##default: 1,
					# "aEE" : np.array([8.]),
					# "rec_conn_het" : np.array(["smooth"]),
					# "rec_conn_mean_ecc" : np.array([0.8]),

					## FF connections
					# "Wret_to_LGN_profile" : np.array(["Exponential"]), #default: Gaussian,
					# "MH_strength" : np.array([1.]),
					"arbor_profile" : ["gaussian"],
					# "lgn_corr_het"	: np.array(["smooth"]),
					# "connectivity_type" : np.array(["EI"]),
					# "arbor_ampl" : np.array(["1. 1.2","1. 1.5","1. 2."]), #"1. 2.",

					"saving_stepsize" : np.array([50]),

					# "test_lowDsubset" : np.array([1]),

					# "plasticity_rule" : np.array(["activity_based","activity_corr"])

					"load_params_file" : np.array(["onepop_dynamic"]) # GaussInpCorr  antolik_etal
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
		Version = 1032+i#misc.get_version(data_dir + "layer4/",version=None,readonly=False)
		print("Version={}".format(Version));sys.stdout.flush()
		c1 = "{:s} --V {:.0f}".format(inpath,Version) + var_string.format(*var_setting)

		jobname="v{}_ori_dev".format(Version)
		jobnameDir = os.path.join(ofilesdir, jobname)

		text_file = open(jobnameDir, "w");
		os. system("chmod u+x "+ jobnameDir)
		text_file.write("#!/bin/sh \n")


		if current_user=="hein":
			text_file.write("#SBATCH --job-name="+jobname+ "\n")
			text_file.write("#SBATCH -t 0-05:00  \n")


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
			text_file.write("#SBATCH --mem=50gb \n")## 60gb for 32x32,30gb for 25x25

			text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
			text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
			# text_file.write("source /rigel/home/bh2757/.bashrc \n")
			text_file.write("source /burg/home/bh2757/.bashrc \n")
			text_file.write("module load anaconda \n")
			# text_file.write("module load singularity \n")
			# text_file.write("singularity exec tensorflow_latest.sif python " + c1 + " \n")
			# text_file.write("singularity exec intel-optimized-tensorflow_latest.sif python " + c1 + " \n")
			text_file.write("python " + c1 + " \n")
			text_file.write("echo $PATH  \n")
			text_file.write("exit 0  \n")
			text_file.close()

		os.system("sbatch " + jobnameDir);
		#Popen(c1, shell=True)
		time.sleep(0.2)
		
		count += 1



	# count=0;
	# for iSI in SI:
	# 	for jRC in RC:
	# 		for krA in r_A:

	# 			time.sleep(0.2)
				
	# 			Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)
	# 			print("Version={}".format(Version));sys.stdout.flush()

	# 			c1 = "{:s} --V {:.0f} --rA {:.2f} --sI {:.2f} --rC {:.2f} ".format(inpath,\
	# 				Version,krA,iSI,jRC)
				
	# 			jobname="v{}_SI_{}_RC_{}_rA_{}".format(Version,iSI,jRC,krA)
	# 			jobnameDir=os.path.join(ofilesdir, jobname)

	# 			text_file=open(jobnameDir, "w");
	# 			os. system("chmod u+x "+ jobnameDir)
	# 			text_file.write("#!/bin/sh \n")


	# 			if current_user=="hein":
	# 				text_file.write("#SBATCH --job-name="+jobname+ "\n")
	# 				text_file.write("#SBATCH -t 0-05:00  \n")


	# 				# text_file.write("#SBATCH --mem=120gb \n")
	# 				# text_file.write("#SBATCH --gres=gpu:1 \n")

	# 				text_file.write("#SBATCH -p sleuths \n")
	# 				text_file.write("#SBATCH --gres gpu \n")
	# 				text_file.write("#SBATCH --mem=60gb \n")
	# 				text_file.write("#SBATCH --nodelist=turbine \n")
					

	# 				text_file.write("#SBATCH -o "+ ofilesdir + "%j.o # STDOUT \n")
	# 				text_file.write("#SBATCH -e "+ ofilesdir +"%j.e # STDERR \n")
	# 				text_file.write("source /home/hein/data_mpif2/code/bettina/tfenv/bin/activate \n") 
	# 				text_file.write("srun python " + c1 + " \n")
	# 				text_file.write("echo $PATH  \n")
	# 				text_file.write("exit 0  \n")
	# 				text_file.close()

	# 			elif current_user=="bh2757":
	# 				text_file.write("#SBATCH --account=theory \n")
	# 				text_file.write("#SBATCH --job-name="+jobname+ "\n")
	# 				text_file.write("#SBATCH -t 2-24:00  \n")


	# 				# text_file.write("#SBATCH --mem=120gb \n")
	# 				# text_file.write("#SBATCH --gres=gpu:1 \n")

	# 				text_file.write("#SBATCH -c 1 \n")
	# 				text_file.write("#SBATCH --mem=30gb \n")
					

	# 				text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
	# 				text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
	# 				text_file.write("source /rigel/home/bh2757/.bashrc \n") 
	# 				text_file.write("module load anaconda \n")
	# 				text_file.write("module load singularity \n")
	# 				text_file.write("singularity exec tensorflow_latest.sif python " + c1 + " \n")
	# 				text_file.write("echo $PATH  \n")
	# 				text_file.write("exit 0  \n")
	# 				text_file.close()

	# 			os.system("sbatch " + jobnameDir);
	# 			#Popen(c1, shell=True)
	# 			time.sleep(0.5)
				
	# 			count += 1


	print("{} jobs started".format(count));sys.stdout.flush()


if __name__ == "__main__":
	runjobs_oridevModel_Cluster()