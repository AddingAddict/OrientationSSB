import os
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random

from importlib import reload


def runjobs():


    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=int, default=0)
    parser.add_argument("--cluster_", help=" String", default='burg')
    parser.add_argument('--num_inner', '-ni', help='number of samples per outer loop',type=int, default=50)
    parser.add_argument('--num_outer', '-no', help='number of outer loops',type=int, default=5)
    parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=2)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    num_inner = int(args['num_inner'])
    num_outer = int(args['num_outer'])
    bayes_iter = int(args['bayes_iter'])
    gb = int(args['gb'])

    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()

    #--------------------------------------------------------------------------
    # Ofiles folder

    user = os.environ["USER"]
        
    if cluster=='haba':
        path_2_package="/rigel/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='moto':
        path_2_package="/moto/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='burg':
        path_2_package="/burg/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='axon':
        path_2_package="/home/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='local':
        path_2_package="/Users/tuannguyen/OrientationSSB/scripts"
        ofilesdir = path_2_package+"/Ofiles/"
        resultsdir = path_2_package + "/results/"


    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    time.sleep(0.2)
    
    outer_jobs = 5
    inner_jobs = 10

    for outer_idx in range(outer_jobs):
        #--------------------------------------------------------------------------
        # Make SBTACH
        inpath = currwd + "/sbi_l4_sel_mod_norm_kern.py"
        c1 = "{:s} -i $SLURM_ARRAY_TASK_ID -ni {:d} -no {:d} -bi {:d}".format(
            inpath,num_inner,num_outer,bayes_iter)
        jobname="{:s}_bayes_iter={:d}_job_id={:d}".format(
            'sbi_l4_sel_mod_nk',bayes_iter,outer_idx)

        if not args2.test:
            jobnameDir=os.path.join(ofilesdir, jobname)
            text_file=open(jobnameDir, "w");
            os. system("chmod u+x "+ jobnameDir)
            text_file.write("#!/bin/sh \n")
            if cluster=='haba' or cluster=='moto' or cluster=='burg':
                text_file.write("#SBATCH --account=theory \n")
            text_file.write("#SBATCH --job-name="+jobname+ "\n")
            text_file.write("#SBATCH -t 0-23:59  \n")
            text_file.write("#SBATCH --mem-per-cpu={:d}gb \n".format(gb))
            # text_file.write("#SBATCH --gres=gpu\n")
            text_file.write("#SBATCH -c 1 \n")
            text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%A_%a.o # STDOUT \n")
            text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%A_%a.e # STDERR \n")
            text_file.write("python  -W ignore " + c1+" \n")
            text_file.write("echo $PATH  \n")
            text_file.write("exit 0  \n")
            text_file.close()

            if cluster=='axon':
                os.system(f"sbatch -a {inner_jobs*outer_idx}-{inner_jobs*(outer_idx+1)-1} -p burst " +jobnameDir);
            else:
                os.system(f"sbatch -a {inner_jobs*outer_idx}-{inner_jobs*(outer_idx+1)-1} " +jobnameDir);
        else:
            print (c1)



if __name__ == "__main__":
    runjobs()


