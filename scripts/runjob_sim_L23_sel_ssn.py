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
    parser.add_argument('--n_inp', '-ni', help='number of inputs',type=int, default=200)
    parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=20)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_inp = int(args['n_inp'])
    n_int = int(args['n_int'])
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
    
    ksels = 10**np.linspace(-3,0,7)
    lkers = np.concatenate(([0],10**np.linspace(-3,0,7)[:-1]))
    grecs = 1.02*np.linspace(0.95,1.05,5)
    betxs = np.linspace(0.5,1.5,5)
    seeds = range(1)

    for ksel in ksels:
        for lker in lkers:
            for grec in grecs:
                for betx in betxs:
                    for seed in seeds:
                        #--------------------------------------------------------------------------
                        # Make SBTACH
                        inpath = currwd + "/sim_L23_sel_ssn.py"
                        c1 = "{:s} -s {:d} -ni {:d} -nt {:d} -k {:f} -l {:f} -g {:f} -b {:f}".format(
                            inpath,seed,n_inp,n_int,ksel,lker,grec,betx)
                        
                        jobname="{:s}_ksel={:.3f}_lker={:.3f}_grec={:.3f}_betx={:.2f}_seed={:d}".format(
                            'ori_dev_sim_L23_sel_ssn',ksel,lker,grec,betx,seed)
                        
                        if not args2.test:
                            jobnameDir=os.path.join(ofilesdir, jobname)
                            text_file=open(jobnameDir, "w");
                            os. system("chmod u+x "+ jobnameDir)
                            text_file.write("#!/bin/sh \n")
                            if cluster=='haba' or cluster=='moto' or cluster=='burg':
                                text_file.write("#SBATCH --account=theory \n")
                            text_file.write("#SBATCH --job-name="+jobname+ "\n")
                            text_file.write("#SBATCH -t 0-11:59  \n")
                            text_file.write("#SBATCH --mem-per-cpu={:d}gb \n".format(gb))
                            # text_file.write("#SBATCH --gres=gpu\n")
                            text_file.write("#SBATCH -c 1 \n")
                            text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                            text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                            text_file.write("python  -W ignore " + c1+" \n")
                            text_file.write("echo $PATH  \n")
                            text_file.write("exit 0  \n")
                            text_file.close()

                            if cluster=='axon':
                                os.system("sbatch -p burst " +jobnameDir);
                            else:
                                os.system("sbatch " +jobnameDir);
                        else:
                            print (c1)



if __name__ == "__main__":
    runjobs()


