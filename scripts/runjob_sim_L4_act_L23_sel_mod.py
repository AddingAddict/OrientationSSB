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
    parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=4)
    parser.add_argument('--n_phs', '-np', help='number of phases',type=int, default=12)
    parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=10)
    parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=20)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_ori = int(args['n_ori'])
    n_phs = int(args['n_phs'])
    n_rpt = int(args['n_rpt'])
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
    
    denss = 0.01*2**np.linspace(-1,1,5)
    grecs = np.linspace(1.05,1.25,5)
    seeds = range(20)

    for dens in denss:
        for grec in grecs:
            for seed in seeds:
                #--------------------------------------------------------------------------
                # Make SBTACH
                inpath = currwd + "/sim_L4_act_L23_sel_mod.py"
                c1 = "{:s} -s {:d} -no {:d} -np {:d} -nr {:d} -nt {:d} -d {:f} -g {:f}".format(
                    inpath,seed,n_ori,n_phs,n_rpt,n_int,dens,grec)
                
                jobname="{:s}_dens={:.4f}_grec={:.3f}_seed={:d}".format(
                    'sim_L4_act_L23_sel_mod',dens,grec,seed)
                
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


