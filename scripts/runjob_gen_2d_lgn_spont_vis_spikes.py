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
    parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=15)
    parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
    parser.add_argument('--n_shrink', '-nh', help='factor by which to shrink stimuli',type=float, default=1.0)
    parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=100)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_wave = int(args['n_wave'])
    n_stim = int(args['n_stim'])
    n_shrink = args['n_shrink']
    n_grid = int(args['n_grid'])
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
    
    seeds = range(10)
    modes = ['spont','vis','spont_vis']

    for mode in modes:
        for seed in seeds:
            #--------------------------------------------------------------------------
            # Make SBTACH
            inpath = currwd + "/gen_2d_lgn_spont_vis_spikes.py"
            c1 = "{:s} -s {:d} -nw {:d} -ns {:d} -nh {:.2f} -ng {:d} -m {:s}".format(
                inpath,seed,n_wave,n_stim,n_shrink,n_grid,mode)
            
            jobname="{:s}_m={:s}_nw={:d}_ns={:d}_nh={:.2f}_ng={:d}_seed={:d}".format(
                'gen_2d_lgn_spikes',n_wave,n_stim,n_shrink,n_grid,seed)
            
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


