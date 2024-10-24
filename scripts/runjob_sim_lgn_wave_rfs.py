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
    parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=16)
    parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=4)
    parser.add_argument('--init_iter', '-iit', help='initial iteration number',type=int, default=0)
    parser.add_argument('--batch_iter', '-bit', help='number of iterations to run per batch',type=int, default=100)
    parser.add_argument('--max_iter', '-mit', help='max iteration number',type=int, default=100)
    parser.add_argument('--gain_e', '-ge', help='gain of excitatory cells',type=float, default=1.0)
    parser.add_argument('--gain_i', '-gi', help='gain of inhibitory cells',type=float, default=2.0)
    parser.add_argument('--wii_sum', '-wii', help='max sum of wii',type=float, default=0.25)
    parser.add_argument('--hebb_wii', '-hii', help='whether wii has Hebbian learning rule',type=int, default=0)
    parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
    parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=60)
    parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=6)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_e = int(args['n_e'])
    n_i = int(args['n_i'])
    init_iter = int(args['init_iter'])
    batch_iter = int(args['batch_iter'])
    max_iter = int(args['max_iter'])
    gain_e = args['gain_e']
    gain_i = args['gain_i']
    wii_sum = args['wii_sum']
    hebb_wii = int(args['hebb_wii']) > 0
    seed = int(args['seed'])
    n_wave = int(args['n_wave'])
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
    
    #--------------------------------------------------------------------------
    # Make SBTACH
    inpath = currwd + "/sim_lgn_wave_rfs.py"
    c1 = "{:s} -ne {:d} -ni {:d} -iit {:d} -bit {:d} -mit {:d} -s {:d} -nw {:d} -ng {:d} -ge {:.1f} -gi {:.1f} -wii {:.2f} -hii {:d}".format(
        inpath,n_e,n_i,init_iter,batch_iter,max_iter,seed,n_wave,n_grid,gain_e,gain_i,wii_sum,hebb_wii)
    
    jobname="{:s}".format('sim_lgn_wave_rfs_s_{:d}_n_{:d}_ge_{:.1f}_gi_{:.1f}_wii_{:.2f}_hii_{:d}'.format(
        seed,init_iter,gain_e,gain_i,wii_sum,hebb_wii))
    
    if not args2.test:
        jobnameDir=os.path.join(ofilesdir, jobname)
        text_file=open(jobnameDir, "w");
        os. system("chmod u+x "+ jobnameDir)
        text_file.write("#!/bin/sh \n")
        if cluster=='haba' or cluster=='moto' or cluster=='burg':
            text_file.write("#SBATCH --account=theory \n")
        text_file.write("#SBATCH --job-name="+jobname+ "\n")
        text_file.write("#SBATCH -t 0-01:59  \n")
        text_file.write("#SBATCH --mem-per-cpu={:d}gb \n".format(gb))
        text_file.write("#SBATCH --gres=gpu\n")
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


