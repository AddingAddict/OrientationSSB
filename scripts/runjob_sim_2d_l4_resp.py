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
    parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=1)
    parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=1)
    parser.add_argument('--load_iter', '-lit', help='2d L4 kayser iteration number to load',type=int, default=50)
    parser.add_argument('--s_x', '-sx', help='feedforward arbor decay length',type=float, default=0.08)
    parser.add_argument('--s_e', '-se', help='excitatory recurrent arbor decay length',type=float, default=0.08)
    parser.add_argument('--s_i', '-si', help='inhibitory recurrent arbor decay length',type=float, default=0.08)
    parser.add_argument('--gain_i', '-gi', help='gain of inhibitory cells',type=float, default=2.0)
    parser.add_argument('--hebb_wei', '-hei', help='whether wei has Hebbian learning rule',type=int, default=0)
    parser.add_argument('--hebb_wii', '-hii', help='whether wii has Hebbian learning rule',type=int, default=0)
    parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
    parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=60)
    parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
    parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
    parser.add_argument('--f_vis', '-fv', help='spatial frequency of grating',type=float, default=5.0)
    parser.add_argument('--n_ori', '-no', help='number of grating orientations',type=int, default=4)
    parser.add_argument('--n_phs', '-np', help='number of grating phases',type=int, default=8)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=6)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_e = int(args['n_e'])
    n_i = int(args['n_i'])
    load_iter = int(args['load_iter'])
    s_x = args['s_x']
    s_e = args['s_e']
    s_i = args['s_i']
    gain_i = args['gain_i']
    hebb_wei = int(args['hebb_wei'])
    hebb_wii = int(args['hebb_wii'])
    seed = int(args['seed'])
    n_wave = int(args['n_wave'])
    n_stim = int(args['n_stim'])
    n_grid = int(args['n_grid'])
    f_vis = args['f_vis']
    n_ori = int(args['n_ori'])
    n_phs = int(args['n_phs'])
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
    inpath = currwd + "/sim_2d_l4_resp.py"
    c1 = "{:s} -ne {:d} -ni {:d} -lit {:d} -s {:d} -nw {:d} -ns {:d} -ng {:d} -fv {:.1f} -no {:d} -np {:d} -sx {:.2f} -se {:.2f} -si {:.2f} -gi {:.1f} -hei {:d} -hii {:d}".format(
        inpath,n_e,n_i,load_iter,seed,n_wave,n_stim,n_grid,f_vis,n_ori,n_phs,s_x,s_e,s_i,gain_i,hebb_wei,hebb_wii)
    
    jobname="{:s}".format('sim_2d_l4_resp_s_{:d}_n_{:d}_sx={:.2f}_se={:.2f}_si={:.2f}_gi={:.1f}_hei={:d}_hii={:d}'.format(
        seed,load_iter,s_x,s_e,s_i,gain_i,hebb_wei,hebb_wii))
    
    if not args2.test:
        jobnameDir=os.path.join(ofilesdir, jobname)
        text_file=open(jobnameDir, "w");
        os. system("chmod u+x "+ jobnameDir)
        text_file.write("#!/bin/sh \n")
        if cluster=='haba' or cluster=='moto' or cluster=='burg':
            text_file.write("#SBATCH --account=theory \n")
        text_file.write("#SBATCH --job-name="+jobname+ "\n")
        text_file.write("#SBATCH -t 0-03:59  \n")
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


