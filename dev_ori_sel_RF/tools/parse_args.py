import argparse
from bettina.modeling.ori_dev_model import data_dir


## ==================Optional Input Arguments ===========================
parser = argparse.ArgumentParser()
parser.add_argument("--V", dest='V', required=False, type=int)
parser.add_argument("--not_saving_temporal_data",dest="not_saving_temp",\
					nargs="?", const=True, default=False,type=bool)## deprecated

## spatial ranges
parser.add_argument("--rA", dest='r_A', required=False, type=float, default=None, nargs="+")
parser.add_argument("--sI", dest='sI', required=False, type=float)
parser.add_argument("--rC", dest='rC', required=False, type=float)

## learning/plasticity
parser.add_argument("--norm", dest='normalisation_mode_lgn', required=False, type=str,\
					help="str, 'xalpha', 'x' or 'alpha' ")
parser.add_argument("--mult_norm", dest="multiplicative_normalisation_lgn", required=False,\
					type=str)
parser.add_argument("--betaP", dest='beta_P', required=False, type=float, nargs="+")
parser.add_argument("--gamma_lgn", dest='gamma_lgn', required=False, type=float)
parser.add_argument("--runtime", dest='runtime', required=False, type=float)
parser.add_argument("--sim", dest='simulate_activity',\
					choices=['dynamics','antolik_etal','steady_state'],required=False, type=str)
parser.add_argument("--patt_dur", dest='pattern_duration', required=False, type=int)
parser.add_argument("--avg_no_inp", dest='avg_no_inp', required=False, type=int)

## recurrent network layout
parser.add_argument("--nonlin", dest='nonlinearity_l4', required=False, type=str,\
					choices=[ 'rectifier', 'linear', 'powerlaw'])
parser.add_argument("--wrec", dest='Wrec_mode', required=False, type=str)
parser.add_argument("--w4to4_ew", dest='w4to4_ew', required=False, type=float)
parser.add_argument("--w4to23", dest='W4to23_mode', required=False, type=str)
parser.add_argument("--w23to4", dest='W23to4_mode', required=False, type=str)
parser.add_argument("--w23to23", dest='W23to23_mode', required=False, type=str)
parser.add_argument("--density", dest='density', required=False, type=float)
parser.add_argument("--ncluster", dest='ncluster', required=False, type=int)
parser.add_argument("--Nvert", dest='Nvert', required=False, type=int)
parser.add_argument("--noise_rec", dest='noise_rec', required=False, type=float)
parser.add_argument("--aEE", help="ampl of rec E to E in L4", dest='aEE', required=False,\
					type=float)
parser.add_argument("--rec_conn_het", dest='rec_conn_het',required=False,type=str)
parser.add_argument("--rec_conn_mean_ecc", dest='rec_conn_mean_ecc',required=False,type=float)
parser.add_argument("--w4to23_plastic",dest="W4to23_plastic",required=False,nargs="?",\
					const=True,default=False,type=bool)
parser.add_argument("--w4to23_width",help="multiplic factor to width of L4 to L23 conn",\
					dest="w4to23_width",required=False,type=float)
parser.add_argument("--w23to4_ampl", help="multiplic factor to ampl of L23 to L4 conn",\
					dest="w23to4_ampl",required=False,type=float)


## LGN input
parser.add_argument("--Wret_to_LGN_profile", dest='Wret_to_LGN_profile',\
					required=False, type=str)
parser.add_argument("--MH_strength", dest='MH_strength', required=False, type=float)
parser.add_argument("--onoff_rel_weight", dest='onoff_rel_weight', required=False,\
					 type=float, nargs="+")
parser.add_argument("--onoff_corr_factor", dest='onoff_corr_factor', required=False,\
					 type=float)
parser.add_argument("--off_bias_strength", dest='off_bias_strength', required=False,\
					 type=float,help="float, between 0 to 1", nargs="+")
parser.add_argument("--off_bias_region_size", dest='off_bias_region_size', required=False,\
					 type=float,help="float, between 0 to 1", nargs="+")
parser.add_argument("--lgn_corr_het", dest='lgn_corr_het',required=False,type=str)
parser.add_argument("--connectivity_type", dest='connectivity_type',required=False,type=str)
parser.add_argument("--Wlim", dest='Wlim', required=False, type=float)
parser.add_argument("--expanse_time", dest='expanse_time', required=False, type=int)


## ARBOR variations
parser.add_argument("--arbor_profile",dest='arbor_profile',required=False,type=str,\
					default=None,nargs="+")
parser.add_argument("--arbor_ampl",dest='arbor_ampl',required=False,type=float,default=None,\
					nargs="+")


parser.add_argument("--plasticity_rule", dest="plasticity_rule",choices=["activity_based",\
					"activity_corr"],required=False,type=str)
parser.add_argument("--test_lowDsubset", dest='test_lowDsubset',required=False,type=bool)

## saving tools
parser.add_argument("--saving_stepsize", dest='saving_stepsize',required=False,type=int)

## tools for scanning through versions
parser.add_argument("--index", dest='idx', help="specifying run number (either\
					 individual runs as x1 x2 x3 or endpoints of successive numbers of\
					 runs as x1:xn or x1-xn)", required=False,  nargs="*", default=None)
parser.add_argument("--cl", dest='load_external_from', help="specify where data lies\
					(e.g. None, aws, habanero)", required=False,  nargs="*", default=[""])


parser.add_argument("--load_params_file",dest="load_params_file",choices=["antolik_etal",\
					"default","GaussInpCorr","norecMH","onepop","twopop","onepop_dynamic"],\
					required=False,type=str)

args = parser.parse_args()



def get_data_dir():
	if args.load_external_from[0]=="":
		lookup_dir = data_dir+"layer4/"
	elif args.load_external_from[0]=="aws":
		lookup_dir = data_dir+"layer4/aws/"
	elif args.load_external_from[0]=="habanero":
		lookup_dir = "/media/bettina/TOSHIBA EXT/physics/columbia/projects/" +\
			 		 "ori_dev_model/data/layer4/habanero/"
	return lookup_dir