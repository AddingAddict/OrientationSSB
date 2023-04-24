import os


## ============================ Paths =====================================
home_dir = os.environ["HOME"]
current_user = os.environ["USER"]
if current_user=="bettina":
	base_dir = os.path.join(home_dir, "physics/columbia/projects/ori_dev_model/")
elif current_user=="hein":
	base_dir = os.path.join(home_dir, "ori_dev_model/")
elif current_user=="bh2757":
	cwd = os.getcwd()
	# base_dir = "/rigel/theory/users/bh2757/columbia/projects/ori_dev_model/"
	# base_dir = "/burg/theory/users/bh2757/columbia/projects/ori_dev_model/"
	# base_dir = cwd + "/projects/ori_dev_model/"
	base_dir = os.path.join(cwd,"projects/ori_dev_model")
elif current_user=="ubuntu":
	base_dir = "/home/ubuntu/ori_dev_model"
elif current_user=="tuannguyen":
	base_dir = os.path.join(home_dir, "OrientationSSB/dev_ori_sel_RF/")
elif current_user=="thn2112":
	home_dir = "/burg/theory/users/thn2112/"
	base_dir = os.path.join(home_dir, "OrientationSSB/dev_ori_sel_RF/")
elif current_user=="alex":
	base_dir = os.path.join(home_dir, "Desktop/Columbia/Ken Miller/Codes/SSB/OrientationSSB/dev_ori_sel_RF/")
data_dir = os.path.join(base_dir,'data/')
image_dir = os.path.join(base_dir,'image/')
movie_dir = os.path.join(base_dir,'movie/')


## ============================ Layer 2/3 =====================================
Ninp = 30
Inp_params = {
				"ndim"		:	25,
				"radius"	:	Ninp/10.,#number of wavelength in pattern
				"width"		:	Ninp/30.,

				## only do plasticity update after avg_no_inp patterns
				## simulating activity for pattern_duration timesteps
				# "simulate_activity"		:	True,
				# "pattern_duration"		:	2500,#in units of dt,
				# "avg_no_inp"			:	100,
				## do plasticity update after each stimulus
				## simulating activity for pattern_duration timesteps
				"simulate_activity"		:	"dynamics",
				"pattern_duration"		:	2500,#in units of dt,
				"avg_no_inp"			:	1,
				## only do plasticity update after avg_no_inp patterns
				## activity is Gaussian filtered lgn input
				# "simulate_activity"		:	False,
				# "pattern_duration"		:	1,#in units of dt,
				# "avg_no_inp"			:	2,
				# number of update steps are runtime /(pattern_duration*dt)


				"expanse_time"			:	1,
				"init_radius"			:	0.3,
				"mean_width"			:	0.3,
				"onoff_corr_factor"		:	1.0,
				"onoff_rel_weight"		:	1.0,
				"input_type"			:	"white_noise_online",

				"off_bias_strength"		:	0.0,
				"off_bias_region_size"	:	0.5,
}

WIto23_params = {
				"sigma"		:	0.1,
				"ampl"		:	1.0,
				"noise"		:	0.01
}

## ============================ Layer 4 =====================================
runtime = 20000  # in units of tau
N = 10
normalisation_mode = "xalpha"
random_seed = 19
if current_user=="hein":
	N = 25
	runtime = 2000
elif current_user=="bh2757":
	N = 25
	runtime = 100000#1250000#500000#15000
	# runtime = 5000## simulate_activity=False
elif current_user=="ubuntu":
	N = 20
	runtime = 100000#3000#1501#

Nret = N
Nlgn = N
N4 = N
N23 = N
Nvert = 1
Wret_to_lgn_params = {
				"sigma1"	:	0.07,
				"sigma2"	:	0.2,
				"ampl1"		:	1.0,
				"ampl2"		:	1.,
				"gamma_ret"	:	1.0,

				"noise"		:	0.0,

				"sigma"		:	0.045,#0.07,
				"ampl"		:	1.0,
				"profile"	:	"Gaussian",#default:"Gaussian"

				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"		:	0.6,
				"SD_eccentricity"		:	0.05,
				"SD_size"				:	0.05,
				"heterogeneity_type"	:	None,##"independent"#"smooth"
}
Wlgn_to4_params = {
				"sigma"					:	0.2,
				"ampl"					:	1.0,
				"noise"					:	0.0,
				"s_noise"				:	0.01,
				"Wlim"					:	12,
				"load_from_prev_run"	:	None,#None or 95
				"connectivity_type"		:	"E",#"E","EI"

				## arbor params
				"r_A_on"				:	0.15,
				"arbor_profile_on"		:	"gaussian",#"gaussian"#"heaviside"#"overlap"
				"ampl_on"				:	1.,
				"r_A_off"				:	0.15,
				"arbor_profile_off"		:	"gaussian",
				"ampl_off"				:	1.,

				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"		:	0.,
				"SD_eccentricity"		:	0.,
				"SD_size"				:	0.,
				"heterogeneity_type"	:	None,##"independent"#"smooth"

				# plasticity params
				"beta_P"				:	0.005,
				"learning_rate"			:	0.5,
				"plasticity_rule"		:	"activity_based",
				"constraint_mode"		:	"xalpha_approx",
				"mult_norm"				:	"x",

}
deltax = 1./N4
factor = 2*3.14159
W4to4_params = {
				## MH
				"sigma1"	:	0.16,
				"sigma2"	:	0.3,
				"ampl1"		:	1.0,
				"ampl2"		:	1.0,
				##Gaussian
				"sigma"		:	0.2,
				"ampl"		:	1.0,
				"noise"		:	0.0,
				##Gaussian2pop
				"sigma_EE"	:	0.2,
				"sigma_IE"	:	0.2,
				"sigma_EI"	:	0.15,
				"sigma_II"	:	0.075,
				"aEE"		:	11.4,#11.5,#11.4,
				"aIE"		:	11.5,#11.1,
				"aEI"		:	10.9,#11.1,
				"aII"		:	10.,

				# "sigma_EE"	:	0.2,
				# "sigma_IE"	:	0.2,#0.21,
				# "sigma_EI"	:	0.2,
				# "sigma_II"	:	0.2,#0.18,
				# "aEE"		:	2.5,#0.1,#10.4,#11.5,#11.4,
				# "aIE"		:	2.4,#0.38,#0.8,#11.,
				# "aEI"		:	1.3,#0.089,#10.9,
				# "aII"		:	1.0,#0.096,#10.,
				## 1.1,2.,0.9,1.
				## 0.044,0.042,0.023,0.018 (omega_e<omega_i<0)
				## 0.01,0.038,0.089,0.096 (omega_i>omega_e>0)
				## ahmadian, fig2 (different regimes):
				## A: 2.5,2.4,1.3,1. (e<i<0)
				## B: 2.5,4.7,1.3,1. (e<0<i)
				## C: 2.5,4.7,1.3,2.2 (0<e<i)
				## D: 0.8,3.6,1.3,5. (0<i<e)
				## E: 2.5,2.2,1.3,1. (i<e<0)

				## dina paper parameters
				# "sigma_EE"	:	3*deltax,
				# "sigma_IE"	:	6*deltax,
				# "sigma_EI"	:	2*deltax,
				# "sigma_II"	:	2*deltax,
				# "aEE"		:	0.072 * factor*3**2,
				# "aIE"		:	0.06 * factor*6**2,
				# "aEI"		:	0.0528 * factor*2**2,
				# "aII"		:	0.0288 * factor*2**2,

				"rA_E"		:	0.3,
				"rA_I"		:	0.3,
				
				"Wrec_mode"		:	"Gaussian2pop",#Gaussian
				"sigma_factor"	:	1.0,
				"max_ew"		:	0.95,#0.013,#
				## params for clustered/sparse Connectivity
				"density"		:	0.5,
				"ncluster"		:	3,

				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"	:	0.8,
				"SD_eccentricity"	:	0.1,
				"SD_size"			:	0.03,
				"heterogeneity_type":	None,##"independent"#"smooth"#None
				"smoothness_width"  :	0.05, 

				# plasticity params
				"nu_4"				:	3.,
				"beta_P"			:	0.1,
				"learning_rate"		:	0.5,
				"plasticity_rule"	:	None,
				"constraint_mode"	:	None,
				"mult_norm"			:	None,

				"firing_threshold"	:	"adaptive",
				"theta_4"			:	0.9,
				"l4_avg"			:	0.0,

}

W23_params = {	
				"ampl"		:	1.0,
				"noise"		:	0.0,
				
				"sigma_EE"	:	0.2,
				"sigma_IE"	:	0.2,
				"sigma_EI"	:	0.15,
				"sigma_II"	:	0.075,
				"aEE"		:	11.4,
				"aIE"		:	11.5,
				"aEI"		:	10.9,
				"aII"		:	10.,

				"rA_E"		:	0.3,
				"rA_I"		:	0.3,

				"Wrec_mode"		:	"Gaussian2pop",#Gaussian
				"sigma_factor"	:	0.2,
				"max_ew"		:	0.95,
				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"	:	0.85,
				"SD_eccentricity"	:	0.1,
				"SD_size"			:	0.05,
				"heterogeneity_type":	"independent",##None#"independent"#"smooth"
				"smoothness_width"  :	0.05,

				# plasticity params
				"nu_23"				:	1.,
				"beta_P"			:	0.1,
				"learning_rate"		:	0.2,
				"plasticity_rule"	:	None,
				"constraint_mode"	:	None,
				"mult_norm"			:	None,

				"firing_threshold"	:	0.0,
}

W4to23_params = {	
				"ampl"		:	1.0,
				"noise"		:	0.0,
				"sigma_EE"	:	0.1,
				"sigma_IE"	:	0.1,#0.21,
				"sigma_EI"	:	0.1,
				"sigma_II"	:	0.1,#0.18,
				"aEE"		:	0.9,#0.1,#10.4,#11.5,#11.4,
				"aIE"		:	0.9,#0.38,#0.8,#11.,
				"aEI"		:	0.0,#0.089,#10.9,
				"aII"		:	0.0,#0.096,#10.,

				"Wrec_mode"		:	"Gaussian2pop",#Gaussian
				"sigma_factor"	:	1.0,
				"max_ew"		:	"orig",
				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"	:	0.,
				"SD_eccentricity"	:	0.,
				"SD_size"			:	0.,
				"heterogeneity_type":	None,##"independent"#"smooth"
				"smoothness_width"  :	0.05, 

				"arbor_profile"		:	"gaussian",
				"s_noise"			:	0.2,
				"r_A"				:	0.3,

				# plasticity params
				"gain"				:	1.,
				"beta_P"			:	0.1,
				"learning_rate"		:	0.2,
				"plasticity_rule"	:	None,
				"constraint_mode"	:	None,
				"mult_norm"			:	None,
}		

W23to4_params = {	
				"ampl"		:	1.0,
				"noise"		:	0.0,
				"sigma_EE"	:	0.1,
				"sigma_IE"	:	0.1,#0.21,
				"sigma_EI"	:	0.2,
				"sigma_II"	:	0.18,#0.18,
				"aEE"		:	0.1,#0.1,#10.4,#11.5,#11.4,
				"aIE"		:	0.1,#0.38,#0.8,#11.,
				"aEI"		:	0.0,#0.089,#10.9,
				"aII"		:	0.0,#0.096,#10.,

				"Wrec_mode"		:	"Gaussian2pop",#Gaussian
				"sigma_factor"	:	1.0,
				"max_ew"		:	"orig",
				## params for heterogeneously varying anisotropic Connections
				"mean_eccentricity"	:	0.,
				"SD_eccentricity"	:	0.,
				"SD_size"			:	0.,
				"heterogeneity_type":	None,##"independent"#"smooth"
				"smoothness_width"  :	0.05, 
}

# developing_params = {
# 					"beta_P" : [0.005,0.001,10],
# 					"onoff_rel_weight" : [3.,1.,10],
# 					"off_bias_strength" : [0.3,0.,10],
# 					"off_bias_region_size" : [0.8,0.5,10],
# 					}
developing_params = ["learning_rate_l4","learning_rate_l23"]
config_dict = {
				"Wret_to_lgn_params"	:	Wret_to_lgn_params,\
				"Wlgn_to4_params"		:	Wlgn_to4_params,\
				"W4to4_params"			:	W4to4_params,\
				"W23_params"			:	W23_params,\
				"W4to23_params"			:	W4to23_params,\
				"W23to4_params"			:	W23to4_params,\
				"Inp_params"			:	Inp_params,\

				"Nret"					:	Nret,\
				"Nlgn"					:	Nlgn,\
				"N4"					:	N4,\
				"N23"					:	N23,\
				"Nvert"					:	Nvert,\

				"runtime"				:	runtime,\
				"dt"					:	0.05,\
				"tau"					:	1.,\
				"num_lgn_paths"			:	2 if Wlgn_to4_params["connectivity_type"]=="E" else 4,\

				"gamma_lgn"				:	0.9,#0.9,\
				"gamma_4"				:	1.0,\
				"gamma_23"				:	1.0,\
				"learning_rate_decay_const"	:	160000.,

				"saving_stepsize"		:	10,\
				
				"plasticity_rule"		:	"activity_based",#"activity_corr"

				"random_seed"			:	random_seed,\
				"nonlinearity_l4"		:	"rectifier",#"rectifier"#"linear"
				"nonlinearity_l23"		:	"rectifier",
				"integrator"			:	"Euler",#"RK4"
				"comments"				:	"  ",\

				"test_lowDsubset" 		:	False,

				"developing_params"		:	developing_params,\
				}





