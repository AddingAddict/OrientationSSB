import numpy as np
import pickle
import os
import sys
from scipy import linalg
# from copy import copy

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dev_ori_sel_RF import data_dir,image_dir,inputs,connectivity
from dev_ori_sel_RF.tools import misc



def read_txtfile_into_dict(path_to_txtfile):
    txt_dict = {}
    key = None
    with open(path_to_txtfile,"r") as f:
        content = f.readlines()
        for item in content:
            fields = item.split("\t")
            if len(fields)==3:
                if key is None:
                    txt_dict[fields[1]] = fields[2].split("\n")[0]
                else:
                    txt_dict[key][fields[1]] = fields[2].split("\n")[0]
            else:
                key = fields[0].split("\n")[0]
                txt_dict[key] = {}

    return txt_dict


def scan_simulation_for_params(load_external_from, Versions, **kwargs):
    """
    scan all version numbers for parameter settings that match settings given in kwargs
    """
    matching_versions = (np.copy(Versions)).tolist()
    for Version in Versions:
        try:
            ## load parameters
            if load_external_from=="habanero":
                file_dir =\
                 "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/data/" +\
                 "layer4/habanero/v{v}/".format(v=Version)
                if not os.path.isfile(file_dir+"config_v{v}.p".format(v=Version)):
                     file_dir = data_dir + "layer4/habanero/pickle_files/"
            elif load_external_from=="aws":
                file_dir = data_dir + "layer4/aws/v{v}/".format(v=Version)
            else:
                file_dir = data_dir + "layer4/v{v}/".format(v=Version)
            params = pickle.load(open(file_dir + "config_v{v}.p".format(v=Version),"rb"))
            # if habanero:
            # 	if os.path.isfile(data_dir+\
            # 		"layer4/habanero/pickle_files/config_v{v}.p".format(v=Version)):
            # 		params = pickle.load(open(data_dir +\
            # 		 "layer4/habanero/pickle_files/config_v{v}.p".format(v=Version),"rb"))
            # 	else:
            # 		params = read_txtfile_into_dict(data_dir +\
            # 		 "layer4/habanero/config_v{v}.txt".format(v=Version))
            # else:
            # 	params = pickle.load(open(data_dir + "layer4/v{v}/config_v{v}.p".format(\
            # 		v=Version),"rb"))

            for key, value in kwargs.items():
                if Version not in matching_versions:
                    continue

                if key in params.keys():
                    if params[key] != value:
                        matching_versions.remove(Version)
                        continue

                elif key in params["Wret_to_lgn_params"].keys():
                    if params["Wret_to_lgn_params"][key] != value:
                        matching_versions.remove(Version)
                        continue

                elif key in params["W4to4_params"].keys():
                    if params["W4to4_params"][key] != value:
                        matching_versions.remove(Version)
                        continue

                elif key in params["Inp_params"].keys():
                    if params["Inp_params"][key] != value:
                        matching_versions.remove(Version)
                        continue

                elif key in params["Wlgn_to4_params"].keys():
                    if params["Wlgn_to4_params"][key] != value:
                        matching_versions.remove(Version)

        except Exception as e:
            # print(e)
            # print("Skipping Version={}".format(Version))
            matching_versions.remove(Version)

    return np.sort(matching_versions).tolist()


def compare_simulation_params(load_external_from, RefVersion, Versions):
    """
    compare simulation parameters between RefVersion and list of Versions
    """
    try:
        if load_external_from=="habanero":
            Refparams = read_txtfile_into_dict(data_dir +\
                "layer4/habanero/config_v{v}.txt".format(v=RefVersion))
        else:
            Refparams = pickle.load(open(data_dir + "layer4/v{v}/config_v{v}.p".format(\
                v=RefVersion),"rb"))

        # if not isinstance(Versions,list):
        # 	print("oh",Versions)
        # 	Versions = [Versions,]

        for Version in Versions:
            try:
                if load_external_from=="habanero":
                    # params = read_txtfile_into_dict(data_dir +\
                    # 		 "layer4/habanero/config_v{v}.txt".format(v=Version))
                    try:
                        params = read_txtfile_into_dict(data_dir +\
                                "layer4/habanero/config_v{v}.txt".format(v=Version))
                    except:
                        path = "/media/bettina/TOSHIBA EXT/physics/columbia/projects/"+\
                                 "ori_dev_model/data/layer4/habanero/v{v}/config_v{v}.p".format(\
                                 v=Version)
                        if os.path.isfile(path):
                            params = pickle.load(open(path,"rb"))
                elif load_external_from=="aws":
                    params = pickle.load(open(data_dir +\
                                 "layer4/aws/v{v}/config_v{v}.p".format(\
                                 v=Version),"rb"))
                else:
                    params = pickle.load(open(data_dir + "layer4/v{v}/config_v{v}.p".format(\
                        v=Version),"rb"))
                print("")
                print(" *** *** *** Start comparison between V{} and V{} *** *** *** ".format(\
                    RefVersion,Version))
                ## comparison
                for key,value in sorted(params.items()):
                    if key in Refparams.keys():
                        if isinstance(value,dict):
                            for jkey,jvalue in value.items():
                                if jkey in Refparams[key].keys():
                                    if jvalue==Refparams[key][jkey]:
                                        continue
                                    else:
                                        print("")
                                        print("Key {} = {} in {} in RefVersion {}.".format(\
                                            jkey,Refparams[key][jkey],key,RefVersion))
                                        print("{} = {} in {} in Version {}".format(jkey,\
                                            jvalue,key,Version))
                                else:
                                    print("")
                                    print("Key {} not found in {} in RefVersion {}.".format(\
                                        jkey,key,RefVersion))
                                    print("{} = {} in {} in Version {}".format(jkey,\
                                        jvalue,key,Version))
                        else:
                            if value==Refparams[key]:
                                continue
                            else:
                                print("")
                                print("Key {} = {} in RefVersion {}.".format(key,\
                                    Refparams[key],RefVersion))
                                print("{} = {} in Version {}".format(key,value,Version))
                    else:
                        print("")
                        print("Key {} not found in RefVersion {}.".format(key,RefVersion))
                        print("{} = {} in Version {}".format(key,value,Version))

                print("")
            except Exception as e:
                # print(e)
                # print("Skipping Version={}".format(Version))
                pass

    except Exception as e:
        # print(e)
        # print("Skipping Version={}".format(RefVersion))
        pass


def range_of_params(load_external_from, Versions, labels=None, print_out=True, *args):
    """
    gives range of paraemters given in args used in list of Versions
    """
    param_range = {}
    for i,arg in enumerate(args):
        print("arg",arg)

        arg_vals = []
        used_versions = []
        for Version in Versions:
            try:
                if load_external_from=="habanero":
                    # params = read_txtfile_into_dict(data_dir +\
                    # 		 "layer4/habanero/config_v{v}.txt".format(v=Version))
                    try:
                        params = pickle.load(open(data_dir +\
                                 "layer4/habanero/pickle_files/config_v{v}.p".format(\
                                 v=Version),"rb"))
                    except:
                        path = "/media/bettina/TOSHIBA EXT/physics/columbia/projects/"+\
                                 "ori_dev_model/data/layer4/habanero/v{v}/config_v{v}.p".format(\
                                 v=Version)
                        if os.path.isfile(path):
                            params = pickle.load(open(path,"rb"))
                elif load_external_from=="aws":
                    params = pickle.load(open(data_dir +\
                                 "layer4/aws/v{v}/config_v{v}.p".format(\
                                 v=Version),"rb"))
                else:
                    params = pickle.load(open(data_dir + "layer4/v{v}/config_v{v}.p".format(\
                            v=Version),"rb"))

                if isinstance(arg,list):
                    if arg[0] in params.keys():
                        if arg[1] in params[arg[0]].keys():
                            if isinstance(params[arg[0]][arg[1]],float):
                                value = np.around(params[arg[0]][arg[1]],3)
                            else:
                                value = params[arg[0]][arg[1]]

                            arg_vals.append(value)
                            used_versions.append(Version)
                else:
                    if arg in params.keys():
                        if isinstance(params[arg],float):
                            value = np.around(params[arg],3)
                        else:
                            value = params[arg]
                        arg_vals.append(params[arg])
                        used_versions.append(Version)
            except:
                if print_out:
                    misc.PrintException(Version)
                    print("Skipping Version {}".format(Version))
                pass

        arg_vals = np.array(arg_vals)
        used_versions = np.array(used_versions)
        uni_vals = np.unique(arg_vals)
        if print_out:
            print("Variable {} has been varied over: ".format(arg),uni_vals)
            if len(uni_vals)>1:
                for uni_val in uni_vals:
                    print("{}={} in Versions: ".format(arg,uni_val),\
                                                    used_versions[arg_vals==uni_val])
            print("")

        # param_range[arg[0]] = {}
        if labels is None:
            if isinstance(arg,list):
                param_range[arg[0] + "/" + arg[1]] = uni_vals
            else:
                param_range[arg] = uni_vals
        else:
            param_range[labels[i]] = uni_vals

    if print_out:
        print("param_range",param_range)
    return param_range


def get_all_parameters(params_dict,keys,key_word):
    values = []
    for key in keys:
        Nlgn = params_dict[key]["Nlgn"]
        N4 = params_dict[key]["N4"]
        if key_word=="lgn_corr_length":
            x = params_dict[key]["Wret_to_lgn_params"]["sigma"]*Nlgn
        elif key_word=="intracortical_length":
            x = params_dict[key]["W4to4_params"]["sigma_factor"]*\
                params_dict[key]["W4to4_params"]["sigma_EE"]*N4
        elif key_word=="onoff_input_factor":
            try:
                x = params_dict[key]["Inp_params"]["onoff_rel_weight"]
            except:
                x = 1
        values.append(x)
    return values



if __name__=="__main__":
    import argparse
    import re

    ## ==================Optional Input Arguments ===========================
    parser = argparse.ArgumentParser(description="Plot results from simulation.")
    parser.add_argument("--index", dest='idx', help="specifying run number (either\
     individual runs as x1 x2 x3 or endpoints of successive numbers of runs as x1:xn)",\
     required=False,  nargs="*", default=None)
    parser.add_argument("--cl", dest='load_external_from', help="specify where data lies\
        (e.g. None, aws, habanero)", required=False,  nargs="*", default=[""])

    args = parser.parse_args()
    args_dict = vars(args)

    load_external_from = args.load_external_from[0]


    Versions = np.array([],dtype=int)
    ## no arguments given, index falls back to default value
    if isinstance(args.idx,int):
        Versions = np.concatenate([Versions,np.array([args.idx])])
    elif args.idx is None:
        version_folders = os.listdir(data_dir + "layer4/")
        version_folders = np.array(version_folders)[np.array(["v" in string for string in version_folders])]
        Versions = [int(re.sub("v", "", string)) for string in version_folders]
    ## arguments given, either
    else:
        for item in args.idx:
            if ":" in item:
                version_list = item.split(":")
                assert version_list[0]!="",\
                 "Error: start and end value expected for list of indices"
                assert version_list[1]!="",\
                 "Error: start and end value expected for list of indices"
                Versions = np.concatenate([Versions,np.arange(int(version_list[0]),\
                    int(version_list[1])+1,1)])
            elif "-" in item:
                version_list = item.split("-")
                assert version_list[0]!="",\
                 "Error: start and end value expected for list of indices"
                assert version_list[1]!="",\
                 "Error: start and end value expected for list of indices"
                Versions = np.concatenate([Versions,np.arange(int(version_list[0]),\
                    int(version_list[1])+1,1)])
            else:
                assert isinstance(int(item),int), "Error: int value expected for index"
                Versions = np.concatenate([Versions,np.array([int(item)])])


    print("Versions",Versions)
    print("")

    ## =========== FIND VERSION WITH SPECIFIC PARAM SETTINGS ==================
    matching_versions = scan_simulation_for_params(load_external_from, Versions,\
                                                     saving_stepsize=1)
    print("{} Matching versions found are: ".format(len(matching_versions)), matching_versions)

    ## =========== COMPARE PARAM SETTINGS BETWEEN VERSIONS ===================
    # path_to_txtfile = data_dir + "layer4/habanero/config_v{v}.txt".format(v=405)
    # read_txtfile_into_dict(path_to_txtfile)
    # RefVersion = Versions[0]
    # compare_simulation_params(load_external_from, RefVersion, Versions[1:])


    ## =========== FIND PARAMETER RANGE =====================================
    # matching_versions = Versions
    # ## possible arguments:
    # ## ["W4to4_params","ncluster"]
    # ## ["Wret_to_lgn_params","ampl2"]
    # ## ["Wlgn_to4_params","r_A"]
    # ## ["Inp_params","Nvert"]
    # ## ["Wret_to_lgn_params","sigma"]
    # range_of_params(load_external_from, matching_versions, None,\
    # #  ["Wret_to_lgn_params","sigma2"],\
    #  ["Wret_to_lgn_params","sigma"],\
    # #  ["Inp_params","Nvert"],\
    # #  ["Wlgn_to4_params","r_A"],\
    #  # ["Wret_to_lgn_params","ampl2"],\
    #  # ["W4to4_params","density"],\
    #  )
