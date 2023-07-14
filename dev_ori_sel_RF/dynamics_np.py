import numpy as np


def fio_lin(x):
    return x
def fio_rect(x):
    x[x<0] = 0
    return x
def fio_powerlaw(x):
    x[x<0] = 0
    return x**2

## ######## dynamical equations with E/I input separated ########
def dynamics_rec_sep_inp(y,Wrec,inp_ff,Wff,inp_fb,Wfb,gamma_rec,gamma_ff):
    N = y.size//2
    exc_rec_inp = gamma_rec * np.dot(Wrec[:,:N],y[:N])
    inh_rec_inp = gamma_rec * np.dot(Wrec[:,N:],y[N:])

    exc_fb_inp = np.dot(Wfb[:,:N],inp_fb[:N])
    inh_fb_inp = np.dot(Wfb[:,N:],inp_fb[N:])

    exc_ff_inp = gamma_ff * (np.dot(Wff[0,:,:],inp_ff[0,:]) + np.dot(Wff[1,:,:],inp_ff[1,:]))
    return exc_rec_inp,inh_rec_inp,exc_fb_inp,inh_fb_inp,exc_ff_inp

def dynamics_l4_fullinput_sep_inp(y,Wrec, inp_ff,Wff, inp_fb,Wfb, gamma_rec,gamma_ff,N,tau=1,\
    fio=fio_powerlaw):
    exc_rec_inp_toE,inh_rec_inp_toE,exc_fb_inp_toE,inh_fb_inp_toE,exc_ff_inp_toE =\
        dynamics_rec_sep_inp(y,Wrec[:N,:],inp_ff,Wff[:2,:,:],inp_fb,\
                             Wfb[:N,:],gamma_rec,gamma_ff)
    argE = exc_fb_inp_toE + inh_fb_inp_toE + exc_ff_inp_toE + exc_rec_inp_toE +\
           inh_rec_inp_toE

    exc_rec_inp_toI,inh_rec_inp_toI,exc_fb_inp_toI,inh_fb_inp_toI,exc_ff_inp_toI =\
        dynamics_rec_sep_inp(y,Wrec[N:,:],inp_ff,Wff[2:,:,:],inp_fb,\
                             Wfb[N:,:],gamma_rec,gamma_ff)
    argI = exc_fb_inp_toI + inh_fb_inp_toI + exc_ff_inp_toI + exc_rec_inp_toI +\
           inh_rec_inp_toI

    IO_dict = {
                "exc_rec_inp_toE" : exc_rec_inp_toE,\
                "inh_rec_inp_toE" : inh_rec_inp_toE,\
                "exc_ff_inp_toE" : exc_ff_inp_toE,\
                "exc_fb_inp_toE" : exc_fb_inp_toE,\
                "inh_fb_inp_toE" : inh_fb_inp_toE,\
                "exc_rec_inp_toI" : exc_rec_inp_toI,\
                "inh_rec_inp_toI" : inh_rec_inp_toI,\
                "exc_ff_inp_toI" : exc_ff_inp_toI,\
                "exc_fb_inp_toI" : exc_fb_inp_toI,\
                "inh_fb_inp_toI" : inh_fb_inp_toI,\
    }
    return 1./tau*( -y + fio(np.concatenate([argE,argI]))), IO_dict

def dynamics_l23_sep_inp(y,Wrec,inp_ff,Wff,N,tau=1,fio=fio_powerlaw):
    exc_rec_inp = np.dot(Wrec[:,:N],y[:N])
    inh_rec_inp = np.dot(Wrec[:,N:],y[N:])

    exc_ff_inp = np.dot(Wff[:,:N],inp_ff[:N])
    inh_ff_inp = np.dot(Wff[:,N:],inp_ff[N:])
    arg = exc_rec_inp + inh_rec_inp + exc_ff_inp + inh_ff_inp
    # print("inh_ff_inp[:N]",inh_ff_inp[:N])
    IO_dict = {
                "exc_rec_inp_toE" : exc_rec_inp[:N],\
                "inh_rec_inp_toE" : inh_rec_inp[:N],\
                "exc_ff_inp_toE" : exc_ff_inp[:N],\
                "inh_ff_inp_toE" : inh_ff_inp[:N],\

                "exc_rec_inp_toI" : exc_rec_inp[N:],\
                "inh_rec_inp_toI" : inh_rec_inp[N:],\
                "exc_ff_inp_toI" : exc_ff_inp[N:],\
                "inh_ff_inp_toI" : inh_ff_inp[N:],\
    }
    return 1./tau*( -y + fio(arg)), IO_dict


def dynamics_twolayer_fullinput_sep_inp(y,inp_ff,Wff,W4to4,W4to23,W23to23,W23to4,gamma_rec,\
    gamma_ff,N4,N23,tau=1,fio=fio_powerlaw):
    l4 = y[:N4*2]
    l23 = y[2*N4:]
    dl4,IO_dict_4 = dynamics_l4_fullinput_sep_inp(l4,W4to4,inp_ff,Wff,l23,W23to4,\
                                                   gamma_rec,gamma_ff,N4,tau,fio)
    dl23,IO_dict_23 = dynamics_l23_sep_inp(l23,W23to23,l4,W4to23,N23,tau,fio)
    IO_dict = {}
    for key,value in IO_dict_4.items():
        IO_dict[key+"_4"] = value
    for key,value in IO_dict_23.items():
        IO_dict[key+"_23"] = value
    return np.concatenate([dl4,dl23]), IO_dict



########################################################################################


def dynamics_rec(y,Wrec,inp_ff,Wff,inp_fb,Wfb,gamma_rec,gamma_ff):
    arg_rec = gamma_rec * np.dot(Wrec,y)
    arg_fb = np.dot(Wfb,inp_fb)
    arg_ff = gamma_ff * (np.dot(Wff[0,:,:],inp_ff[0,:]) + np.dot(Wff[1,:,:],inp_ff[1,:]))
    # arg = arg_rec + arg_fb + arg_ff
    return arg_rec, arg_fb, arg_ff

def dynamics_l4(y,Wrec, inp_ff,Wff, inp_fb,Wfb, gamma_rec,gamma_ff,N,tau=1,fio=fio_powerlaw):
    argE = gamma_rec * np.dot(Wrec[:N,:],y) + np.dot(Wfb[:N,:],inp_fb) +\
           gamma_ff * (np.dot(Wff[0,:,:],inp_ff[0,:]) + np.dot(Wff[1,:,:],inp_ff[1,:]))
    argI = gamma_rec * np.dot(Wrec[N:,:],y) + np.dot(Wfb[N:,:],inp_fb)
    return 1./tau*( -y + fio(np.concatenate([argE,argI])) )

def dynamics_l4_fullinput(y,Wrec, inp_ff,Wff, inp_fb,Wfb, gamma_rec,gamma_ff,N,tau=1,\
    fio=fio_powerlaw):
    arg_recE, arg_fbE, arg_ffE = dynamics_rec(y,Wrec[:N,:],inp_ff,Wff[:2,:,:],inp_fb,\
                                              Wfb[:N,:],gamma_rec,gamma_ff)
    argE = arg_recE + arg_fbE + arg_ffE
    arg_recI, arg_fbI, arg_ffI = dynamics_rec(y,Wrec[N:,:],inp_ff,Wff[2:,:,:],inp_fb,\
                                              Wfb[N:,:],gamma_rec,gamma_ff)
    argI = arg_recI + arg_fbI + arg_ffI
    return 1./tau*( -y + fio(np.concatenate([argE,argI]))),arg_recE,arg_fbE,arg_ffE,\
            arg_recI,arg_fbI,arg_ffI

def dynamics_l23(y,Wrec,inp_ff,Wff,N,tau=1,fio=fio_powerlaw):
    arg_recE = np.dot(Wrec[:N,:],y)
    arg_ffE = np.dot(Wff[:N,:],inp_ff)
    argE = arg_recE + arg_ffE
    arg_recI = np.dot(Wrec[N:,:],y)
    arg_ffI = np.dot(Wff[N:,:],inp_ff)
    argI = arg_recI + arg_ffI
    return 1./tau*( -y + fio(np.concatenate([argE,argI]))), arg_recE, arg_ffE,\
            arg_recI, arg_ffI

def dynamics_twolayer(y,inp_ff,Wff,W4to4,W4to23,W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,\
    tau=1,fio=fio_powerlaw):
    l4 = y[:N4*2]
    l23 = y[2*N4:]
    dl4 = dynamics_l4(l4,W4to4,inp_ff,Wff,l23,W23to4, gamma_rec,gamma_ff,N4,tau,fio)
    dl23 = dynamics_l23(l23,W23to23,l4,W4to23,N23,tau,fio)
    return np.concatenate([dl4,dl23])

def dynamics_twolayer_fullinput(y,inp_ff,Wff,W4to4,W4to23,W23to23,W23to4,gamma_rec,gamma_ff,\
    N4,N23,tau=1,fio=fio_powerlaw):
    l4 = y[:N4*2]
    l23 = y[2*N4:]
    dl4,arg_recE,arg_fbE,arg_ffE,arg_recI,arg_fbI,arg_ffI = dynamics_l4_fullinput(l4,W4to4,\
                                                            inp_ff,Wff,l23,W23to4,\
                                                             gamma_rec,gamma_ff,N4,tau,fio)
    dl23,arg_recE_23,arg_ffE_23,arg_recI_23,arg_ffI_23 = dynamics_l23(l23,W23to23,l4,W4to23,\
                                                                        N23,tau,fio)
    IO_dict = {
               "arg_recE_23" : arg_recE_23,\
               "arg_ffE_23" : arg_ffE_23,\
               "arg_recI_23" : arg_recI_23,\
               "arg_ffI_23" : arg_ffI_23,\
               "arg_recE" : arg_recE,\
               "arg_fbE" : arg_fbE,\
               "arg_ffE" : arg_ffE,\
               "arg_recI" : arg_recI,\
               "arg_fbI" : arg_fbI,\
               "arg_ffI" : arg_ffI
               }
    return np.concatenate([dl4,dl23]), IO_dict



def dynamics_rec_onelayer(y,Wrec,inp_ff,Wff,gamma_rec,gamma_ff):
    arg = gamma_rec * np.dot(Wrec,y) + \
            gamma_ff * (np.dot(Wff[0,:,:],inp_ff[0,:]) + np.dot(Wff[1,:,:],inp_ff[1,:]))
    return arg

def dynamics_onelayer_fullinput(y,inp_ff,Wff,Wrec,gamma_rec,gamma_ff,N,tau=1,fio=fio_powerlaw):
    argE = dynamics_rec_onelayer(y,Wrec[:N,:],inp_ff,Wff[:2,:,:],gamma_rec,gamma_ff)
    argI = dynamics_rec_onelayer(y,Wrec[N:,:],inp_ff,Wff[2:,:,:],gamma_rec,gamma_ff)
    return 1./tau*( -y + fio(np.concatenate([argE,argI])))

def dynamics_onelayer(y,inp_ff,Wff,Wrec,gamma_rec,gamma_ff,N,tau=1,fio=fio_powerlaw):
    argE = dynamics_rec_onelayer(y,Wrec[:N,:],inp_ff,Wff[:2,:,:],gamma_rec,gamma_ff)
    argI = gamma_rec * np.dot(Wrec[N:,:],y)
    return 1./tau*( -y + fio(np.concatenate([argE,argI])))



