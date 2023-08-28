import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,hsv_to_rgb
from hsluv import hsluv_to_rgb

def hsluv_to_rgb_vec(H,S,V):
    RGB = np.zeros(H.shape + (3,))
    it = np.nditer(H,flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        RGB[idx] = hsluv_to_rgb((360*H[idx],100*S[idx],100*V[idx]))
    return RGB

hue_cmap = ListedColormap(hsluv_to_rgb_vec(np.linspace(0,1,100),np.ones(100),0.5*np.ones(100)))
lit_cmap = ListedColormap(hsluv_to_rgb_vec(np.zeros(100),np.zeros(100),np.linspace(0,1,100)))
    
def ytitle(ax,text,xloc=-0.25,**kwargs):
    ax.text(xloc,0.5,text,horizontalalignment='left',verticalalignment='center',
        rotation='vertical',transform=ax.transAxes,**kwargs)

def imshowbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(plot, cax=cax, orientation='vertical')

def doubimshbar(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cmap1 = plt.get_cmap(cmap, 24)
    cmap2 = plt.get_cmap(cmap, 24)
    cmap1._init()
    cmap2._init()
    alpha = np.concatenate((np.linspace(0, 0.8, cmap2.N//2)[::-1],np.linspace(0, 0.8, cmap1.N//2),np.ones(3)))
    cmap1._lut[:,-1] = alpha
    cmap2._lut[:,-1] = alpha
    plot1 = ax.imshow(A1,cmap=cmap1,**kwargs)
    plot2 = ax.imshow(A2,cmap=cmap2,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot1, cax=cax, orientation='vertical')
    
def contourbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot = ax.contour(A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(plot, cax=cax, orientation='vertical')
    
def doubcontbar(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot1 = ax.contour(A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(A2,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot1, cax=cax, orientation='vertical')

def domcol(fig,ax,A,hide_ticks=True,rlim=None,alim=None,**kwargs):
    H = np.angle(A)/(2*np.pi) + 0.5
    # r = np.log2(1. + np.abs(A))
    # S = 0.5 * (1. + np.abs(np.sin(2. * np.pi * r)))
    # V = 0.5 * (1. + np.abs(np.cos(2. * np.pi * r)))
    r = np.abs(A)
    if rlim is None:
        rlim = [None,None]
        rlim[0] = np.min(r)
        rlim[1] = np.max(r)
    elif rlim[0] is None:
        rlim[0] = np.min(r)
    elif rlim[1] is None:
        rlim[1] = np.max(r)
    if alim is None:
        alim = [None,None]
        alim[0] = -np.pi
        alim[1] = np.pi
    r = (r - rlim[0]) / (rlim[1] - rlim[0])
    S = 0.05 + 0.9 * np.sin(0.5 * np.pi * r)
    V = 0.05 + 0.9 * np.cos(0.5 * np.pi * r)
    # rgb = hsv_to_rgb(np.dstack((H,S,V)))
    rgb = hsluv_to_rgb_vec(H,np.ones_like(H),1-r)#S,V)
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.imshow(rgb,**kwargs)

def domcolbar(fig,ax,A,hide_ticks=True,rlim=None,alim=None,**kwargs):
    H = np.angle(A)/(2*np.pi) + 0.5
    # r = np.log2(1. + np.abs(A))
    # S = 0.5 * (1. + np.abs(np.sin(2. * np.pi * r)))
    # V = 0.5 * (1. + np.abs(np.cos(2. * np.pi * r)))
    r = np.abs(A)
    if rlim is None:
        rlim = [None,None]
        rlim[0] = np.min(r)
        rlim[1] = np.max(r)
    elif rlim[0] is None:
        rlim[0] = np.min(r)
    elif rlim[1] is None:
        rlim[1] = np.max(r)
    if alim is None:
        alim = [None,None]
        alim[0] = -np.pi
        alim[1] = np.pi
    r = (r - rlim[0]) / (rlim[1] - rlim[0])
    S = 0.05 + 0.9 * np.cos(0.5 * np.pi * r)
    V = 0.05 + 0.9 * np.sin(0.5 * np.pi * r)
    # rgb = hsv_to_rgb(np.dstack((H,S,V)))
    rgb = hsluv_to_rgb_vec(H,np.ones_like(H),r)#S,V)
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cbars = [None,None]
    plot = ax.imshow(np.angle(A),cmap=hue_cmap,vmin=alim[0],vmax=alim[1],**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbars[0] = fig.colorbar(plot, cax=cax, orientation='vertical')
    plot = ax.imshow(np.abs(A),cmap=lit_cmap,vmin=rlim[0],vmax=rlim[1],**kwargs)
    # divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.55)
    cbars[1] = fig.colorbar(plot, cax=cax, orientation='vertical')
    ax.imshow(rgb,**kwargs)
    return cbars
