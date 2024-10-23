import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,hsv_to_rgb
from hsluv import hpluv_to_rgb

def hpluv_to_rgb_vec(H,S,V):
    RGB = np.zeros(H.shape + (3,))
    it = np.nditer(H,flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        RGB[idx] = hpluv_to_rgb((360*H[idx],100*S[idx],100*V[idx]))
    return RGB

hue_cmap = ListedColormap(hpluv_to_rgb_vec(np.linspace(0,1,100),np.ones(100),0.7*np.ones(100)))
lit_cmap = ListedColormap(hpluv_to_rgb_vec(np.zeros(100),np.zeros(100),np.linspace(0,1,100)))

def ytitle(ax,text,xloc=-0.25,**kwargs):
    ax.text(xloc,0.5,text,horizontalalignment='left',verticalalignment='center',
        rotation='vertical',transform=ax.transAxes,**kwargs)
    
def imshowticks(ax,xvals,yvals,xskip=1,yskip=1,xfmt=None,yfmt=None):
    if xfmt is None:
        xfmt = '{:.1f}'
    if yfmt is None:
        yfmt = '{:.1f}'
    ax.set_xticks(np.arange(len(xvals))[::xskip],[xfmt.format(xval) for xval in xvals[::xskip]])
    ax.set_yticks(np.arange(len(yvals))[::yskip],[yfmt.format(yval) for yval in yvals[::yskip]])

def imshowbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(plot, cax=cax, orientation='vertical')

def doubimsh(fig,ax,A1,A2,hide_ticks=True,cmap_name='RdBu_r',vmin=0,vmax=1,**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    data1 = (A1-vmin)/(vmax-vmin)
    data2 = (A2-vmin)/(vmax-vmin)
    data1 = np.clip(data1,0,1)
    data2 = np.clip(data2,0,1)
    data1 = 0.9*data1 + 0.05
    data2 = 0.9*data2 + 0.05
    cmap = mpl.colormaps[cmap_name](data1)
    cmap += mpl.colormaps[cmap_name](data2)
    cmap = 0.5*cmap
    cmap[...,-1] = 1
    plot = ax.imshow(cmap,**kwargs)

def doubimshbar(fig,ax,A1,A2,hide_ticks=True,cmap_name='RdBu_r',vmin=0,vmax=1,**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    data1 = (A1-vmin)/(vmax-vmin)
    data2 = (A2-vmin)/(vmax-vmin)
    data1 = np.clip(data1,0,1)
    data2 = np.clip(data2,0,1)
    data1 = 0.9*data1 + 0.05
    data2 = 0.9*data2 + 0.05
    cmap = mpl.colormaps[cmap_name](data1)
    cmap += mpl.colormaps[cmap_name](data2)
    cmap = 0.5*cmap
    cmap[...,-1] = 1
    plot = ax.imshow(cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mpl.cm.ScalarMappable(
                 norm=mpl.colors.Normalize(vmin=vmin-0.05*(vmax-vmin),vmax=vmax+0.05*(vmax-vmin)),cmap=cmap_name),
                 cax=cax,orientation='vertical')
    
def contourbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A.shape[1]),np.arange(A.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot = ax.contour(x,y,A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(plot, cax=cax, orientation='vertical')
    
def doubcont(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A1.shape[1]),np.arange(A1.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot1 = ax.contour(x,y,A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(x,y,A2,cmap=cmap,**kwargs)
    
def doubcontbar(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A1.shape[1]),np.arange(A1.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot1 = ax.contour(x,y,A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(x,y,A2,cmap=cmap,**kwargs)
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
    S = 0.05 + 0.9 * np.cos(0.5 * np.pi * r)
    V = 0.05 + 0.9 * np.sin(0.5 * np.pi * r)
    # rgb = hsv_to_rgb(np.dstack((H,S,V)))
    rgb = hpluv_to_rgb_vec(H,np.ones_like(H),r)#S,V)
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
        rlim[1] = 1.1*np.max(r)
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
    rgb = hpluv_to_rgb_vec(H,np.ones_like(H),r)#S,V)
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
