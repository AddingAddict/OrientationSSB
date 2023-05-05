import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshowbar(fig,ax,A,cmap='RdBu',**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

def doubimshbar(fig,ax,A1,A2,cmap='RdBu',**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cmap1 = mpl.cm.get_cmap(cmap, 24)
    cmap2 = mpl.cm.get_cmap(cmap, 24)
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
    
def contourbar(fig,ax,A,cmap='RdBu',**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot = ax.contour(A,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')
    
def doubcontbar(fig,ax,A1,A2,cmap='RdBu',**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot1 = ax.contour(A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(A2,cmap=cmap,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot1, cax=cax, orientation='vertical')