import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshowbar(fig,ax,A,**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')
    
def contourbar(fig,ax,A,**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot = ax.contour(A,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')
    
def doubcontbar(fig,ax,A1,A2,**kwargs):
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plot1 = ax.contour(A1,**kwargs)
    plot2 = ax.contour(A2,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot1, cax=cax, orientation='vertical')