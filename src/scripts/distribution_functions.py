"""
This file contains generic functions for plotting distributions of data
binned by galactic radius and z-height.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from utils import get_bin_centers
from _globals import ABSZ_BINS, GALR_BINS


def plot_multizone_mdfs(mzs, axs, col='[fe/h]', colors=[], label='VICE', 
                        galr_bins=GALR_BINS, absz_bins=ABSZ_BINS, 
                        titlepad=None, **kwargs):
    """
    Plot [Fe/H] MDFs from a VICE multizone run.
    
    Parameters
    ----------
    mzs : MultizoneStars object
        VICE multizone stars.
    axs : list of matplotlib.axes.Axes
        Column of axes to plot MDFs. Must have len(axs) == len(absz_bins) - 1.
    colors : list of strings or matplotlib colors, optional
        List of colors to code by Rgal bins. If len(colors) != len(galr_bins)-1,
        colors will be chosen automatically. The default is [].
    label : str, optional
        Title for this column of axes. The default is 'VICE'.
    galr_bins: list of floats, optional
    absz_bins: list of floats, optional
    titlepad: int, optional
        Vertical padding for axes title. The default is None.
    **kwargs passed to MultizoneStars.mdf()
    """
    if len(colors) != len(galr_bins) - 1:
        colors = [None for galr in galr_bins[:-1]]
    if len(axs) == len(absz_bins) - 1:
        for i, ax in enumerate(axs.flatten()):
            absz_lim = absz_bins[-(i+2):len(absz_bins)-i]
            for j in range(len(galr_bins)-1):
                galr_lim = galr_bins[j:j+2]
                subset = mzs.region(galr_lim, absz_lim)
                mdf, bin_edges = subset.mdf(col, **kwargs)
                ax.plot(get_bin_centers(bin_edges), mdf, 
                        color=colors[j], linewidth=1)
        # need y=0.0000001 to allow negative padding
        axs[0].set_title(label, pad=titlepad, y=1.0000001)
    else:
        raise ValueError('Mismatch between number of axes and |z|-height bins.')
    

def setup_axes(ncols=2, figure_width=3.25, xlabel='', xlim=None, 
               major_tick_spacing=1, galr_bins=GALR_BINS, absz_bins=ABSZ_BINS, 
               cbar_width=0.6, cmap='plasma_r', panel_aspect_ratio=1.5,
               major_minor_tick_ratio=5., include_yaxis=False):
    """
    Set up matplotlib figure and axes for the distribution plot.
    
    Parameters
    ----------
    ncols : int, optional
        Number of columns in figure. The default is 2.
    figure_width : float, optional
        Width of the figure in inches. For an AASTeX document, this should be
        3.25 for a single-column figure or 7.0 for a double-column figure. The
        figure height will be scaled to maintain a consistent aspect ratio.
        The default is 3.25.
    xlabel : str, optional
        The x-axis label. The default is ''.
    xlim : tuple or NoneType, optional
        The x-axis limits. The default is None.
    major_tick_spacing : float, optional
        The spacing in data units between major tick marks. The default is 1.
    galr_bins : list, optional
        Bin edges of galactic radius in kpc. The default is 
        [3, 5, 7, 9, 11, 13, 15].
    absz_bins : list, optional
        Bin edges of galactic z-height in kpc. The default is [0, 0.5, 1, 2].
    cbar_width : float, optional
        The width of the colorbar as a fraction of the figure size. The default
        is 0.6.
    panel_aspect_ratio : float, optional
        The aspect ratio of the panels, equal to the panel width divided by
        the panel height. The default is 1.5.
    major_minor_tick_ratio : float, optional
        Ratio between major and minor tick spacing in data units. The default
        is 5.
    cmap : str or matplotlib colormap, optional
        Colormap for radial bins. If a string, it is assumed to be a standard
        matplotlib colormap and will be imported with plt.get_cmap().
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    # Determine figure dimensions
    nrows = len(absz_bins) - 1
    ax_width = (figure_width - 0.2) / ncols
    ax_height = ax_width / panel_aspect_ratio
    figure_height = ax_height * nrows + 1
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey='row',
                            figsize=(figure_width, figure_height))
    fig.subplots_adjust(left=0.09, top=0.93, right=0.98, bottom=0.22,
                        wspace=0.07, hspace=0.07)
    # Format x-axis
    axs[0,0].set_xlim(xlim)
    axs[0,0].xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    minor_tick_spacing = major_tick_spacing / major_minor_tick_ratio
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    # Remove spines and y-axis labels
    if not include_yaxis:
        for ax in axs.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('none')
            ax.yaxis.set_ticklabels([])
            ax.patch.set_alpha(0)
            ax.tick_params(top=False, which='both')
            # Set bottom ticks pointing out
            ax.tick_params(axis='x', which='both', direction='out')
    # Add common y-axis label
    axs[1,0].set_ylabel('Normalized PDF')
    # Label rows
    for i in range(len(absz_bins)-1):
        absz_lim = tuple(absz_bins[-(i+2):len(absz_bins)-i])
        # axs[i,-1].yaxis.set_label_position('right')
        # axs[i,-1].set_ylabel(r'$|z| = %s - %s$' % absz_lim)
        axs[i,0].text(
            0.5, 0.9, r'$|z| = %s - %s$' % absz_lim,
            ha='center', va='top', transform=axs[i,0].transAxes
        )
    # Add colorbar on bottom
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(galr_bins, cmap.N)
    cax = plt.axes([0.5 - (cbar_width / 2), 0.09, cbar_width, 0.02])
    cbar = fig.colorbar(ScalarMappable(norm, cmap), cax,
                        orientation='horizontal')
    cbar.set_label('Galactocentric radius [kpc]')
    return fig, axs
