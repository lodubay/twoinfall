"""
This file contains utility functions related to generating a grid of scatter
plots, e.g., showing [O/Fe] vs [Fe/H] over a range of Galactic regions.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.cm import ScalarMappable
from utils import sample_dataframe
from _globals import GALR_BINS, ABSZ_BINS, TWO_COLUMN_WIDTH


def plot_vice_sample(ax, stars, xcol, ycol, zcol='galr_origin', 
                     cmap=None, norm=None, sampled=True, 
                     nsamples=10000, markersize=0.1):
    """
    Generate a scatter plot of specified parameters from VICE multizone data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the scatter plot.
    stars : pandas.DataFrame
        VICE multizone stars data, usually a subset defined by a particular
        Galactic region.
    xcol : str
        Name of data column for x-axis values, e.g. '[fe/h]'.
    ycol : str
        Name of data column for y-axis values, e.g. '[o/fe]'.
    zcol : str, optional
        Name of data column for color mapping. The default is 'galr_origin'.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to color-code the scattered points. The default is None.
    norm : matplotlib.colors.Normalize, optional
        Normalization of the color-bar mapping. The default is None.
    sampled : bool, optional
        If True, sample the VICE stars (weighted by mass) instead of plotting 
        them all. The default is True.
    nsamples : int, optional
        Number of stellar populations to sample from the VICE output. The
        default is 10000.
    markersize : float, optional
        Size of scatter plot markers. The default is 0.1.
    """
    if sampled:
        # weight random sample by particle mass
        sample_weights = stars['mass'] / stars['mass'].sum()
        sample = sample_dataframe(stars, nsamples, weights=sample_weights)
    else:
        sample = stars.copy()
    # If zcol is not a valid column, don't color-code points
    if zcol in sample.columns:
        colors = sample[zcol]
    else:
        colors = None
    # Scatter plot of stellar particles
    ax.scatter(sample[xcol], sample[ycol], c=colors, s=markersize,
               cmap=cmap, norm=norm, rasterized=True, edgecolor='none')


def setup_colorbar(fig, cmap=None, vmin=None, vmax=None, label='', 
                   width=0.02, pad=0.01, labelpad=0, lognorm=False, 
                   bounds=[], extend='neither'):
    """
    Configure a vertical colorbar with a specified colormap and normalization.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    cmap : str or matplotlib.colors.Colormap
        Colormap to use. The default is None.
    vmin : float or None
        Minimum data value which will be mapped to 0.
    vmax : float or None
        Maximum data value which will be mapped to 1.
    label : str, optional
        Colorbar label
    width : float, optional
        Width of the colorbar as a fraction of figure size. The default is 0.02
    pad : float, optional
        Spacing between the right-most plot and the colorbar as a fraction of
        figure size. The default is 0.02.
    labelpad : float, optional
        Padding between colorbar and label in points. The default is 0.
    lognorm : bool, optional
        If True, assigns a logarithmic normalization instead of linear.
        The default is False.
    bounds : list, optional
        If provided, a discrete colorbar will be created using BoundaryNorm.
        The default is [].

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        Colorbar object
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    # Adjust subplots
    plt.subplots_adjust(right=fig.subplotpars.right - (width + pad + 0.03))
    # Define colorbar axis
    height = fig.subplotpars.top - fig.subplotpars.bottom
    cax = plt.axes([fig.subplotpars.right + pad, fig.subplotpars.bottom, 
                    width, height])
    # Set normalization
    if len(bounds) > 0:
        norm = BoundaryNorm(bounds, cmap.N, extend=extend)
    elif lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(ScalarMappable(norm, cmap), cax)
    cbar.set_label(label, labelpad=labelpad)
    return cbar


def setup_axes(galr_bins=GALR_BINS[:-1], absz_bins=ABSZ_BINS,
               width=TWO_COLUMN_WIDTH, xlim=None, ylim=None,
               xlabel='', ylabel='', xlabelpad=0., ylabelpad=0.,
               row_label_pos=(0.07, 0.88), spacing=0., title=''):
    """
    Set up a blank grid of axes with a default subplot spacing.

    Parameters
    ----------
    galr_bins : list, optional
        Bins of Galactocentric radius in kpc. The default is in _globals.py.
    absz_bins : list, optional
        Bins of absolute Galactic z-height in kpc. The default is in _globals.py.
    rows : int, optional
        Number of rows of axes. The default is 3.
    cols : int, optional
        Number of columns of axes. The default is 5.
    width : float, optional
        Width of the figure in inches. The default is 7 in.
    xlim, ylim : tuple or None, optional
        Axis limits for all panels. The default is None.
    xlabel, ylabel : str, optional
        The x- and y-axis labels. The default is ''.
    xlabelpad, ylabelpad : float, optional
        The x- and y-axis label pad in points. The default is 0.
    row_label_pos : tuple, optional
        The (x, y) position, in axis coordinates, of the row label text.
        The default is (0.07, 0.88).
    spacing : float, optional
        Sets the wspace and hspace parameters in subplots_adjust. The default
        is 0.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    rows = len(absz_bins) - 1
    cols = len(galr_bins) - 1
    fig, axs = plt.subplots(rows, cols, figsize=(width, (width/cols)*rows),
                            sharex=True, sharey=True)
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.08, top=0.95,
                        wspace=spacing, hspace=spacing)
    # Figure title
    if len(title) > 0:
        fig.suptitle(title)
        fig.subplots_adjust(top=0.88)
    # Axis limits
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim(ylim)
    # Axis labels
    for ax in axs[-1]:
        ax.set_xlabel(xlabel, labelpad=xlabelpad)
    for i, ax in enumerate(axs[:,0]):
        ax.set_ylabel(ylabel, labelpad=ylabelpad)
        # Label bins in z-height
        absz_lim = (absz_bins[-(i+2)], absz_bins[-(i+1)])
        ax.text(row_label_pos[0], row_label_pos[1], 
                r'$%s\leq |z| < %s$ kpc' % absz_lim,
                transform=ax.transAxes)
    # Label bins in Rgal
    for i, ax in enumerate(axs[0]):
        galr_lim = (galr_bins[i], galr_bins[i+1])
        ax.set_title(r'$%s\leq R_{\rm{Gal}} < %s$ kpc' % galr_lim)
    return fig, axs
