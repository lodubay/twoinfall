"""
This file contains functions for plotting abundance tracks in [Fe/H] vs [O/Fe]
alongside their corresponding metallicity distribution functions (MDFs).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice
from utils import get_bin_centers, gaussian_smooth
from _globals import ONE_COLUMN_WIDTH


def plot_vice_onezone(output, fig=None, axs=[], label=None, color=None,
                      marker_labels=False, mdf_smoothing=0.02, 
                      markersize=9, xcol='[fe/h]', ycol='[o/fe]', **kwargs):
    """
    Wrapper for plot_track_and_mdf given a VICE onezone output.

    Parameters
    ----------
    output : str
        Path to VICE output, not necessarily including the '.vice' extension
    fig : instance of matplotlib.figure.figure, optional
        If no figure is provided, one is generated from setup_axes.
    axs : list of matplotlib.axes.Axes
        There should be three axes: the first for the main [Fe/H] vs [O/Fe]
        panel, the second for the MDF in [Fe/H], and the third for the MDF
        in [O/Fe]. If none are provided, they are generated from setup_axes.
    label : str, optional
        Plot label to add to main panel legend
    color : str, optional
        Line color. The default is None, which chooses a color automatically.
    marker_labels : bool, optional
        If True, label the time markers. The default is False.
    mdf_smoothing : float, optional
        Width of Gaussian smoothing to apply to the marginal distributions
        in data units. The default is 0.02 dex.
    markersize : float, optional
        Size of time markers. The default is 9.
    xcol : str, optional
        Column with x-axis data. The default is '[fe/h]'.
    ycol : str, optional
        Column with y-axis data. The default is '[o/fe]'.
    **kwargs passed to matplotlib.plot
    style_kw : dict, optional
        Dict of style-related keyword arguments to pass to both
        matplotlib.pyplot.plot and matplotlib.pyplot.hist

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    if fig == None or len(axs) != 3:
        fig, axs = setup_figure()
    hist = vice.history(output)
    mdf = vice.mdf(output)
    mdf_bins = mdf['bin_edge_left'] + mdf['bin_edge_right'][-1:]
    # Plot abundance tracks on main panel
    axs[0].plot(hist[xcol], hist[ycol], label=label, color=color, 
                **kwargs)
    # Apply same color to marginal plots
    if color == None:
        color = axs[0].lines[-1].get_color()
    plot_mdf_curve(axs[1], mdf['dn/d%s' % xcol], mdf_bins, smoothing=mdf_smoothing,
                   color=color, **kwargs)
    plot_mdf_curve(axs[2], mdf['dn/d%s' % ycol], mdf_bins, smoothing=mdf_smoothing,
                   orientation='horizontal', color=color, **kwargs)
    # Time markers should have same z-order as lines
    zorder = axs[0].lines[-1].get_zorder()
    plot_time_markers(hist['time'], hist[xcol], hist[ycol], axs[0],
                      color=color, show_labels=marker_labels, zorder=zorder,
                      markersize=markersize)
    return fig, axs


def plot_time_markers(time, feh, ofe, ax, loc=[0.1, 0.3, 1, 3, 10],
                      color=None, show_labels=False, zorder=10, markersize=9):
    """
    Add temporal markers to the [O/Fe] vs [Fe/H] tracks.

    Parameters
    ----------
    time : array-like
        Array of simulation time in Gyr.
    feh : array-like
        Array of [Fe/H] abundances.
    ofe : array-like
        Array of [O/Fe] abundances.
    ax : matplotlib.axes.Axes
        Axis in which to plot the time markers.
    loc : list, optional
        List of times in Gyr to mark. The default is [0.1, 0.3, 1, 3, 10].
    color : color or None, optional
        Color of the markers. The default is None.
    show_labels : bool, optional
        Whether to add marker labels. The default is False.
    zorder : int, optional
        Z-order of markers.
    markersize : float, optional
        Size of time markers. The default is 9.
    """
    # Get default font size
    default_font_size = plt.rcParams['font.size']
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'X']
    time = np.array(time)
    for i, t in enumerate(loc):
        idx = np.argmin(np.abs(time - t))
        ax.scatter(feh[idx], ofe[idx], s=markersize, marker=markers[i],
                   edgecolors=color, facecolors='w', zorder=zorder)
        if show_labels:
            if t < 1:
                label = f'{int(t*1000)} Myr'
            else:
                label = f'{int(t)} Gyr'
            if i == 0:
                xpad = -0.1
                ypad = 0.01
            else:
                xpad = 0.03
                ypad = 0.008
            ax.text(feh[idx] + xpad, ofe[idx] + ypad, label, 
                    fontsize=default_font_size * 7/8,
                    ha='left', va='bottom', zorder=10,
                    # bbox={
                    #     'facecolor': 'w',
                    #     'edgecolor': 'none',
                    #     'boxstyle': 'round',
                    #     'pad': 0.05,
                    #     'alpha': 0.8,
                    # },
            )


def plot_mdf(ax, mdf, bins, histtype='step', log=False, bin_mult=1, **kwargs):
    """
    Plot a histogram of the metallicity distribution function (MDF).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    mdf : array-like
        Values of the MDF.
    bins : array-like
        MDF bins. Size should be 1 greater than mdf.
    histtype : str, optional
        Histogram style. The default is 'step'.
    log : bool, optional
        Whether to plot the histogram on a log scale. The default is False.
    bin_mult : int, optional
        If greater than 1, will join that number of adjacent bins together.
    """
    # join bins together
    bins = bins[::bin_mult]
    mdf = [sum(mdf[i:i+bin_mult]) for i in range(0, len(mdf), bin_mult)]
    # mask zeros before taking log
    mdf = np.array(mdf)
    if log:
        mdf[mdf == 0] = 1e-10
        weights = np.log10(mdf)
    else:
        weights = mdf
    ax.hist(bins[:-1], bins, weights=weights, histtype=histtype, **kwargs)


def plot_mdf_curve(ax, mdf, bins, smoothing=0., orientation='vertical', **kwargs):
    """
    Plot marginal abundance distribution as a curve.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    mdf : array-like
    bins : array-like
        Length must be len(mdf) + 1
    smoothing : float, optional
        Width of Gaussian smoothing to apply. The default is 0.
    orientation : str, optional
        Orientation of the plot. Options are 'horizontal' or 'vertical'.
        The default is 'vertical'.
    **kwargs passed to matplotlib.plot
    """
    bin_centers = get_bin_centers(bins)
    if smoothing > 0:
        mdf = gaussian_smooth(mdf, bins, smoothing)
    if orientation == 'horizontal':
        ax.plot(mdf, bin_centers, **kwargs)
    else:
        ax.plot(bin_centers, mdf, **kwargs)
        

def setup_figure(width=ONE_COLUMN_WIDTH, **kwargs):
    """
    Create a figure with a three-panel setup for onezone evolutionary tracks.

    Parameters
    ----------
    width : float, optional
        Width of the figure in inches. The default is 3.25 in.
    **kwargs : dict
        Keyword arguments passed to setup_axes().

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    fig = plt.figure(figsize=(width, width))
    axs = setup_axes(fig, **kwargs)
    plt.subplots_adjust(top=0.98, right=0.98, bottom=0.11, left=0.14)
    return fig, axs


def setup_axes(fig, title='', xlim=(-2.1, 0.4), ylim=(-0.1, 0.52), ylabel=True,
               xname='[Fe/H]', yname='[O/Fe]'):
    """
    Create three axes: the main abundance track axis plus two
    side panels for [Fe/H] and [O/Fe] distribution functions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure or SubFigure which will contain the axes.
    title : str, optional
        Figure title which will be positioned in the upper MDF panel.
    xlim : tuple, optional
        Bounds on x-axis.
    ylim : tuple, optional
        Bounds on y-axis.
    ylabel : bool, optional
        If False, remove y-axis labels and tick labels.
    xname : str, optional
        Abundance label for the x-axis. The default is '[Fe/H]'.
    yname : str, optional
        Abundance label for the y-axis. THe default is '[O/Fe]'.

    Returns
    -------
    fig : matplotlib.figure
    axs : list of matplotlib.axes.Axes
        Ordered [ax_main, ax_mdf, ax_odf]
    """
    # Get default axis label size
    default_label_size = plt.rcParams['axes.labelsize']
    # small_label_size = default_label_size * 0.7
    small_label_size = 'small'
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0., hspace=0.)
    # Start with the center panel for [Fe/H] vs [O/Fe]
    ax_main = fig.add_subplot(gs[1,0])
    ax_main.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_main.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_main.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_main.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax_main.set_xlabel(xname)
    if ylabel:
        ax_main.set_ylabel(yname, labelpad=-2)
    else:
        ax_main.yaxis.set_ticklabels([])
        gs.update(left=0.02)
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    # Add panel above for MDF in [Fe/H]
    ax_mdf = fig.add_subplot(gs[0,0], sharex=ax_main)
    ax_mdf.tick_params(axis='x', labelbottom=False)
    ax_mdf.tick_params(axis='y', which='both', left=False, right=False, 
                       labelleft=False)
    if ylabel:
        ax_mdf.set_ylabel(r'$P($%s$)$' % xname, size=small_label_size)
    # Add plot title
    ax_mdf.set_title(title, loc='left', x=0.05, y=0.8, va='top', pad=0)
    # Add panel to the right for MDF in [O/Fe]
    ax_odf = fig.add_subplot(gs[1,1], sharey=ax_main)
    ax_odf.tick_params(axis='y', labelleft=False)
    ax_odf.tick_params(axis='x', which='both', bottom=False, top=False, 
                       labelbottom=False)
    ax_odf.set_xlabel(r'$P($%s$)$' % yname, size=small_label_size)
    axs = [ax_main, ax_mdf, ax_odf]
    return axs
