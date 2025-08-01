"""
This script plots stellar age vs abundance across a Galactocentric grid.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scatter_plot_grid import plot_gas_abundance, setup_axes, setup_colorbar
from _globals import GALR_BINS, ABSZ_BINS, MAX_SF_RADIUS, END_TIME
from utils import vice_to_apogee_col, capitalize_abundance
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths

ABUNDANCE_COLUMNS = ['[o/h]', '[fe/h]', '[o/fe]', '[fe/o]']
AXES_LABEL = {
    'galr_origin': r'Birth $R_{\rm{Gal}}$ [kpc]',
    # 'age': 'Stellar age [Gyr]',
    # 'log_age': 'log(stellar age [yr])',
    '[fe/h]': '[Fe/H]',
    '[o/h]': '[O/H]',
    '[o/fe]': '[O/Fe]',
    '[fe/o]': '[Fe/O]',
}
AXES_LIM = {
    '[o/h]': (-1.3, 0.8),
    '[fe/h]': (-1.5, 0.7),
    '[o/fe]': (-0.15, 0.55),
    '[fe/o]': (-0.55, 0.15),
    'galr_origin': (0, MAX_SF_RADIUS),
    # 'age': (0, 14),
    # 'log_age': (8.5, 10.3),
}
AXES_MAJOR_LOCATOR = {
    '[o/h]': 0.5,
    '[fe/h]': 0.5,
    '[o/fe]': 0.2,
    '[fe/o]': 0.2,
    'galr_origin': 2,
}
AXES_MINOR_LOCATOR = {
    '[o/h]': 0.1,
    '[fe/h]': 0.1,
    '[o/fe]': 0.05,
    '[fe/o]': 0.05,
    'galr_origin': 0.5,
}
BIN_WIDTH = {
    '[o/h]': 0.1,
    '[fe/h]': 0.1,
    '[o/fe]': 0.05,
    '[fe/o]': 0.05,
    'age': 2
}
ROW_LABEL_POS = {
    '[o/h]': (0.07, 0.07),
    '[fe/h]': (0.07, 0.07),
    '[o/fe]': (0.07, 0.88),
    '[fe/o]': (0.07, 0.07),
}


def main(output_name, ydata, verbose=False, uncertainties=False, **kwargs):
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    # Import VICE multizone outputs
    mzs = MultizoneStars.from_output(output_name, verbose=verbose)
    # Model uncertainties
    if uncertainties:
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
    plot_age_abundance_grid(mzs, ydata, apogee_sample=apogee_sample, 
                            verbose=verbose, **kwargs)


def plot_age_abundance_grid(mzs, col, apogee_sample=None, fname='',
                            style='paper', cmap='winter_r', log=False, verbose=False,
                            medians=False, tracks=False, color_by='galr_origin',
                            bin_by_age=False):
    """
    Plot a grid of age (x) vs abundance (y) across multiple Galactic regions.
    
    Parameters
    ----------
    mzs : MultizoneStars object
        Instance containing VICE multi-zone stellar output.
    col : str
        Column containing abundance data in the VICE output.
    apogee_sample : APOGEESample object or None, optional
        APOGEE sample data. If None, will be loaded automatically. The default
        is None.
    fname : str, optional
        File name (excluding parent directory) of plot output. The default is
        'age_ofe.png'.
    cmap : str, optional
        Name of colormap for scatterplot. The default is 'winter'.
    log : bool, optional
        If True, plot the x (age) axis on a log scale. The default is False.
    score : bool, optional
        If True, calculate a numerical score for how well the VICE output
        matches the age data. The default is False.
    verbose : bool, optional
        If True, print verbose output to the terminal. The default is False.
    bin_by_age : bool, optional
        If True, calculate abundance medians in bins of age. If False, 
        calculate age medians in abundance bins.
    """
    col = col.lower()
    if col not in ABUNDANCE_COLUMNS:
        raise ValueError('%s is not an allowed abundance column.' % col)
    color_by = color_by.lower()
    if color_by not in AXES_LABEL.keys():
        raise ValueError('%s is not an allowed abundance column.' % color_by)
    if apogee_sample is None:
        apogee_sample = APOGEESample.load()
    apogee_col = vice_to_apogee_col(col)
        
    # Set x-axis limits
    if log:
        age_lim = (0.3, 20)
    else:
        age_lim = (0, 14)
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    ylim = AXES_LIM[col]
    fig, axs = setup_axes(xlim=age_lim, ylim=AXES_LIM[col], 
                          xlabel='Age [Gyr]', 
                          ylabel=capitalize_abundance(col),
                          xlabelpad=2, ylabelpad=4,
                          galr_bins=GALR_BINS, absz_bins=ABSZ_BINS,
                          title=mzs.name, width=8,
                          row_label_pos=ROW_LABEL_POS[col])
    cbar_lim = AXES_LIM[color_by.lower()]
    cbar = setup_colorbar(fig, cmap=cmap, vmin=cbar_lim[0], vmax=cbar_lim[1], 
                          label=AXES_LABEL[color_by.lower()], pad=0.02)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(AXES_MAJOR_LOCATOR[color_by]))
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(AXES_MINOR_LOCATOR[color_by]))
    
    # Plot sampled points and medians
    if verbose: 
        print('Plotting [O/Fe] vs age in galactic regions...')
    for i, row in enumerate(axs):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            if verbose:
                print('\tRGal=%s kpc, |z|=%s kpc' \
                      % (str(galr_lim), str(absz_lim)))
            subset = mzs.region(galr_lim, absz_lim)
            subset.scatter_plot(ax, 'age', col, color=color_by,
                                cmap=cmap, norm=cbar.norm)
            if tracks:
                plot_gas_abundance(ax, subset, 'lookback', col, ls='--')
            if medians:
                apogee_subset = apogee_sample.region(galr_lim, absz_lim)
                if bin_by_age:
                    age_bins = np.arange(0, END_TIME + BIN_WIDTH['age'], 
                                         BIN_WIDTH['age'])
                    plot_vice_median_abundances(ax, subset, col, age_bins, 
                                                label='Model')
                    plot_apogee_median_abundances(ax, apogee_subset, apogee_col, 
                                                  age_bins, age_col='L23_AGE', label='L23')
                else:
                    abund_bins = np.arange(ylim[0], ylim[1] + BIN_WIDTH[col], 
                                           BIN_WIDTH[col])
                    plot_vice_median_ages(ax, subset, col, abund_bins, label='Model')
                    plot_apogee_median_ages(ax, apogee_subset, apogee_col, abund_bins, label='L23')
    
    # Add legend to top-right panel
    if medians:
        axs[0,-1].legend(loc='upper left', frameon=False, handletextpad=0.1,
                         borderpad=0.2, handlelength=1.5)
                             
    # Set x-axis scale and ticks
    if log:
        axs[0,0].set_xscale('log')
        axs[0,0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    else:
        axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
        axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
        
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(AXES_MAJOR_LOCATOR[col]))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(AXES_MINOR_LOCATOR[col]))
    
    # Save
    if fname == '':
        fname = 'age_%s_grid.png' % col[1:-1].replace('/', '')
    fullpath = paths.extra / 'multizone' / mzs.name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


def plot_vice_median_abundances(ax, mzs, col, bin_edges, label=None,
                                color='k', min_mass_frac=0.01, offset=0.,
                                marker='s'):
    """
    Plot median stellar abundances binned by age from VICE multi-zone output.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    mzs : MultizoneStars instance
        VICE multizone stars output.
    col : str
        Data column with abundance data.
    bin_edges : array-like
        Age bin edges to group the data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    min_mass_frac : float, optional
        The minimum stellar mass fraction in a bin required for that bin
        to be plotted. The default is 0.01, or 1% of the total mass in the
        VICE output subset.
    offset : float, optional
        Offset to age bin center, in Gyr, for visual clarity. The default is 0.
    """
    abundance_intervals = mzs.binned_intervals(col, 'age', bin_edges,
                                               quantiles=[0.16, 0.5, 0.84])
    # Drop bins with few targets
    include = abundance_intervals['mass_fraction'] >= min_mass_frac
    abundance_intervals = abundance_intervals[
        abundance_intervals['mass_fraction'] >= min_mass_frac
    ]
    bin_edges_left = abundance_intervals.index.categories[include].left.values
    bin_edges_right = abundance_intervals.index.categories[include].right.values
    bin_centers = (bin_edges_left + bin_edges_right) / 2
    # ax.errorbar(bin_centers + offset, abundance_intervals[0.5], 
    #             xerr=((bin_centers - bin_edges_left) + offset,
    #                   (bin_edges_right - bin_centers) - offset),
    #             yerr=(abundance_intervals[0.5] - abundance_intervals[0.16], 
    #                   abundance_intervals[0.84] - abundance_intervals[0.5]),
    #             color=color, linestyle='none', capsize=1, elinewidth=0.5,
    #             capthick=0.5, marker='^', markersize=2, label=label,
    # )
    line2d = ax.plot(bin_centers, abundance_intervals[0.5], color=color, 
            linestyle='none', marker=marker, ms=4)
    return line2d


def plot_vice_median_ages(ax, mzs, col, bin_edges, label=None, 
                          color='k', min_mass_frac=0.01):
    """
    Plot median stellar ages binned by abundance from VICE multi-zone output.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    mzs : MultizoneStars instance
        VICE multizone stars output.
    col : str
        Data column to group by.
    bin_edges : array-like
        Bin edges to group the data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    min_mass_frac : float, optional
        The minimum stellar mass fraction in a bin required for that bin
        to be plotted. The default is 0.01, or 1% of the total mass in the
        VICE output subset.
    """
    age_intervals = mzs.binned_intervals('age', col, bin_edges, 
                                         quantiles=[0.16, 0.5, 0.84])
    # Drop bins with few targets
    include = age_intervals['mass_fraction'] >= min_mass_frac
    age_intervals = age_intervals[age_intervals['mass_fraction'] >= min_mass_frac]
    bin_edges_left = age_intervals.index.categories[include].left
    bin_edges_right = age_intervals.index.categories[include].right
    bin_centers = (bin_edges_left + bin_edges_right) / 2
    ax.errorbar(age_intervals[0.5], bin_centers, 
                xerr=(age_intervals[0.5] - age_intervals[0.16], 
                      age_intervals[0.84] - age_intervals[0.5]),
                yerr=(bin_centers - bin_edges_left,
                      bin_edges_right - bin_centers),
                color=color, linestyle='none', capsize=1, elinewidth=0.5,
                capthick=0.5, marker='^', markersize=2, label=label,
    )


def plot_apogee_median_abundances(ax, apogee_sample, col, bin_edges, label=None, 
                                  color='r', age_col='L23_AGE', min_stars=10,
                                  alpha=0.2, linestyle='-', marker='o', **kwargs):
    """
    Plot APOGEE stellar abundance medians and 1-sigma range binned by age.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    apogee_sample : APOGEESample instance
        APOGEE sample data or subset of the sample.
    col : str
        Data column with abundance data.
    bin_edges : array-like
        Age bin edges to group the data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    age_col : str, optional
        Name of column containing ages. The default is 'L23_AGE'.
    min_stars : int, optional
        The minimum number of stars in a bin required to plot the age interval.
        The default is 10.
    alpha : float, optional
        Transparency of the 1-sigma range. The default is 0.3.
    **kwargs passed to matplotlib.pyplot.fill_between
    
    Returns
    -------
    spatch : matplotlib.patches.StepPatch
    pcol : matplotlib.collections.FillBetweenPolyCollection
    """
    abundance_intervals = apogee_sample.binned_intervals(col, age_col, bin_edges, 
                                                         quantiles=[0.16, 0.5, 0.84])
    # Drop bins with few targets
    include = abundance_intervals['count'] >= min_stars
    abundance_intervals = abundance_intervals[include]
    bin_edges_left = abundance_intervals.index.categories[include].left.values
    bin_edges_right = abundance_intervals.index.categories[include].right.values
    bin_centers = (bin_edges_left + bin_edges_right) / 2
    bin_edges = np.append(bin_edges_left, bin_edges_right[-1:])
    low = abundance_intervals[0.16].values
    high = abundance_intervals[0.84].values
    # duplicate first and last points to extend 1-sigma bounds from 0-14 Gyr
    pcol = ax.fill_between(
        np.concatenate((bin_edges[0:1], bin_centers, bin_edges[-1:])),
        np.concatenate((low[0:1], low, low[-1:])),
        np.concatenate((high[0:1], high, high[-1:])),
        # step='post', 
        color=color, alpha=alpha, label=label, edgecolor=color, linestyle=linestyle,
        **kwargs
    )
    # plot median line
    # spatch = ax.stairs(abundance_intervals[0.5].values, edges=bin_edges,
    #                    baseline=None, color=color, linestyle=linestyle)
    line2d = ax.plot(bin_centers, abundance_intervals[0.5].values, 
                     color=color, linestyle=linestyle, marker=marker, ms=4)
    # print(line2d)
    return line2d[0], pcol


def plot_apogee_median_ages(ax, apogee_sample, col, bin_edges, label=None, 
                            color='r', age_col='L23_AGE', min_stars=10):
    """
    Plot median stellar ages binned by abundance from APOGEE data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    apogee_sample : APOGEESample instance
        APOGEE sample data or subset of the sample.
    col : str
        Data column to group by.
    bin_edges : array-like
        Bin edges to group the data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    age_col : str, optional
        Name of column containing ages. The default is 'L23_AGE'.
    min_stars : int, optional
        The minimum number of stars in a bin required to plot the age interval.
        The default is 10.
    """
    age_intervals = apogee_sample.binned_intervals(age_col, col, bin_edges, 
                                                   quantiles=[0.16, 0.5, 0.84])
    # Drop bins with few targets
    include = age_intervals['count'] >= min_stars
    age_intervals = age_intervals[include]
    bin_edges_left = age_intervals.index.categories[include].left
    bin_edges_right = age_intervals.index.categories[include].right
    bin_centers = (bin_edges_left + bin_edges_right) / 2
    ax.errorbar(age_intervals[0.5], bin_centers, 
                xerr=(age_intervals[0.5] - age_intervals[0.16], 
                      age_intervals[0.84] - age_intervals[0.5]),
                yerr=(bin_centers - bin_edges_left,
                      bin_edges_right - bin_centers),
                color=color, linestyle='none', capsize=1, elinewidth=0.5,
                capthick=0.5, marker='^', markersize=2, label=label,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='age_abundance_grid.py',
        description='Generate a grid of stellar abundance vs age plots across \
multiple galactic regions comparing multizone model outputs and APOGEE data.'
    )
    parser.add_argument(
        'output_name', 
        metavar='NAME',
        type=str,
        help='Name of VICE multizone output located within src/data/multizone.'
    )
    parser.add_argument(
        '-y', '--ydata', 
        metavar='COLUMN', 
        required=True,
        type=str,
        help='Abundance to plot on the y-axis. Must be a column in the VICE \
multizone output.'
    )
    parser.add_argument(
        '-c', '--color-by', 
        metavar='COLUMN', 
        default='galr_origin', 
        help='Column to color multizone output by.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '-t', '--tracks', 
        action='store_true',
        help='Plot ISM tracks in addition to stellar abundances.'
    )
    parser.add_argument(
        '-l', '--log', 
        action='store_true',
        help='Plot age on a log scale.'
    )
    parser.add_argument(
        '-u', '--uncertainties', 
        action='store_true',
        help='Forward-model APOGEE uncertainties in VICE output.'
    )
    parser.add_argument(
        '-m', '--medians', 
        action='store_true',
        help='Plot age medians in abundance bins for model and APOGEE data.'
    )
    parser.add_argument(
        '--bin-by-age', 
        action='store_true',
        help='Calculate abundance medians in stellar age bins, rather than the reverse.'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default='winter_r',
        help='Name of colormap for color-coding model output (default: winter_r).'
    )
    args = parser.parse_args()
    main(**vars(args))
