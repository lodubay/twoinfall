"""
This script plots a grid of panels showing [O/Fe] vs age in multiple galactic
regions. The plot includes a sample of VICE stellar populations, mass-weighted 
medians of the VICE output, and the median estimated ages from Feuillet et al. 
(2019), Mackereth et al. (2019; astroNN), or Leung et al. (2023).
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scatter_plot_grid import setup_axes, setup_colorbar, plot_gas_abundance
from utils import weighted_quantile, group_by_bins, get_bin_centers
from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from _globals import GALR_BINS, ABSZ_BINS
import paths

# GALR_BINS = GALR_BINS[:-1]
AGE_LIM_LINEAR = (-1, 15)
AGE_LIM_LOG = (0.3, 20)
OFE_LIM = (-0.15, 0.55)
OFE_BIN_WIDTH = 0.05
AGE_LABEL = 'L23'

def main(output_name, verbose=False, uncertainties=False, **kwargs):
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    # Import VICE multizone outputs
    mzs = MultizoneStars.from_output(output_name, verbose=verbose)
    # Model uncertainties
    if uncertainties:
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
    # Main plot function
    plot_age_ofe_grid(mzs, apogee_sample, verbose=verbose, **kwargs)


def plot_age_ofe_grid(mzs, apogee_sample, fname='age_ofe_grid.png', 
                      style='paper', cmap='winter_r', log=True, verbose=False,
                      apogee_medians=True, tracks=True):
    """
    Plot a grid of [O/Fe] vs age across multiple Galactic regions.
    
    Parameters
    ----------
    vice_stars : MultizoneStars object
        Instance containing VICE multi-zone stellar output.
    apogee_sample : APOGEESample object
        APOGEE sample
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
    savedir : str or pathlib.Path, optional
        Parent directory to save the plot. The default is ../debug/age_ofe/.
    """
    # Set x-axis limits
    if log:
        age_lim = AGE_LIM_LOG
    else:
        age_lim = AGE_LIM_LINEAR
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_axes(xlim=age_lim, ylim=OFE_LIM, 
                          xlabel='Age [Gyr]', ylabel='[O/Fe]',
                          xlabelpad=2, ylabelpad=4,
                          galr_bins=GALR_BINS, absz_bins=ABSZ_BINS,
                          title=mzs.name, width=8)
    cbar = setup_colorbar(fig, cmap=cmap, vmin=0, vmax=15.5, 
                          label=r'Birth $R_{\rm{Gal}}$ [kpc]', pad=0.02)
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
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
            subset.scatter_plot(ax, 'age', '[o/fe]', color='galr_origin',
                                     cmap=cmap, norm=cbar.norm)
            if tracks:
                plot_gas_abundance(ax, subset, 'lookback', '[o/fe]')
            if apogee_medians:
                ofe_bins = np.arange(OFE_LIM[0], OFE_LIM[1]+OFE_BIN_WIDTH, 
                                     OFE_BIN_WIDTH)
                apogee_subset = apogee_sample.region(galr_lim, absz_lim)
                plot_apogee_median_ages(ax, apogee_subset, 'O_FE', ofe_bins, label='L23')
                plot_vice_median_ages(ax, subset, '[o/fe]', ofe_bins, label='Model')
                # Add legend to top-right panel
                if i==0 and j==len(row)-1:
                    ax.legend(loc='upper left', frameon=False, handletextpad=0.1,
                              borderpad=0.2, handlelength=1.5)
                             
    # Set x-axis scale and ticks
    if log:
        axs[0,0].set_xscale('log')
        axs[0,0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    else:
        axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
        axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
        
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Save
    fullpath = paths.extra / mzs.name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


def plot_vice_median_ages(ax, mzs, col, bin_edges, label=None, 
                          color='k', min_mass_frac=0.01):
    """
    Plot median stellar ages binned by [O/Fe] from APOGEE data.
    
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
    age_col : str, optional
        Name of column containing ages. The default is 'AGE'.
    min_mass_frac : float, optional
        The minimum stellar mass fraction in a bin required for that bin
        to be plotted. The default is 0.01, or 1% of the total mass in the
        VICE output subset.
    """
    age_intervals = mzs.age_intervals(col, bin_edges, 
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


def plot_apogee_median_ages(ax, apogee_sample, col, bin_edges, label=None, 
                            color='r', age_col='AGE', min_stars=10):
    """
    Plot median stellar ages binned by [O/Fe] from APOGEE data.
    
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
        Name of column containing ages. The default is 'AGE'.
    min_stars : int, optional
        The minimum number of stars in a bin required to plot the age interval.
        The default is 10.
    """
    age_intervals = apogee_sample.age_intervals(col, bin_edges, 
                                                quantiles=[0.16, 0.5, 0.84], 
                                                age_col=age_col)
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
        prog='age_ofe_grid.py',
        description='Generate a grid of [O/Fe] vs age plots comparing the' + \
            ' output of a VICE multizone run to APOGEE and astroNN data.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--cmap', metavar='COLORMAP', type=str,
                        default='winter_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: winter_r)')
    parser.add_argument('-t', '--tracks', action='store_true',
                        help='Plot ISM tracks in addition to stellar abundances')
    parser.add_argument('-l', '--log', action='store_true',
                        help='Plot age on a log scale')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('-a', '--apogee-medians', action='store_true',
                        help='Plot age medians from APOGEE data')
    args = parser.parse_args()
    main(**vars(args))
