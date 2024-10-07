"""
This script plots a grid of panels showing [O/Fe] vs age in multiple galactic
regions. The plot includes a sample of VICE stellar populations, mass-weighted 
medians of the VICE output, and the median estimated ages from Feuillet et al. 
(2019), Mackereth et al. (2019; astroNN), or Leung et al. (2023).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import vice
from scatter_plot_grid import setup_axes, setup_colorbar
from utils import weighted_quantile, group_by_bins, feuillet2019_data
from multizone_stars import MultizoneStars
from apogee_tools import import_apogee, apogee_region
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
    apogee_data = import_apogee(verbose=verbose)
    # Import VICE multizone outputs
    vice_stars = MultizoneStars.from_output(output_name, verbose=verbose)
    # Model uncertainties
    if uncertainties:
        vice_stars.model_uncertainty(apogee_data, inplace=True)
    # Main plot function
    plot_age_ofe(vice_stars, apogee_data, verbose=verbose, **kwargs)


def plot_age_ofe(vice_stars, apogee_data, fname='age_ofe.png', style='paper',
                 cmap='winter_r', log=True, verbose=False):
    """
    Plot a grid of [O/Fe] vs age across multiple Galactic regions.
    
    Parameters
    ----------
    vice_stars : MultizoneStars object
        Instance containing VICE multi-zone stellar output.
    apogee_data : pandas.DataFrame
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
        
    Returns
    -------
    float
        If scores==True, the weighted average score across all regions. The
        score represents the difference between the median ages from VICE
        and APOGEE. A lower score represents a better fit.
    """
    age_col = 'LATENT_AGE'
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
                          title=vice_stars.name, width=8)
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
            vice_subset = vice_stars.region(galr_lim, absz_lim)
            vice_subset.scatter_plot(ax, 'age', '[o/fe]', color='galr_origin',
                                     cmap=cmap, norm=cbar.norm)
            apogee_subset = apogee_region(apogee_data, galr_lim, absz_lim)
            plot_apogee_medians(ax, apogee_subset, label=AGE_LABEL)
            plot_vice_medians(ax, vice_subset.stars, label='Model')
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
    
    # Output figure
    save_dir = paths.extra / vice_stars.name.split('/diskmodel')[0]
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    fig.savefig(save_dir / fname, dpi=300)
    plt.close()


def rms_median_diff(vice_stars, apogee_data, age_col='LATENT_AGE', 
                    ofe_lim=OFE_LIM, ofe_bin_width=OFE_BIN_WIDTH):
    """
    Calculate the RMS of the difference in medians between VICE and data ages.
    
    Parameters
    ----------
    vice_stars : pandas.DataFrame
        VICE multizone stars data, typically a subset of a galactic region.
    apogee_data : pandas.DataFrame
        APOGEE data with ages, typically a subset of a galactic region.
    age_col : str, optional
        Name of column with age data in apogee_data. The default is 'LATENT_AGE'
        which is the Leung et al. (2023) ages.
    ofe_lim : tuple, optional
        Outermost bounds on [O/Fe]. The default is (-0.15, 0.55).
    ofe_bin_width : float, optional
        The [O/Fe] bin width in dex. The default is 0.05.
        
    Returns
    -------
    d_rms : float
        Root-mean-square of the difference in median ages in each [O/Fe] bin.
    """
    ofe_bins = np.arange(ofe_lim[0], ofe_lim[1]+ofe_bin_width, ofe_bin_width)
    # bin APOGEE ages by [O/Fe]
    apogee_grouped = group_by_bins(apogee_data, 'O_FE', ofe_bins)[age_col]
    apogee_medians = apogee_grouped.median()
    # count all APOGEE stars in each bin
    apogee_counts = apogee_grouped.count()
    # bin mass-weighted VICE ages by [O/Fe]
    vice_grouped = group_by_bins(vice_stars, '[o/fe]', bins=ofe_bins)
    # weighted medians of VICE
    wm = lambda x: weighted_quantile(x, 'age', 'mstar', quantile=0.5)
    vice_medians = vice_grouped.apply(wm)
    # RMS of median difference
    notna = (pd.notna(apogee_medians) & pd.notna(vice_medians))
    median_diffs = vice_medians[notna] - apogee_medians[notna]
    d_rms = np.sqrt(np.average(median_diffs**2, weights=apogee_counts[notna]))
    
    return d_rms


def plot_vice_medians(ax, stars, ofe_lim=OFE_LIM, ofe_bin_width=OFE_BIN_WIDTH,
                      plot_low_mass_bins=True, low_mass_cutoff=0.01,
                      marker='s', small_marker='x', label='VICE', 
                      small_label=None, markersize=2):
    """
    Plot median stellar ages binned by [O/Fe] from VICE multizone data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    stars : pandas.DataFrame
        VICE multizone stars data, typically a subset of a galactic region.
    ofe_lim : tuple, optional
        Outermost bounds on [O/Fe]. The default is (-0.15, 0.55).
    ofe_bin_width : float, optional
        The [O/Fe] bin width in dex. The default is 0.05.
    plot_low_mass_bins : bool, optional
        Whether to include bins with little stellar mass. If True, these bins
        are shown with a different, smaller marker. The default is True.
    low_mass_cutoff : float, optional
        The cutoff as a fraction of the total mass for what constitutes a
        "low mass bin". The default is 0.01, meaning a bin which has less than
        1% the total mass of the simulation.
    marker : str, optional
        The marker style for the high-mass bins. The default is 's', a square.
    small_marker : str, optional
        The marker style for the low-mass bins. The default is 'x'.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    small_label : str, optional
        The small scatter plot / error bar label. The default is None.
    markersize : float, optional
        The marker size. The default is 2.
    """
    stars = stars.dropna(how='any')
    # Lambda functions for weighted quantiles
    wm = lambda x: weighted_quantile(x, 'age', 'mstar', quantile=0.5)
    wl = lambda x: weighted_quantile(x, 'age', 'mstar', quantile=0.16)
    wu = lambda x: weighted_quantile(x, 'age', 'mstar', quantile=0.84)
    # Define [O/Fe] bins
    ofe_bins = np.arange(ofe_lim[0], ofe_lim[1]+ofe_bin_width, ofe_bin_width)
    # Mass-weighted median and standard deviation of ages
    grouped = group_by_bins(stars, '[o/fe]', bins=ofe_bins)
    age_median = grouped.apply(wm)
    age_lower = grouped.apply(wl)
    age_upper = grouped.apply(wu)
    # Separate bins with very little mass and plot with different marker
    mtot = grouped['mstar'].sum()
    high_mass_bins = mtot[mtot >= low_mass_cutoff * mtot.sum()].index
    ax.errorbar(age_median[high_mass_bins], high_mass_bins, 
                xerr=(age_median[high_mass_bins] - age_lower[high_mass_bins], 
                      age_upper[high_mass_bins] - age_median[high_mass_bins]),
                yerr=ofe_bin_width/2,
                color='k', linestyle='none', capsize=1, elinewidth=0.5,
                capthick=0.5, marker=marker, markersize=markersize, label=label
    )
    # Plot bins with little (but non-zero) stellar mass with smaller markers
    if plot_low_mass_bins:
        low_mass_bins = mtot[(mtot > 0) & 
                             (mtot < low_mass_cutoff * mtot.sum())].index
        ax.errorbar(age_median[low_mass_bins], low_mass_bins, 
                    xerr=(age_median[low_mass_bins] - age_lower[low_mass_bins], 
                          age_upper[low_mass_bins] - age_median[low_mass_bins]),
                    yerr=ofe_bin_width/2,
                    color='k', linestyle='none', capsize=1, elinewidth=0.25,
                    capthick=0.25, marker=small_marker, markersize=markersize, 
                    markeredgewidth=0.5, label=small_label
        )
    return age_median, age_lower, age_upper, ofe_bins


def plot_apogee_medians(ax, data, ofe_lim=OFE_LIM, ofe_bin_width=OFE_BIN_WIDTH,
                         plot_low_count_bins=True, low_count_cutoff=0.01,
                         marker='^', small_marker='2', label=None, 
                         small_label=None, markersize=2, age_col='LATENT_AGE'):
    """
    Plot median stellar ages binned by [O/Fe] from astroNN data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    data : pandas.DataFrame
        APOGEE data with astroNN ages, typically a subset of a galactic region.
    ofe_lim : tuple, optional
        Outermost bounds on [O/Fe]. The default is (-0.15, 0.55).
    ofe_bin_width : float, optional
        The [O/Fe] bin width in dex. The default is 0.05.
    plot_low_count_bins : bool, optional
        Whether to include bins with few stars. If True, these bins
        are shown with a different, smaller marker. The default is True.
    low_count_cutoff : float, optional
        The cutoff as a fraction of the total number of stars for what 
        constitutes a "low mass bin". The default is 0.01, meaning a bin which 
        has less than 1% the total number of stars.
    marker : str, optional
        The marker style for the high-count bins. The default is '^', an upwards
        pointing triangle.
    small_marker : str, optional
        The marker style for the low-count bins. The default is '2', an upwards
        pointing skinny triangle.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    small_label : str, optional
        The small scatter plot / error bar label. The default is None.
    markersize : float, optional
        The marker size. The default is 2.
    age_col : str, optional
        Name of column containing ages. The default is 'ASTRONN_AGE'
    """
    # Remove entries with no age estimate
    data.dropna(subset=age_col, inplace=True)
    # Define [O/Fe] bins
    ofe_bins = np.arange(ofe_lim[0], ofe_lim[1]+ofe_bin_width, ofe_bin_width)
    age_grouped = group_by_bins(data, 'O_FE', bins=ofe_bins)[age_col]
    age_median = age_grouped.median()
    age_lower = age_grouped.quantile(0.16)
    age_upper = age_grouped.quantile(0.84)
    # Separate bins with very little mass and plot with different markers
    counts = age_grouped.count()
    high_count_bins = counts[counts >= low_count_cutoff * counts.sum()].index
    ax.errorbar(age_median[high_count_bins], high_count_bins, 
                xerr=(age_median[high_count_bins] - age_lower[high_count_bins], 
                      age_upper[high_count_bins] - age_median[high_count_bins]),
                yerr=ofe_bin_width/2,
                color='r', linestyle='none', capsize=1, elinewidth=0.5,
                capthick=0.5, marker=marker, markersize=markersize, label=label,
    )
    # Plot bins with few (but non-zero) stars with a different, smaller marker
    if plot_low_count_bins:
        low_count_bins = counts[(counts > 0) & 
                                (counts < low_count_cutoff * counts.sum())].index
        ax.errorbar(age_median[low_count_bins], low_count_bins, 
                    xerr=(age_median[low_count_bins] - age_lower[low_count_bins], 
                          age_upper[low_count_bins] - age_median[low_count_bins]),
                    yerr=ofe_bin_width/2,
                    color='r', linestyle='none', capsize=1, elinewidth=0.25,
                    capthick=0.25, marker=small_marker, markersize=markersize, 
                    markeredgewidth=0.5, label=small_label,
        )
    return age_median, age_lower, age_upper, ofe_bins
    

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
    parser.add_argument('-l', '--log', action='store_true',
                        help='Plot age on a log scale')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    args = parser.parse_args()
    main(**vars(args))
