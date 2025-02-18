"""
This script plots the evolution of the MDF with age across different Galactic
radii for APOGEE and several multi-zone models.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from scatter_plot_grid import setup_colorbar
from utils import get_color_list, get_bin_centers, capitalize_abundance, vice_to_apogee_col
from _globals import TWO_COLUMN_WIDTH
import paths

GALR_BINS = [3, 5, 7, 9, 11, 13]
AGE_BINS = [0, 2, 4, 6, 8, 10]
# AGE_BINS = [0, 2, 4, 6, 8, 10, 13]
# AGE_BINS = [0, 1, 3, 5, 7, 9, 13]
ABSZ_LIM = (0, 0.5)
NBINS = 100
SMOOTH_WIDTH = 0.1
# FEH_LIM = (-1.1, 0.6)
FEH_LIM = (-0.8, 0.6)
YSCALE = 1e7
AGE_COL = 'CN_AGE'

OUTPUT_NAMES = [
    'yZ1/pre_enrichment/mh07_alpha00/diskmodel',
    'yZ2/pre_enrichment/mh07_alpha00/diskmodel'
]
MODEL_LABELS = [
    r'(a) $y/Z_\odot=1$, ${\rm [X/H]}_{\rm CGM}=-0.7$',
    r'(b) $y/Z_\odot=2$, ${\rm [X/H]}_{\rm CGM}=-0.7$'
]

def main(style='paper', col='[fe/h]', cmap='coolwarm', smoothing=SMOOTH_WIDTH):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, len(GALR_BINS)-1,
        figsize=(TWO_COLUMN_WIDTH, 0.5 * TWO_COLUMN_WIDTH), 
        sharex=True, sharey='row', 
        gridspec_kw={'hspace': 0.25, 'wspace': 0.1}
    )
    # Plot multizone outputs
    apogee_data = APOGEESample.load()
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        # mzs.model_uncertainty(apogee_data=apogee_data.data)
        plot_mdf_evolution(mzs, axs[i], col=col, smoothing=smoothing, cmap=cmap,
                           title=MODEL_LABELS[i])
    # Plot APOGEE data
    plot_mdf_evolution(apogee_data, axs[-1], col=vice_to_apogee_col(col), 
                       smoothing=smoothing, cmap=cmap, 
                       title='(c) APOGEE ([C/N]-derived ages)', 
                       age_col=AGE_COL, indicate_mode=True)
    
    # Add colorbar at left
    cbar = setup_colorbar(fig, cmap=cmap, bounds=AGE_BINS,
                          label=r'Stellar age [Gyr]',
                          width=0.015, pad=0.015, labelpad=2)
    
    for i in range(len(axs[0,:])):
        galr_lim = GALR_BINS[i:i+2]
        # Column labels
        axs[0,i].set_title(r'$%s \leq R_{\rm gal} < %s$ kpc' % tuple(galr_lim),
                           size=plt.rcParams['font.size'], pad=10)
        axs[-1,i].set_xlabel(capitalize_abundance(col))
    
    # Remove spines and y-axis labels
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

    # Format axes
    for ax in axs[:,0]:
        ax.set_ylim((0, None))
    axs[1,0].set_ylabel('Normalized PDF')
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].set_xlim(FEH_LIM)
    
    plt.savefig(paths.figures / 'mdf_evolution')
    plt.close()


def plot_mdf_evolution(obj, axs, col='[fe/h]', smoothing=SMOOTH_WIDTH, 
                       xlim=FEH_LIM, cmap='Spectral_r', nbins=NBINS, title='',
                       age_col='age', indicate_mode=False):
    """
    Plot the MDF in bins of Rgal for different ages.
    """
    colors = get_color_list(plt.get_cmap(cmap), AGE_BINS)
    # Row label with white background
    axs[2].text(
        0.5, 1.1, title, 
        ha='center', va='top', 
        transform=axs[2].transAxes,
        size=plt.rcParams['axes.titlesize'],
        bbox={
            'facecolor': 'w',
            'edgecolor': 'none',
            'boxstyle': 'round',
            'pad': 0.15,
            'alpha': 1.,
        }
    )
    for i in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[i:i+2]
        region_subset = obj.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
        # for j in reversed(list(range(len(AGE_BINS)-1))):
        for j in range(len(AGE_BINS)-1):
            age_lim = AGE_BINS[j:j+2]
            age_subset = region_subset.filter({age_col: tuple(age_lim)})
            # Plot normalized MDFs
            if age_subset.nstars > 100:
                mdf, bin_edges = age_subset.mdf(
                    col, range=xlim, bins=nbins, smoothing=smoothing
                )
                axs[i].plot(
                    get_bin_centers(bin_edges), mdf, 
                    color=colors[j], linewidth=1, #alpha=alpha,
                    label='%s - %s' % tuple(age_lim)
                )
                if indicate_mode and j == 0:
                    mode = np.argmax(mdf)
                    axs[i].axvline(bin_edges[mode:mode+2].mean(), 
                                   color='gray', ls=':')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='mdf_evolution.py',
        description='Plot the MDF in bins of age from a VICE multizone run.'
    )
    parser.add_argument('--col', metavar='COL', type=str, default='[fe/h]',
                        help='Abundance column name (default: "[fe/h]")')
    parser.add_argument('--smoothing', metavar='WIDTH', type=float,
                        default=SMOOTH_WIDTH,
                        help='Width of boxcar smoothing (default: 0.1)')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='jet',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: jet)')
    parser.add_argument('--style', metavar='STYLE', type=str, default='paper',
                        choices=['paper', 'poster', 'presentation'],
                        help='Plot style to use (default: paper)')
    args = parser.parse_args()
    main(**vars(args))
