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
AGE_BINS = [1, 2, 4, 6, 8, 10]
ABSZ_LIM = (0, 0.5)
NBINS = 100
SMOOTH_WIDTH = 0.1
XLIM = (-0.9, 0.6)
YSCALE = 1e7
AGE_COL = 'CN_AGE'
CMAP = 'turbo'

def main(style='paper', col='[fe/h]', cmap=CMAP, smoothing=SMOOTH_WIDTH):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, len(GALR_BINS)-1,
        figsize=(TWO_COLUMN_WIDTH, 0.5 * TWO_COLUMN_WIDTH), 
        sharex=True, sharey='row', 
        gridspec_kw={'hspace': 0.25, 'wspace': 0.1, 'left': 0.}
    )
    # Get list of line colors
    colors = get_color_list(plt.get_cmap(cmap), AGE_BINS)
    gray = '0.5'
    # Add colorbar at left
    cbar = setup_colorbar(fig, cmap=cmap, bounds=AGE_BINS,
                          label=r'Stellar age [Gyr]',
                          width=0.015, pad=0.015, labelpad=2)

    # Plot multizone outputs
    output_names = [
        'yZ1-fiducial/diskmodel',
        'yZ2-fiducial/diskmodel'
    ]
    model_labels = [r'(a) $y/Z_\odot=1$ model', r'(a) $y/Z_\odot=2$ model']
    apogee_data = APOGEESample.load()
    for i, output_name in enumerate(output_names):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_data=apogee_data.data, age_col=AGE_COL, inplace=True)
        axes_title_background(axs[i,2], model_labels[i])
        for j in range(len(GALR_BINS)-1):
            galr_lim = GALR_BINS[j:j+2]
            region_subset = mzs.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
            # plot total MDF
            mdf, bin_edges = region_subset.mdf(
                col, range=XLIM, bins=NBINS, smoothing=smoothing
            )
            axs[i,j].plot(
                get_bin_centers(bin_edges), mdf, 
                color=gray, linewidth=2,
                label='All stars'
            )
            for k in range(len(AGE_BINS)-1):
                age_lim = AGE_BINS[k:k+2]
                age_subset = region_subset.filter({'age': tuple(age_lim)})
                # Plot normalized MDFs
                mdf, bin_edges = age_subset.mdf(
                    col, range=XLIM, bins=NBINS, smoothing=smoothing
                )
                axs[i,j].plot(
                    get_bin_centers(bin_edges), mdf, 
                    color=colors[k], linewidth=1,
                    label='%s - %s' % tuple(age_lim)
                )
    
    # Plot APOGEE data
    if AGE_COL == 'L23_AGE':
        axes_title_background(axs[-1,2], '(c) APOGEE (L23 ages)')
    else:
        axes_title_background(axs[-1,2], '(c) APOGEE ([C/N]-based ages)')
    for j in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[j:j+2]
        region_subset = apogee_data.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
        # plot overall MDF for region
        mdf, bin_edges = region_subset.mdf(
            vice_to_apogee_col(col), 
            range=XLIM, bins=NBINS, smoothing=smoothing
        )
        axs[-1,j].plot(
            get_bin_centers(bin_edges), mdf, 
            color=gray, linewidth=2,
            label='All stars'
        )
        if AGE_COL == 'CN_AGE':
            # Indicate cut below [Fe/H] < -0.4 for URGB and RC stars
            axs[-1,j].axvline(-0.4, color=gray, ls='--')
            # Indicate hard cuts in [Fe/H] for all [C/N]-based ages
            if XLIM[0] < -0.9:
                axs[-1,j].axvline(-0.9, color=gray, ls='-')
            axs[-1,j].axvline(0.45, color=gray, ls='-')
        youngest_mode = False
        for k in range(len(AGE_BINS)-1):
            age_lim = AGE_BINS[k:k+2]
            age_subset = region_subset.filter({AGE_COL: tuple(age_lim)})
            # Plot normalized MDFs
            if age_subset.nstars > 100 and age_lim[0] >= 1 and age_lim[1] <= 10:
                mdf, bin_edges = age_subset.mdf(
                    vice_to_apogee_col(col), 
                    range=XLIM, bins=NBINS, smoothing=smoothing
                )
                axs[-1,j].plot(
                    get_bin_centers(bin_edges), mdf, 
                    color=colors[k], linewidth=1, #alpha=alpha,
                    label='%s - %s' % tuple(age_lim)
                )
                # Indicate mode of youngest bin
                if not youngest_mode:
                    mode = np.argmax(mdf)
                    axs[-1,j].axvline(bin_edges[mode:mode+2].mean(), 
                                   color=colors[k], ls=':')
                    youngest_mode = True
    
    for i in range(len(axs[0,:])):
        galr_lim = GALR_BINS[i:i+2]
        # Column labels
        axs[0,i].set_title(
            r'$%s \leq R_{\rm gal} < %s$ kpc' % tuple(galr_lim) + '\n' + r'$|z|<0.5$ kpc',
            size=plt.rcParams['font.size'], pad=10
        )
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
    axs[0,0].set_xlim(XLIM)
    
    plt.savefig(paths.figures / 'mdf_evolution')
    plt.close()


def axes_title_background(ax, text, bgcolor='w', x=0.5, y=1.12):
    ax.text(
        x, y, text, 
        ha='center', va='top', 
        transform=ax.transAxes,
        size=plt.rcParams['axes.titlesize'],
        bbox={
            'facecolor': bgcolor,
            'edgecolor': 'none',
            'boxstyle': 'round',
            'pad': 0.1,
            'alpha': 1.,
        }
    )


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
                        default=CMAP,
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: %s)' % CMAP)
    parser.add_argument('--style', metavar='STYLE', type=str, default='paper',
                        choices=['paper', 'poster', 'presentation'],
                        help='Plot style to use (default: paper)')
    args = parser.parse_args()
    main(**vars(args))
