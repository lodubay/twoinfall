"""
This script plots the MDF width as a function of age and radius.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from utils import get_color_list, get_bin_centers, capitalize_abundance, weighted_quantile
from _globals import GALR_BINS, ONE_COLUMN_WIDTH
import paths


def main(output_name, **kwargs):
    mzs = MultizoneStars.from_output(output_name)
    plot_mdf_widths(mzs, **kwargs)


def plot_mdf_widths(mzs, col='[fe/h]', style='paper', cmap='plasma_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH))
    fig.suptitle(mzs.name)

    age_bins = np.arange(14)
    for i in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[i:i+2]
        sigma_feh = []
        for j in range(len(age_bins)-1):
            age_lim = age_bins[j:j+2]
            subset = mzs.filter({'age': tuple(age_lim), 'galr_final': tuple(galr_lim), 'zfinal': (0, 0.5)})
            # Calculate MDF widths
            per1 = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.16)
            med = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.5)
            per2 = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.84)
            sigma_feh.append((per2 - per1) / 2)
        ax.plot(get_bin_centers(age_bins), sigma_feh, color=colors[i], label=str(galr_lim))

    # Format axes
    ax.set_xlabel('Age [Gyr]')
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel(r'$\sigma_{\rm %s}$' % capitalize_abundance(col))
    ax.legend(frameon=False, title='Radial bin [kpc]')
    
    # Save
    simple_colname = col[1:-1].replace('/', '')
    fname = mzs.name.replace('diskmodel', ('%s_df_widths.png' % simple_colname))
    fullpath = paths.extra / fname
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='mdf_widths.py',
        description='Plot the MDF in bins of age from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    # parser.add_argument('-u', '--uncertainties', action='store_true',
    #                     help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('--col', metavar='COL', type=str, default='[fe/h]',
                        help='Abundance column name (default: "[fe/h]")')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='plasma_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: plasma_r)')
    parser.add_argument('--style', metavar='STYLE', type=str, default='paper',
                        choices=['paper', 'poster', 'presentation'],
                        help='Plot style to use (default: paper)')
    args = parser.parse_args()
    main(**vars(args))
