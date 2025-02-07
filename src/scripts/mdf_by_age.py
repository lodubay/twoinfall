"""
This script plots the MDF for stars of different ages in different radial bins.
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from utils import get_color_list, get_bin_centers, capitalize_abundance, weighted_quantile
from _globals import GALR_BINS, TWO_COLUMN_WIDTH
import paths

AGE_BINS = [0, 2, 4, 6, 8, 10]
NBINS = 100
SMOOTH_WIDTH = 0.1
FEH_LIM = (-1.0, 0.7)
YSCALE = 1e7

def main(output_name, **kwargs):
    mzs = MultizoneStars.from_output(output_name)
    plot_mdf_by_age(mzs, **kwargs)


def plot_mdf_by_age(mzs, col='[fe/h]', smoothing=SMOOTH_WIDTH, xlim=FEH_LIM, 
                    style='paper', cmap='Spectral_r', nbins=NBINS):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    colors = get_color_list(plt.get_cmap(cmap), AGE_BINS)

    fig, axs = plt.subplots(2, len(GALR_BINS)-1, 
                            figsize=(12, 3.5), 
                            sharex=True, gridspec_kw={'hspace': 0.})
    fig.suptitle(mzs.name)
    for i in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[i:i+2]
        axs[0,i].set_title('%s < R < %s' % tuple(galr_lim))
        axs[1,i].set_xlabel(capitalize_abundance(col))
        for j in range(len(AGE_BINS)-1):
            age_lim = AGE_BINS[j:j+2]
            subset = mzs.filter({'age': tuple(age_lim), 'galr_final': tuple(galr_lim), 'zfinal': (0, 0.5)})
            # Plot un-normalized MDFs
            mdf, bin_edges = subset.mdf(col, range=xlim, bins=nbins, 
                                        smoothing=smoothing, density=False)
            axs[0,i].plot(get_bin_centers(bin_edges), mdf / YSCALE, 
                          color=colors[j], linewidth=1, 
                          label='%s - %s' % tuple(age_lim))
            # Plot normalized MDFs
            mdf, bin_edges = subset.mdf(col, range=xlim, bins=nbins, 
                                        smoothing=smoothing, density=True)
            axs[1,i].plot(get_bin_centers(bin_edges), mdf, 
                          color=colors[j], linewidth=1)
            # Plot MDF widths
            per1 = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.16)
            med = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.5)
            per2 = weighted_quantile(subset.stars, val=col, weight='mstar', quantile=0.84)
            axs[1,i].errorbar(med, 5.0 - 0.1*j, xerr=[[per2-med], [med-per1]], fmt='none', ecolor=colors[j])

        axs[0,i].set_ylim((0, None))
        axs[1,i].set_ylim((0, 5.2))

    # Format axes
    axs[0,0].set_ylabel(f'Stellar Mass [x {YSCALE:.01e}]')
    axs[1,0].set_ylabel('Normalized MDF')
    axs[0,0].legend(frameon=False, title='Age Bin [Gyr]')
    axs[1,0].xaxis.set_minor_locator(MultipleLocator(0.2))
    axs[0,0].set_xlim(xlim)
    
    # Save
    simple_colname = col[1:-1].replace('/', '')
    fname = mzs.name.replace('diskmodel', ('%s_df_by_age.png' % simple_colname))
    fullpath = paths.extra / 'multizone' / fname
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='mdf_by_age.py',
        description='Plot the MDF in bins of age from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('--col', metavar='COL', type=str, default='[fe/h]',
                        help='Abundance column name (default: "[fe/h]")')
    parser.add_argument('--nbins', metavar='NBINS', type=int, default=NBINS,
                        help='Number of histogram bins (default: 100)')
    parser.add_argument('--xlim', metavar='XLIM', nargs=2, type=list, 
                        default=FEH_LIM,
                        help='Lower and upper bounds of the MDF ' + \
                             '(default: [-1.2, +0.7])')
    parser.add_argument('--smoothing', metavar='WIDTH', type=float,
                        default=SMOOTH_WIDTH,
                        help='Width of boxcar smoothing (default: 0.1)')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='Spectral_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: Spectral_r)')
    parser.add_argument('--style', metavar='STYLE', type=str, default='paper',
                        choices=['paper', 'poster', 'presentation'],
                        help='Plot style to use (default: paper)')
    args = parser.parse_args()
    main(**vars(args))
