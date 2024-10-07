"""
Plot a grid of [Fe/H] vs age at varying Galactic radii and z-heights.
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from scatter_plot_grid import setup_axes, setup_colorbar, plot_gas_abundance
from _globals import ABSZ_BINS, GALR_BINS, MAX_SF_RADIUS
import paths

FEH_LIM = (-1.5, 0.7)
AGE_LIM = (0, 14)

def main(output_name, uncertainties=True, **kwargs):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    # Import multioutput stars data
    mzs = MultizoneStars.from_output(output_name)
    # Model observational uncertainties
    if uncertainties:
        mzs.model_uncertainty(inplace=True)
    plot_age_feh_grid(mzs, apogee_sample, **kwargs)


def plot_age_feh_grid(mzs, apogee_data, cmap='winter_r', uncertainties=True, 
                      tracks=True, style='paper', apogee_medians=True):
    plt.style.use(paths.styles / ('%s.mplstyle' % style))
    fig, axs = setup_axes(xlim=AGE_LIM, ylim=FEH_LIM, xlabel='Age [Gyr]', 
                          ylabel='[Fe/H]', row_label_pos=(0.18, 0.85),
                          row_label_col = 5,
                          title=mzs.name, galr_bins=GALR_BINS)
    cbar = setup_colorbar(fig, cmap=cmap, vmin=0, vmax=MAX_SF_RADIUS, 
                          label=r'Birth $R_{\rm{Gal}}$ [kpc]')
    cbar.ax.yaxis.set_major_locator(MultipleLocator(2))
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        
    for i, row in enumerate(axs):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            subset = mzs.region(galr_lim, absz_lim)
            subset.scatter_plot(ax, 'age', '[fe/h]', color='galr_origin',
                                cmap=cmap, norm=cbar.norm)
            if tracks:
                plot_gas_abundance(ax, subset, 'lookback', '[fe/h]')
    
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Save
    fname = mzs.name.replace('diskmodel', 'age_feh_grid.png')
    fullpath = paths.extra / fname
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='age_feh_grid.py',
        description='Generate a grid of age vs [Fe/H] scatterplots ' + \
            'from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('-t', '--tracks', action='store_true',
                        help='Plot ISM tracks in addition to stellar abundances')
    parser.add_argument('-a', '--apogee-medians', action='store_true',
                        help='Plot median ages from APOGEE data')
    parser.add_argument('-s', '--style', 
                        choices=['paper', 'poster'],
                        default='paper', 
                        help='Plot style to use (default: paper)')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='winter_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: winter_r)')
    args = parser.parse_args()
    main(**vars(args))
