"""
Plot a grid of [O/Fe] vs [Fe/H] at varying Galactic radii and z-heights.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import numpy as np
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from scatter_plot_grid import setup_axes, setup_colorbar, plot_gas_abundance
from _globals import GALR_BINS, ABSZ_BINS, ZONE_WIDTH, MAX_SF_RADIUS
import paths

FEH_LIM = (-1.3, 0.7)
OFE_LIM = (-0.15, 0.65)
_COLOR_OPTIONS = ['galr_origin', 'age']

def main(output_name, uncertainties=True, **kwargs):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    # Import multioutput stars data
    mzs = MultizoneStars.from_output(output_name)
    # Model observational uncertainties
    if uncertainties:
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
    plot_ofe_feh_grid(mzs, apogee_sample, **kwargs)


def plot_ofe_feh_grid(mzs, apogee_data, tracks=True, apogee_contours=True,
                      style='paper', cmap='winter_r', color_by='galr_origin',
                      fname='ofe_feh_grid.png'):
    color_by = color_by.lower()
    if color_by == 'galr_origin':
        cbar_label = r'Birth $R_{\rm{Gal}}$ [kpc]'
        cbar_lim = (0, MAX_SF_RADIUS)
    elif color_by == 'age':
        cbar_label = 'Stellar age [Gyr]'
        cbar_lim = (0, 13.5)
    else:
        raise ValueError('Parameter "color_by" must be one of %s' % 
                         _COLOR_OPTIONS)
    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_axes(xlim=FEH_LIM, ylim=OFE_LIM, xlabel='[Fe/H]', 
                          ylabel='[O/Fe]', row_label_pos=(0.07, 0.85),
                          title=mzs.name, width=8, galr_bins=GALR_BINS)
    cbar = setup_colorbar(fig, cmap=cmap, vmin=cbar_lim[0], vmax=cbar_lim[1], 
                          label=cbar_label)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(2))
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ism_track_color = 'k'
    ism_track_width = 0.5
        
    for i, row in enumerate(axs):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            subset = mzs.region(galr_lim, absz_lim)
            if subset.nstars:
                subset.scatter_plot(ax, '[fe/h]', '[o/fe]', color=color_by,
                                    cmap=cmap, norm=cbar.norm)
            if tracks:
                plot_gas_abundance(ax, subset, '[fe/h]', '[o/fe]')
            if apogee_contours:
                apogee_subset = apogee_data.region(galr_lim, absz_lim)
                apogee_subset.plot_kde2D_contours(ax, 'FE_H', 'O_FE')
    
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    # Custom legend
    custom_lines = [Line2D([0], [0], color=ism_track_color, linestyle='-', 
                           linewidth=ism_track_width)]
    legend_labels = ['Gas abundance']
    if apogee_contours:
        custom_lines += [
            Line2D([0], [0], color='r', linestyle='-', linewidth=0.5),
            Line2D([0], [0], color='r', linestyle='--', linewidth=0.5)]
        legend_labels += ['APOGEE 30% cont.', 'APOGEE 80% cont.']
    axs[2,-1].legend(custom_lines, legend_labels, frameon=False, 
                     loc='upper right', handlelength=0.6, handletextpad=0.4)
    
    # Save
    fullpath = paths.extra / mzs.name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ofe_feh_grid.py',
        description='Generate a grid of [O/Fe] vs [Fe/H] scatterplots ' + \
            'from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('-t', '--tracks', action='store_true',
                        help='Plot ISM tracks in addition to stellar abundances')
    parser.add_argument('-a', '--apogee-contours', action='store_true',
                        help='Plot contour lines from APOGEE data')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='winter_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: winter_r)')
    parser.add_argument('-s', '--style', 
                        choices=['paper', 'poster'],
                        default='paper', 
                        help='Plot style to use (default: paper)')
    parser.add_argument('-c', '--color-by', 
                        choices=_COLOR_OPTIONS, default='galr_origin',
                        help='Output parameter to assign color-coding ' + \
                             '(default: galr_origin).')
    args = parser.parse_args()
    main(**vars(args))
