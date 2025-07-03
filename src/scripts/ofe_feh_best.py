"""
This script plots the [O/Fe]-[Fe/H] stellar distribution from VICE binned by
Galactocentric radius and midplane distance.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from scatter_plot_grid import setup_axes, setup_colorbar, plot_gas_abundance
from _globals import GALR_BINS, ABSZ_BINS, MAX_SF_RADIUS, TWO_COLUMN_WIDTH
import paths

FEH_LIM = (-1.3, 0.7)
OFE_LIM = (-0.199, 0.599)
GALR_BINS = [3, 5, 7, 9, 11, 13]
COLORMAP = 'jet'
OUTPUT_NAME = 'yZ2/best/cgm07_ratio025_eta18_migr36/diskmodel'
AGE_COL = 'CN_AGE' # for error forward-modeling


def main(style='paper'):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    # Import multioutput stars data
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    # Model observational uncertainties
    mzs.model_uncertainty(apogee_sample.data, inplace=True, age_col=AGE_COL)
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_axes(xlim=FEH_LIM, ylim=OFE_LIM, xlabel='[Fe/H]', 
                          ylabel='[O/Fe]', row_label_pos=(0.07, 0.87),
                          width=TWO_COLUMN_WIDTH, 
                          galr_bins=GALR_BINS, absz_bins=ABSZ_BINS)
    age_bins = np.arange(0, 12.1, 2)
    cbar = setup_colorbar(fig, cmap=COLORMAP, vmin=0, vmax=14, extend='max',
                          bounds=age_bins, label='Stellar age [Gyr]')
    
    ism_track_color = 'k'
    ism_track_width = 1
    ism_track_style = ':'
    apogee_contour_color = 'k'
    apogee_contour_width = 0.7
        
    for i, row in enumerate(axs):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            subset = mzs.region(galr_lim, absz_lim)
            # Stellar abundances
            subset.scatter_plot(ax, '[fe/h]', '[o/fe]', color='age',
                                cmap=COLORMAP, norm=cbar.norm, markersize=0.2)
            # Gas abundance tracks
            # plot_gas_abundance(ax, subset, '[fe/h]', '[o/fe]', 
            #                    lw=ism_track_width, c=ism_track_color, ls=ism_track_style)
            # APOGEE contours
            apogee_subset = apogee_sample.region(galr_lim, absz_lim)
            apogee_subset.plot_kde2D_contours(ax, 'FE_H', 'O_FE', 
                                              enclosed=[0.8, 0.3],
                                              c=apogee_contour_color, 
                                              lw=apogee_contour_width)
    
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    # Custom legend for APOGEE contours
    custom_lines = [
        Line2D([0], [0], color=apogee_contour_color, linestyle='-', linewidth=apogee_contour_width),
        Line2D([0], [0], color=apogee_contour_color, linestyle='--', linewidth=apogee_contour_width)
    ]
    legend_labels = ['30%', '80%']
    axs[0,-1].legend(custom_lines, legend_labels, title='Data contours',
                     loc='upper right', alignment='right')
    
    # Save
    plt.savefig(paths.figures / 'ofe_feh_best', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
