"""
This script compares the abundance evolution of multi-zone models with 
different yield sets and outflow settings.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from age_abundance_grid import plot_apogee_median_abundances, \
    plot_vice_median_abundances, AXES_MAJOR_LOCATOR, AXES_MINOR_LOCATOR
from scatter_plot_grid import plot_gas_abundance, setup_colorbar
from _globals import MAX_SF_RADIUS, END_TIME, ONE_COLUMN_WIDTH
from utils import vice_to_apogee_col, capitalize_abundance
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths

YIELD_SETS = ['yZ1', 'yZ2']
AXES_LIM = {
    '[o/h]': (-1.6, 0.6),
    '[fe/h]': (-1.6, 0.6),
    '[o/fe]': (-0.15, 0.5),
    'age': (-1, 14.99)
}


def main(uncertainties=True, verbose=False, style='paper', cmap_name='winter_r'):
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    solar_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 2))
    age_bins = np.arange(0, END_TIME + 2, 2)

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, 2, sharex=True, sharey='row', 
        figsize=(ONE_COLUMN_WIDTH, 1.5 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # Add colorbar
    birth_galr_bounds = [0, 2, 4, 6, 8, 10, 12, 14, MAX_SF_RADIUS]
    cbar = setup_colorbar(fig, cmap=cmap_name, bounds=birth_galr_bounds,
                          label=r'Birth $R_{\rm{gal}}$ [kpc]',
                          width=0.04, pad=0.02, labelpad=2)

    for j, yield_set in enumerate(YIELD_SETS):
        axs[0,j].set_title(yield_set)
        # Import VICE multizone outputs
        output_name = 'yields/%s/diskmodel' % yield_set
        mzs = MultizoneStars.from_output(output_name, verbose=verbose)
        mzs.region(galr_lim=(7, 9), absz_lim=(0, 2), inplace=True)
        # Model uncertainties
        if uncertainties:
            mzs.model_uncertainty(apogee_sample.data, inplace=True)
        for i, ycol in enumerate(['[o/h]', '[fe/h]', '[o/fe]']):
            mzs.scatter_plot(axs[i,j], 'age', ycol, color='galr_origin',
                             cmap=cmap_name, norm=cbar.norm)
            plot_gas_abundance(axs[i,j], mzs, 'lookback', ycol, ls='--', 
                               label='Gas abundance')
            plot_apogee_median_abundances(
                axs[i,j], apogee_sample, vice_to_apogee_col(ycol), age_bins, 
                offset=-0.2, label='L23 medians'
            )
            plot_vice_median_abundances(
                axs[i,j], mzs, ycol, age_bins, 
                offset=0.2, label='Model medians'
            )
            if j == 0:
                axs[i,j].set_ylabel(capitalize_abundance(ycol))
                axs[i,j].yaxis.set_major_locator(
                    MultipleLocator(AXES_MAJOR_LOCATOR[ycol])
                )
                axs[i,j].yaxis.set_minor_locator(
                    MultipleLocator(AXES_MINOR_LOCATOR[ycol])
                )
                axs[i,j].set_ylim(AXES_LIM[ycol])


    # Axes labels and formatting
    axs[0,0].set_xlim(AXES_LIM['age'])
    for ax in axs[-1,:]:
        ax.set_xlabel('Age [Gyr]')
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,-1].legend(loc='lower left', frameon=False, handletextpad=0.1,
                     borderpad=0.2, handlelength=1.5)
    
    fig.savefig(paths.figures / 'abundance_evolution_yields')
    plt.close()


if __name__ == '__main__':
    main()
