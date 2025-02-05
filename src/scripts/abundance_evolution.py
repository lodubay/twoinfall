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
from _globals import MAX_SF_RADIUS, END_TIME, TWO_COLUMN_WIDTH
from utils import vice_to_apogee_col, capitalize_abundance
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths

OUTPUT_NAMES = [
    'yields/yZ1/diskmodel',
    'yields/yZ2/diskmodel',
    'migration_strength/strength50/diskmodel',
    'pre_enrichment/mh07_alpha00/diskmodel',
]
LABELS = [
    '(a)\nFiducial',
    '(b)\n' + r'$y/Z_\odot=2$',
    '(c)\n' + r'$\sigma_{\rm RM8}=5.0$ kpc',
    '(d)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.7$',
]
AXES_LIM = {
    '[o/h]': (-1.2, 0.4),
    '[fe/h]': (-1.2, 0.4),
    '[o/fe]': (-0.15, 0.5),
    'age': (-1, 14.99)
}
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)


def main(verbose=False, uncertainties=True, style='paper', cmap='winter_r'):
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    solar_sample = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    age_bins = np.arange(0, END_TIME + 2, 2)

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, 4, sharex=True, sharey='row', 
        figsize=(TWO_COLUMN_WIDTH, 0.7 * TWO_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # Add colorbar
    birth_galr_bounds = [3, 5, 7, 9, 11, 13]
    cbar = setup_colorbar(fig, cmap=cmap, bounds=birth_galr_bounds,
                          label=r'Birth $R_{\rm{gal}}$ [kpc]',
                          width=0.02, pad=0.01, labelpad=2, extend='both')

    for j, output_name in enumerate(OUTPUT_NAMES):
        axs[0,j].set_title(LABELS[j])
        # Import VICE multizone outputs
        mzs = MultizoneStars.from_output(output_name, verbose=verbose)
        mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM, inplace=True)
        # Model uncertainties
        if uncertainties:
            mzs.model_uncertainty(solar_sample.data, inplace=True)
        for i, ycol in enumerate(['[o/h]', '[fe/h]', '[o/fe]']):
            mzs.scatter_plot(axs[i,j], 'age', ycol, color='galr_origin',
                             cmap=cmap, norm=cbar.norm, markersize=0.5)
            lines = plot_gas_abundance(
                axs[i,j], mzs, 'lookback', ycol, ls='-', lw=1, 
                label='Gas abundance'
            )
            # plot_vice_median_abundances(
            #     axs[i,j], mzs, ycol, age_bins, 
            #     offset=0.2, label='Model medians'
            # )
            spatch, pcol = plot_apogee_median_abundances(
                axs[i,j], solar_sample, vice_to_apogee_col(ycol), age_bins, 
                age_col='L23_AGE', label='APOGEE data'#, color='Grey',
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
    axs[0,-1].legend(
        [lines[0], (spatch, pcol)],
        ['Gas abundance', 'APOGEE data'],
        loc='lower left', frameon=False, handletextpad=0.5,
        borderpad=0.2, handlelength=1.5
    )
    # cbar.ax.yaxis.set_major_locator(MultipleLocator(2))

    fig.savefig(paths.figures / 'abundance_evolution')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='abundance_evolution_yields.py',
        description='Compare age-abundance relations between multi-zone \
outputs with different yield sets and APOGEE data.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '--style', 
        metavar='STYLE', 
        type=str,
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default='winter_r',
        help='Name of colormap for color-coding model output (default: winter_r).'
    )
    args = parser.parse_args()
    main(**vars(args))
