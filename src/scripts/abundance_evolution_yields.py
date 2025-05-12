"""
This script compares the abundance evolution of multi-zone models with 
different yield sets and outflow settings.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from age_abundance_grid import plot_apogee_median_abundances, \
    AXES_MAJOR_LOCATOR, AXES_MINOR_LOCATOR
from scatter_plot_grid import plot_gas_abundance, setup_colorbar
from _globals import END_TIME, ONE_COLUMN_WIDTH
from utils import vice_to_apogee_col, capitalize_abundance
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths

OUTPUT_NAMES = [
    'yZ1/fiducial/diskmodel',
    'yZ2/fiducial/diskmodel'
]
LABELS = [
    '(a)\n' + r'$y/Z_\odot=1$',
    '(b)\n' + r'$y/Z_\odot=2$'
]
POSTER_LABELS = [
    'Low yields\n& outflows\n' + r'$(y/Z_\odot=1)$',
    'High yields\n& outflows\n' + r'$(y/Z_\odot=2)$',
]
# LABEL_PADS = [18, 18, 6, 6, 6]
AXES_LIM = {
    '[o/h]': (-1.2, 0.4),
    '[fe/h]': (-1.2, 0.4),
    '[o/fe]': (-0.15, 0.5),
    'age': (-1, 14.99)
}
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)


def main(verbose=False, uncertainties=True, style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        {'paper': LABELS, 'poster': POSTER_LABELS}[style],
        (ONE_COLUMN_WIDTH, 1.67 * ONE_COLUMN_WIDTH),
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        cbar_orientation='horizontal',
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM
    )
    savedir = {
        'paper': paths.figures,
        'poster': paths.extra / 'poster'
    }[style]
    if not savedir.is_dir():
        savedir.mkdir()
    fig.savefig(savedir / 'abundance_evolution_yields')
    plt.close()

def compare_abundance_evolution(
        output_names, 
        labels,
        figsize, 
        uncertainties=True, 
        cmap='winter_r', 
        label_pads=[], 
        verbose=False,
        cbar_orientation='vertical',
        galr_lim=(7, 9),
        absz_lim=(0, 0.5),
        age_col='L23_AGE'
    ):
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    solar_sample = apogee_sample.region(galr_lim=galr_lim, absz_lim=absz_lim)
    age_bins = np.arange(0, END_TIME + 2, 2)

    # Set up figure
    fig, axs = plt.subplots(
        3, len(output_names), sharex=True, sharey='row', 
        figsize=figsize,
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    if label_pads == []:
        label_pads = [None] * len(output_names)
    # Add colorbar
    birth_galr_bounds = [3, 5, 7, 9, 11, 13]
    if cbar_orientation == 'horizontal':
        cbar_width = 0.02
        cbar_pad = 0.04
    else:
        cbar_width = 0.04 * ONE_COLUMN_WIDTH / figsize[0]
        cbar_pad = 0.02 * ONE_COLUMN_WIDTH / figsize[0]
    cbar = setup_colorbar(fig, cmap=cmap, bounds=birth_galr_bounds,
                          label=r'Birth $R_{\rm{gal}}$ [kpc]',
                          orientation=cbar_orientation,
                          # scale colorbar width with figure width
                          width=cbar_width, pad=cbar_pad, 
                          labelpad=2, extend='both')

    for j, output_name in enumerate(output_names):
        axs[0,j].set_title(labels[j], pad=label_pads[j])
        # Import VICE multizone outputs
        mzs = MultizoneStars.from_output(output_name, verbose=verbose)
        mzs.region(galr_lim=galr_lim, absz_lim=absz_lim, inplace=True)
        # Model uncertainties
        if uncertainties:
            mzs.model_uncertainty(
                solar_sample.data, inplace=True, age_col=age_col
            )
        for i, ycol in enumerate(['[o/h]', '[fe/h]', '[o/fe]']):
            mzs.scatter_plot(axs[i,j], 'age', ycol, color='galr_origin',
                             cmap=cmap, norm=cbar.norm, markersize=0.5)
            lines = plot_gas_abundance(
                axs[i,j], mzs, 'lookback', ycol, ls='-', lw=1, 
                label='Gas abundance'
            )
            spatch, pcol = plot_apogee_median_abundances(
                axs[i,j], solar_sample, vice_to_apogee_col(ycol), age_bins, 
                age_col=age_col, label='APOGEE data', color='r',
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
        borderpad=0.2, handlelength=1.2
    )

    return fig, axs


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
        choices=('paper', 'poster'),
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
