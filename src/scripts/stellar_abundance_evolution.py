"""
This script compares the stellar age-abundance relations predicted by several
multi-zone models against APOGEE DR17 data.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from age_abundance_grid import plot_apogee_median_abundances, \
    plot_vice_median_abundances, AXES_MAJOR_LOCATOR, AXES_MINOR_LOCATOR
from scatter_plot_grid import plot_gas_abundance, setup_colorbar
from _globals import ONE_COLUMN_WIDTH, TWO_COLUMN_WIDTH
from utils import vice_to_apogee_col, capitalize_abundance
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths

OUTPUT_NAMES = [
    'yZ1/fiducial/diskmodel',
    'yZ1/migration_strength/strength50/diskmodel',
    'yZ1/thick_thin_ratio/solar050/diskmodel',
    'yZ1/pre_enrichment/mh05_alpha00_eta06/diskmodel',
]
LABELS = [
    '(a)\nFiducial',
    '(b)\n' + r'$\sigma_{\rm RM8}=5.0$ kpc',
    '(c)\n' + r'$f_\Sigma(R_\odot)=0.5$',
    '(d)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.5$',
]
AXES_LIM = {
    '[o/h]': (-1.1, 0.4),
    '[fe/h]': (-1.1, 0.4),
    '[o/fe]': (-0.15, 0.5),
    'age': (-1, 14.99)
}
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)
CMAP = 'viridis_r'


def main(verbose=False, uncertainties=True, style='paper', cmap=CMAP, ages='L23'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        (TWO_COLUMN_WIDTH, 0.8 * TWO_COLUMN_WIDTH),
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM,
        age_col='%s_AGE' % ages
    )
    fig.suptitle(r'$y/Z_\odot=1$')
    fig.savefig(paths.figures / 'stellar_abundance_evolution')
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
    age_bins = {
        'L23_AGE': np.arange(0, 14.1, 2),
        'CN_AGE': np.arange(0, 10.1, 2)
    }[age_col]

    # Set up figure
    fig, axs = plt.subplots(
        3, len(output_names), sharex=True, sharey='row', 
        figsize=figsize,
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    fig.subplots_adjust(left=0.1, right=0.98)
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
    # data_label = 'APOGEE data'
    data_label = {
        'L23_AGE': 'APOGEE (NN ages)',
        'CN_AGE': 'APOGEE ([C/N] ages)'
    }[age_col]

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
                axs[i,j], mzs, 'lookback', ycol, ls='--', lw=1,
                label='Gas abundance'
            )
            stars = plot_vice_median_abundances(
                axs[i,j], mzs, ycol, age_bins, 
                label='Median stellar abundance'
            )
            spatch, pcol = plot_apogee_median_abundances(
                axs[i,j], solar_sample, vice_to_apogee_col(ycol), age_bins, 
                age_col=age_col, label=data_label, color='r',
            )
            if age_col == 'CN_AGE':
                # Plot >10 Gyr ages with hatched region (worse fit)
                plot_apogee_median_abundances(
                    axs[i,j], solar_sample, vice_to_apogee_col(ycol), 
                    np.arange(10, 14.1, 2.), 
                    age_col=age_col, label=data_label, color='r', 
                    hatch='///', facecolor='none', linestyle='--', alpha=0.3
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
        [lines[0], stars[0], (spatch, pcol)],
        ['Gas abundance', 'Stellar median', data_label],
        loc='lower left', frameon=False, handletextpad=0.5,
        borderpad=0.2, handlelength=1.2
    )

    return fig, axs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='stellar_abundance_evolution.py',
        description='Compare stellar age-abundance relations between multi-zone \
outputs with different parameters and APOGEE data.'
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
        default=CMAP,
        help='Name of colormap for color-coding model output (default: %s).' % CMAP
    )
    parser.add_argument(
        '--ages', 
        metavar='AGE-SOURCE', 
        type=str,
        default='L23',
        choices=['L23', 'CN'],
        help='Type of age estimate to use (default: L23).'
    )
    args = parser.parse_args()
    main(**vars(args))
