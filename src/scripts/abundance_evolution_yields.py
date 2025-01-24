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


def main(verbose=False, style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plot_abundance_evolution_comparison(
        [f'yields/{yield_set}/diskmodel' for yield_set in YIELD_SETS],
        labels=[f'{yield_set} yields' for yield_set in YIELD_SETS],
        uncertainties=True, verbose=verbose, cmap_name=cmap
    )
    
    fig.savefig(paths.figures / 'abundance_evolution_yields')
    plt.close()


def plot_abundance_evolution_comparison(
        output_names, 
        labels=[], 
        uncertainties=True, 
        verbose=False, 
        cmap_name='winter_r'
    ):
    """
    Compare the abundance evolution of two multi-zone models.
    
    Parameters
    ----------
    output_names : list of strings of length 2
        The two output names of VICE multi-zone models to compare.
    labels : list of strings of length 2, optional
        The column labels for each output. If an empty list, a part of
        the output name is used. The default is [].
    uncertainties : bool, optional
        Whether to forward-model observational uncertainties in the model
        outputs. The default is True.
    verbose : bool, optional
        Whether to print verbose output to terminal. The default is False.
    cmap_name : str, optional
        Name of colormap to color the points by birth radius. The default is
        'winter_r'.
    """
    assert len(output_names) == 2
    # Import APOGEE and astroNN data
    apogee_sample = APOGEESample.load()
    solar_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 2))
    age_bins = np.arange(0, END_TIME + 2, 2)

    # Set up figure
    fig, axs = plt.subplots(
        3, 2, sharex=True, sharey='row', 
        figsize=(ONE_COLUMN_WIDTH, 1.67 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # Add colorbar
    birth_galr_bounds = [0, 2, 4, 6, 8, 10, 12, 14, MAX_SF_RADIUS]
    cbar = setup_colorbar(fig, cmap=cmap_name, bounds=birth_galr_bounds,
                          label=r'Birth $R_{\rm{gal}}$ [kpc]',
                          width=0.02, pad=0.04, labelpad=2,
                          orientation='horizontal')

    for j, output_name in enumerate(output_names):
        if len(labels) == len(output_names):
            axs[0,j].set_title(labels[j])
        else:
            axs[0,j].set_title(output_name.split('/')[1])
        # Import VICE multizone outputs
        mzs = MultizoneStars.from_output(output_name, verbose=verbose)
        mzs.region(galr_lim=(7, 9), absz_lim=(0, 2), inplace=True)
        # Model uncertainties
        if uncertainties:
            mzs.model_uncertainty(solar_sample.data, inplace=True)
        for i, ycol in enumerate(['[o/h]', '[fe/h]', '[o/fe]']):
            mzs.scatter_plot(axs[i,j], 'age', ycol, color='galr_origin',
                             cmap=cmap_name, norm=cbar.norm)
            plot_gas_abundance(axs[i,j], mzs, 'lookback', ycol, ls='--', 
                               label='Gas abundance')
            plot_apogee_median_abundances(
                axs[i,j], solar_sample, vice_to_apogee_col(ycol), age_bins, 
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
