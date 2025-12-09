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
    'yZ1-fiducial/diskmodel',
    'yZ1-migration/diskmodel',
    'yZ1-diskratio/diskmodel',
    'yZ1-preenrich/diskmodel',
]
LABELS = [
    '(a)\nFiducial',
    '(b)\n' + r'$\sigma_{\rm RM8}=5.0$ kpc',
    '(c)\n' + r'$f_\Sigma(R_\odot)=0.5$',
    '(d)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.5$',
]
AXES_LIM = {
    '[o/h]': (-1.1, 0.4),
    '[o/h]_res': (-0.7, 0.4),
    '[fe/h]': (-1.1, 0.4),
    '[fe/h]_res': (-0.7, 0.4),
    '[o/fe]': (-0.15, 0.5),
    '[o/fe]_res': (-0.4, 0.2),
    'age': (-1, 14.99)
}
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)
CMAP = 'viridis_r'


def main(verbose=False, uncertainties=True, residuals=False, style='paper', cmap=CMAP, ages='L23'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    if residuals:
        figsize = (TWO_COLUMN_WIDTH, 1.0 * TWO_COLUMN_WIDTH)
        titley = 0.95
    else:
        figsize = (TWO_COLUMN_WIDTH, 0.8 * TWO_COLUMN_WIDTH)
        titley = 0.98
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        figsize,
        residuals=residuals,
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM,
        age_col='%s_AGE' % ages
    )
    fig.suptitle(r'$y/Z_\odot=1$', y=titley)
    fig.savefig(paths.figures / 'stellar_abundance_evolution')
    plt.close()


def compare_abundance_evolution(
        output_names, 
        labels,
        figsize, 
        residuals=False,
        uncertainties=True, 
        cmap='winter_r', 
        label_pads=[], 
        verbose=False,
        cbar_orientation='vertical',
        galr_lim=(7, 9),
        absz_lim=(0, 0.5),
        age_col='L23_AGE'
    ):
    # Import APOGEE and age data
    apogee_sample = APOGEESample.load()
    solar_sample = apogee_sample.region(galr_lim=galr_lim, absz_lim=absz_lim)
    age_bins = {
        'L23_AGE': np.arange(0, 14.1, 2),
        'CN_AGE': np.arange(0, 10.1, 2)
    }[age_col]
    # Sort by ascending age
    apogee_sorted_ages = apogee_sample.data.dropna(subset=age_col).sort_values(age_col)[
        [age_col, 'O_H', 'FE_H', 'O_FE']
    ]
    # Calculate APOGEE rolling median
    rolling_params = dict(
        min_periods=100, step=100, on=age_col, center=True
    )
    apogee_rolling_medians = apogee_sorted_ages.rolling(1000, **rolling_params).median()

    # Set up figure
    if residuals:
        nrows = 6
        height_ratios = (2, 1, 2, 1, 2, 1)
    else:
        nrows = 3
        height_ratios = None
    fig, axs = plt.subplots(
        nrows, len(output_names), sharex=True, sharey='row', 
        figsize=figsize, height_ratios=height_ratios,
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
        ycols = ['[o/h]', '[fe/h]', '[o/fe]']
        for i, ycol in enumerate(['[o/h]', '[fe/h]', '[o/fe]']):
            apo_col = vice_to_apogee_col(ycol)
            # Plot residuals in next panel down
            if residuals:
                row = 2*i
                # Subtract APOGEE running median from VICE data
                mzs.stars['%s_res' % ycol] = mzs.stars[ycol] - np.interp(
                    mzs.stars['age'], 
                    apogee_rolling_medians[age_col], 
                    apogee_rolling_medians[apo_col]
                )
                # Scatter plot residual abundances
                mzs.scatter_plot(
                    axs[row+1,j], 'age', '%s_res' % ycol, color='galr_origin',
                    cmap=cmap, norm=cbar.norm, markersize=0.5
                )
                # Running median of residuals
                vice_running_median(
                    axs[row+1,j], mzs, '%s_res' % ycol,
                )
                # Shade APOGEE 1-sigma region
                apogee_running_median(
                    axs[row+1,j], solar_sample, apo_col, residuals=True,
                    age_col=age_col, label=data_label, color='r',
                )
                if age_col == 'CN_AGE':
                    # Plot >10 Gyr ages with hatched region (worse fit)
                    apogee_running_median(
                        axs[row+1,j], solar_sample, apo_col, 
                        age_col=age_col, color='r', residuals=True,
                        hatch='///', facecolor='none', linestyle='--', alpha=0.3
                    )
                if j == 0:
                    axs[row+1,j].set_ylabel('Residuals', size='small')
                    axs[row+1,j].yaxis.set_major_locator(
                        MultipleLocator(AXES_MAJOR_LOCATOR[ycol])
                    )
                    axs[row+1,j].yaxis.set_minor_locator(
                        MultipleLocator(AXES_MINOR_LOCATOR[ycol])
                    )
                    axs[row+1,j].set_ylim(AXES_LIM['%s_res' % ycol])
            else:
                row = i
            mzs.scatter_plot(axs[row,j], 'age', ycol, color='galr_origin',
                             cmap=cmap, norm=cbar.norm, markersize=0.5)
            lines = plot_gas_abundance(
                axs[row,j], mzs, 'lookback', ycol, ls='--', lw=1,
                label='Gas abundance'
            )
            stars = vice_running_median(
                axs[row,j], mzs, ycol,
                label='Median stellar abundance'
            )
            spatch, pcol = apogee_running_median(
                axs[row,j], solar_sample, apo_col, 
                age_col=age_col, label=data_label, color='r',
            )
            if age_col == 'CN_AGE':
                # Plot >10 Gyr ages with hatched region (worse fit)
                apogee_running_median(
                    axs[row,j], solar_sample, apo_col, 
                    age_col=age_col, color='r', 
                    hatch='///', facecolor='none', linestyle='--', alpha=0.3
                )
            if j == 0:
                axs[row,j].set_ylabel(capitalize_abundance(ycol))
                axs[row,j].yaxis.set_major_locator(
                    MultipleLocator(AXES_MAJOR_LOCATOR[ycol])
                )
                axs[row,j].yaxis.set_minor_locator(
                    MultipleLocator(AXES_MINOR_LOCATOR[ycol])
                )
                axs[row,j].set_ylim(AXES_LIM[ycol])

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


def apogee_running_median(
        ax, 
        apogee_sample, 
        col, 
        label=None, 
        color='r', 
        age_col='L23_AGE', 
        window=1000,
        alpha=0.2, 
        linestyle='-', 
        marker='o', 
        residuals=False,
        **kwargs):
    """
    Plot APOGEE stellar abundance medians and 1-sigma range binned by age.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    apogee_sample : APOGEESample instance
        APOGEE sample data or subset of the sample.
    col : str
        Data column with abundance data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    age_col : str, optional
        Name of column containing ages. The default is 'L23_AGE'.
    window : int, optional
        Rolling window size. The default is 1000.
    alpha : float, optional
        Transparency of the 1-sigma range. The default is 0.3.
    residuals : bool, optional
        If True, subtract running median from 1-sigma limits.
        The default is False.
    **kwargs passed to matplotlib.pyplot.fill_between
    
    Returns
    -------
    spatch : matplotlib.patches.StepPatch
    pcol : matplotlib.collections.FillBetweenPolyCollection
    """
    # Sort by ascending age
    sorted_ages = apogee_sample.data.sort_values(age_col)[[age_col, col]]
    # Calculate rolling median
    rolling_params = dict(
        min_periods=int(window/10), step=int(window/10), on=age_col, center=True
    )
    rolling_medians = sorted_ages.rolling(window, **rolling_params).median()
    # Rolling 16th and 84th percentiles
    rolling_low = sorted_ages.rolling(window, **rolling_params).quantile(0.16)
    rolling_high = sorted_ages.rolling(window, **rolling_params).quantile(0.84)
    if residuals:
        rolling_low[col] -= rolling_medians[col]
        rolling_high[col] -= rolling_medians[col]
        rolling_medians[col] -= rolling_medians[col]
    pcol = ax.fill_between(
        rolling_medians[age_col],
        rolling_low[col],
        rolling_high[col],
        # step='post', 
        color=color, alpha=alpha, label=label, edgecolor=color, linestyle=linestyle,
        **kwargs
    )
    line2d = ax.plot(rolling_medians[age_col], rolling_medians[col], 
                     color=color, linestyle=linestyle, marker='none')
    return line2d[0], pcol


def vice_running_median(ax, mzs, col, label=None, window=3000, color='k'):
    """
    Plot median stellar ages binned by abundance from VICE multi-zone output.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    mzs : MultizoneStars instance
        VICE multizone stars output.
    col : str
        Data column to group by.
    window : int, optional
        Rolling window size. The default is 1000.
    color : str, optional
        Plot color.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    """
    # Sort by ascending age
    sorted_ages = mzs.stars[mzs.stars['mass'] > 0].sort_values('age')[['age', col]]
    # Calculate rolling median
    rolling_params = dict(
        min_periods=int(window/10), step=int(window/10), on='age', center=True
    )
    rolling_medians = sorted_ages.rolling(window, **rolling_params).median()
    line2d = ax.plot(
        rolling_medians['age'], rolling_medians[col], 
        color=color, linestyle='-', label=label
    )
    return line2d


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
        '-r', '--residuals', 
        action='store_true',
        help='Plot additional residuals panels.'
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
