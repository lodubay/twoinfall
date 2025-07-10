"""
This script plots the evolution of gas abundance in the Solar ring for 
multi-zone models with different yields and mass-loading factors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
from utils import get_bin_centers
from _globals import ONE_COLUMN_WIDTH
import paths
from colormaps import paultol

OH_LIM = (-0.8, 0.6)
FEH_LIM = (-0.9, 0.6)
OFE_LIM = (-0.2, 0.5)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)
SMOOTH_WIDTH = 0.05

OUTPUT_NAMES = [
    'yZ3/fiducial/diskmodel',
    'yZ2/fiducial/diskmodel',
    'yZ1/fiducial/diskmodel'
]


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.bright.colors)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2.2*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 4), wspace=0., hspace=0.)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.tick_params(axis='x', labelbottom=False)
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax2)
    ax3.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax4 = fig.add_subplot(gs[2,0], sharex=ax2)
    ax5 = fig.add_subplot(gs[2,1], sharex=ax3, sharey=ax4)
    ax5.tick_params(axis='y', labelleft=False)
    axs = np.array([[ax0, ax1], [ax2, ax3], [ax4, ax5]])

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    age_bin_width = 1. # Gyr
    age_bins = np.arange(0, 13 + age_bin_width, age_bin_width)
    age_bin_centers = get_bin_centers(age_bins)
    age_col = 'L23_AGE'
    data_color = '0.6'
    mode_color = 'k'
    # Median age errors as a function of time
    big_age_bins = np.arange(0, 15, 4)
    median_age_errors = local_sample.binned_intervals(
        '%s_ERR' % age_col, age_col, big_age_bins, quantiles=[0.5]
    )
    xval_err = get_bin_centers(big_age_bins)
    yval_err = [-0.7, -0.8, -0.15]
    abund_range = [OH_LIM, FEH_LIM, OFE_LIM]
    for i, abund in enumerate(['O_H', 'FE_H', 'O_FE']):
        # Scatter plot of all stars
        axs[i,1].scatter(local_sample(age_col), local_sample(abund), 
                    marker='.', c=data_color, s=1, edgecolor='none', 
                    zorder=0, rasterized=True, label='Individual stars')
        # Median errors at different age bins
        median_abund_errors = local_sample.binned_intervals(
            '%s_ERR' % abund, age_col, big_age_bins, quantiles=[0.5]
        )
        axs[i,1].errorbar(xval_err, yval_err[i] * np.ones(xval_err.shape), 
                    xerr=median_age_errors[0.5], yerr=median_abund_errors[0.5],
                    c=data_color, marker='.', ms=0, zorder=0, linestyle='none',
                    elinewidth=0.5, capsize=0)
        # ax1.hist2d(local_sample(age_col), local_sample('O_H'), cmap='binary', 
        #            bins=[28, 28], range=[[0, 14], OH_LIM], zorder=0)
        # Plot abundance modes in bins of stellar age
        abund_bins = local_sample.binned_modes(abund, age_col, age_bins)
        axs[i,1].errorbar(age_bin_centers, abund_bins['mode'], 
                    xerr=age_bin_width/2, yerr=abund_bins['error'],
                    linestyle='none', c=mode_color, capsize=1, marker='.',
                    zorder=10, label='Binned modes')
        # Plot APOGEE abundance distributions in marginal panels
        abund_df, bin_edges = local_sample.mdf(
            col=abund, range=abund_range[i], smoothing=SMOOTH_WIDTH
        )
        axs[i,0].plot(abund_df / max(abund_df), get_bin_centers(bin_edges),
                color=data_color, linestyle='-', linewidth=2, marker=None)

    # Plot multizone gas abundance
    yZ = [3, 2, 1]
    eta = [2.4, 1.4, 0.2]
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        mzs_local = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        plot_abundance_history(
            axs[0], mzs_local, '[o/h]', range=OH_LIM, smoothing=SMOOTH_WIDTH,
            label=r'$y/Z_\odot = %s$, $\eta_\odot=%s$' % (yZ[i], eta[i])
        )
        plot_abundance_history(
            axs[1], mzs_local, '[fe/h]', range=FEH_LIM, smoothing=SMOOTH_WIDTH,
        )
        plot_abundance_history(
            axs[2], mzs_local, '[o/fe]', range=OFE_LIM, smoothing=SMOOTH_WIDTH,
        )

    # Format axes
    ax0.set_ylabel('[O/H]')
    ax0.set_xlim((1.2, 0))
    ax0.set_ylim(OH_LIM)
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.set_xlim((-1, 14))
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    
    ax2.set_ylabel('[Fe/H]')
    ax2.set_ylim(FEH_LIM)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax4.set_ylabel('[O/Fe]')
    ax4.set_xlabel('P([X/H])', size='small')
    ax4.set_ylim(OFE_LIM)
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax5.set_xlabel('Lookback Time [Gyr]')

    # Legend for data
    handles, labels = ax1.get_legend_handles_labels()
    ax0.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
               loc='lower left', bbox_to_anchor=[0, 1], title='APOGEE (NN ages)')
    # Legend for models
    ax1.legend(handles[1:4], labels[1:4], loc='lower right', bbox_to_anchor=[1, 1])

    fig.savefig(paths.figures / 'gas_abundance_evolution')
    plt.close()


def plot_abundance_history(axs, mzs, col, label='', c=None, ls='-', range=None, smoothing=0.):
    # Plot gas abundance evolution
    zone = int(0.5 * (mzs.galr_lim[0] + mzs.galr_lim[1]) / mzs.zone_width)
    zone_path = str(mzs.fullpath / ('zone%d' % zone))
    hist = vice.history(zone_path)
    axs[1].plot(hist['lookback'], hist[col], color='w', ls=ls, linewidth=2)
    axs[1].plot(hist['lookback'], hist[col], label=label, color=c, ls=ls, linewidth=1)
    # Plot MDFs
    mdf, mdf_bins = mzs.mdf(col, range=range, smoothing=smoothing, bins=100)
    axs[0].plot(mdf / mdf.max(), get_bin_centers(mdf_bins), color='w', ls=ls, linewidth=2)
    axs[0].plot(mdf / mdf.max(), get_bin_centers(mdf_bins), color=c, ls=ls, linewidth=1)


if __name__ == '__main__':
    main()
