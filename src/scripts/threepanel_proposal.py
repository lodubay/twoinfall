"""
Three-panel figure for NSF proposal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
from utils import get_bin_centers
from _globals import TWO_COLUMN_WIDTH
import paths
from colormaps import paultol

OH_LIM = (-0.6, 0.8)
FEH_LIM = (-0.9, 0.6)
OFE_LIM = (-0.15, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.bright.colors)

    fig, axs = plt.subplots(1, 3, 
        figsize=(TWO_COLUMN_WIDTH, 0.3*TWO_COLUMN_WIDTH), 
        gridspec_kw={'wspace': 0.3}
    )

    # Plot APOGEE abundances + Roberts et al. (2025) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    age_bin_width = 1. # Gyr
    age_bins = np.arange(0, 13 + age_bin_width, age_bin_width)
    age_bin_centers = get_bin_centers(age_bins)
    age_col = 'CN_AGE'
    data_color = '0.6'
    data_size = 1
    mode_color = 'k'
    # [O/Fe] vs [Fe/H]
    axs[0].scatter(local_sample('FE_H'), local_sample('O_FE'), 
                    marker='.', c=data_color, s=data_size, edgecolor='none', 
                    zorder=0, rasterized=True)
    # Median abundance error
    # axs[0].errorbar(
    #     0.4, 0.5, 
    #     xerr=local_sample('FE_H_ERR').median(), 
    #     yerr=local_sample('O_FE_ERR').median(),
    #     c=data_color, marker='.', ms=0, zorder=3, 
    #     linestyle='none', elinewidth=1, capsize=0
    # )
    # Median age errors as a function of time
    big_age_bins = np.arange(0, 15, 4)
    median_age_errors = local_sample.binned_intervals(
        '%s_ERR' % age_col, age_col, big_age_bins, quantiles=[0.5]
    )
    xval_err = get_bin_centers(big_age_bins)
    yval_err = [0.7, 0.5]
    abund_range = [OH_LIM, FEH_LIM, OFE_LIM]
    for i, abund in enumerate(['O_H', 'O_FE']):
        # Scatter plot of all stars
        axs[i+1].scatter(local_sample(age_col), local_sample(abund), 
                    marker='.', c=data_color, s=data_size, edgecolor='none', 
                    zorder=0, rasterized=True)
        # 2-D histogram of APOGEE stars
        # axs[i,1].hist2d(local_sample(age_col), local_sample(abund), cmap='binary', 
        #            bins=[28, 28], range=[[0, 14], abund_range[i]], zorder=1, cmin=5)
        # Median errors at different age bins
        axs[i+1].errorbar(
            10, yval_err[i], 
            xerr=local_sample('CN_AGE_ERR').median(), 
            yerr=local_sample('%s_ERR' % abund).median(),
            c=data_color, marker='.', ms=0, zorder=3, 
            linestyle='none', elinewidth=1, capsize=0
        )
        # Plot abundance modes in bins of stellar age
        if i == 0:
            label = 'Mode'
        else:
            label = None
        abund_bins = local_sample.binned_modes(abund, age_col, age_bins)
        axs[i+1].errorbar(age_bin_centers, abund_bins['mode'], 
                    xerr=age_bin_width/2, yerr=abund_bins['error'],
                    linestyle='none', c=mode_color, capsize=1, marker='.',
                    zorder=10, label=label)
    
    # Plot smooth SFH model with y/Zsun=2
    mzs = MultizoneStars.from_output('yZ2-insideout/diskmodel')
    mzs.model_uncertainty(apogee_sample.data, inplace=True)
    mzs_local = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    mzs_inner = mzs.region(galr_lim=(3, 5), absz_lim=ABSZ_LIM)
    mzs_outer = mzs.region(galr_lim=(11, 13), absz_lim=ABSZ_LIM)
    # [O/Fe] vs [Fe/H]
    colors = [paultol.bright.colors[c] for c in [0, 2, 4]]
    for i, model in enumerate([mzs_inner, mzs_local, mzs_outer]):
        radius = int(0.5 * (model.galr_lim[0] + model.galr_lim[1]))
        plot_abundance_history(
            axs[0], model, '[o/fe]', xcol='[fe/h]', ls='--',
            c=colors[i], label='%s kpc' % radius
        )
    # Abundance histories
    plot_abundance_history(
        axs[1], mzs_local, '[o/h]', ls='--', c=colors[1]
        # label=r'$y/Z_\odot = %s$, $\eta_\odot=%s$' % (yZ[i], eta[i])
    )
    plot_abundance_history(
        axs[2], mzs_local, '[o/fe]', ls='--', c=colors[1]
    )

    # Plot two-infall SFH at y/Zsun=1, 2
    yZ = [2, 1]
    eta = [1.4, 0.2]
    output_names = [
        'yZ2-fiducial/diskmodel',
        'yZ1-fiducial/diskmodel',
    ]
    colors = [paultol.bright.colors[c] for c in [5, 1]]
    for i, output_name in enumerate(output_names):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        mzs_local = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        if 'yZ2' in output_name:
            plot_abundance_history(
                axs[0], mzs_local, '[o/fe]', xcol='[fe/h]', c=colors[i]
            )
        plot_abundance_history(
            axs[1], mzs_local, '[o/h]', c=colors[i]
        )
        plot_abundance_history(
            axs[2], mzs_local, '[o/fe]', c=colors[i],
            label=r'$y/Z_\odot = %s$' % yZ[i]
        )

    # Format axes
    axs[0].set_xlabel('[Fe/H]')
    axs[0].set_xlim(FEH_LIM)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].set_ylabel('[O/Fe]')
    axs[0].set_ylim(OFE_LIM)
    axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))

    axs[1].set_xlabel('Lookback Time [Gyr]')
    axs[1].set_xlim((-1, 14))
    axs[1].xaxis.set_major_locator(MultipleLocator(5))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].set_ylabel('[O/H]', labelpad=-3)
    axs[1].set_ylim(OH_LIM)
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))

    axs[2].set_xlabel('Lookback Time [Gyr]')
    axs[2].set_xlim((-1, 14))
    axs[2].xaxis.set_major_locator(MultipleLocator(5))
    axs[2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2].set_ylabel('[O/Fe]')
    axs[2].set_ylim(OFE_LIM)
    axs[2].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.05))

    # Legend for data
    # handles, labels = ax1.get_legend_handles_labels()
    # ax0.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
    #            loc='lower left', bbox_to_anchor=[0, 1], title='APOGEE (NN ages)')
    # # Legend for models
    # ax1.legend(handles[1:4], labels[1:4], loc='lower right', bbox_to_anchor=[1, 1])
    axs[0].legend(title='Smooth SFH')
    axs[1].legend(title='APOGEE', loc='upper left')
    axs[2].legend(title='Two-Infall')

    fig.savefig(paths.extra / 'threepanel_proposal')
    plt.close()


def plot_abundance_history(ax, mzs, col, xcol='lookback', label='', c=None, ls='-'):
    # Plot gas abundance evolution
    zone = int(0.5 * (mzs.galr_lim[0] + mzs.galr_lim[1]) / mzs.zone_width)
    zone_path = str(mzs.fullpath / ('zone%d' % zone))
    hist = vice.history(zone_path)
    ax.plot(hist[xcol], hist[col], color='w', ls='-', linewidth=2)
    ax.plot(hist[xcol], hist[col], label=label, color=c, ls=ls, linewidth=1)


if __name__ == '__main__':
    main()
