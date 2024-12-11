"""
This script plots the MDF across the disk comparing models with and without
stellar migration.
"""

import numpy as np
import matplotlib.pyplot as plt
from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
import paths
import _globals
from utils import get_bin_centers

def main(style='paper', smooth_width=0.1, xlim=(-1.7, 0.7), nbins=100):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,
                            figsize=(_globals.TWO_COLUMN_WIDTH, 
                                     2/3 * _globals.TWO_COLUMN_WIDTH))
    # fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.95)
    
    apogee_data = APOGEESample.load()
    mzs_mig = MultizoneStars.from_output('gaussian/outflow/no_gasflow/pristine/J21/insideout/diskmodel')
    mzs_nomig = MultizoneStars.from_output('nomigration/outflow/no_gasflow/pristine/J21/insideout/diskmodel')
    # mzs_post = MultizoneStars.from_output('post-process/outflow/no_gasflow/J21/twoinfall/diskmodel')
    mzs_fast = MultizoneStars.from_output('gaussian_fast/outflow/no_gasflow/pristine/J21/insideout/diskmodel')
    mzs_f18 = MultizoneStars.from_output('frankel2018/outflow/no_gasflow/pristine/J21/insideout/diskmodel')
    
    for i, ax in enumerate(axs.flatten()):
        galr_lim = (_globals.GALR_BINS[i], _globals.GALR_BINS[i+1])
        # ax.text(0.5, 0.95, r'$%s \leq R_{\rm gal} < %s$ kpc' % galr_lim, 
        #         transform=ax.transAxes, ha='center', va='top')
        ax.set_title(r'$%s \leq R_{\rm gal} < %s$ kpc' % galr_lim)
        nomig_subset = mzs_nomig.region(galr_lim, absz_lim=(0, 2))
        nomig_mdf, mdf_bins = nomig_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                               smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), nomig_mdf, c='k', ls='--',
                label='No migration')
        mig_subset = mzs_mig.region(galr_lim, absz_lim=(0, 2))
        mig_mdf, mdf_bins = mig_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                           smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), mig_mdf, c='b', ls='-',
                label=r'Gaussian ($\tau^{0.33}$)')
        fast_subset = mzs_fast.region(galr_lim, absz_lim=(0, 2))
        fast_mdf, mdf_bins = fast_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                               smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), fast_mdf, c='b', ls='--',
                label=r'Gaussian ($\tau^{0.5}$)')
        f18_subset = mzs_f18.region(galr_lim, absz_lim=(0, 2))
        f18_mdf, mdf_bins = f18_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                               smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), f18_mdf, c='g', ls=':',
                label='Frankel et al. (2018)')
        # Plot data
        # apogee_subset = apogee_data.region(galr_lim=galr_lim, absz_lim=(0, 2))
        # data_mdf, mdf_bins = apogee_subset.mdf(col='FE_H', bins=nbins,
        #                                        range=xlim, smoothing=smooth_width)
        # ax.plot(get_bin_centers(mdf_bins), data_mdf, c='r', ls='--',
        #         label='APOGEE DR17')
    
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim((0, 4))
    for ax in axs[-1]:
        ax.set_xlabel('[Fe/H]')
    for ax in axs[:,0]:
        ax.set_ylabel('PDF')
    
    axs[0,-1].legend(frameon=False)
    fig.savefig(paths.figures / 'migration_comparison')


if __name__ == '__main__':
    main()
