"""
This script plots the MDF across the disk comparing models with and without
stellar migration.
"""

import numpy as np
import matplotlib.pyplot as plt
from multizone_stars import MultizoneStars
from apogee_tools import import_apogee, apogee_region, apogee_mdf
import paths
import _globals
from utils import get_bin_centers

def main(style='paper', smooth_width=0.2, xlim=(-1.7, 0.7), nbins=100):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,
                            figsize=(_globals.TWO_COLUMN_WIDTH, 
                                     2/3 * _globals.TWO_COLUMN_WIDTH))
    # fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.95)
    
    apogee_data = import_apogee()
    mzs_mig = MultizoneStars.from_output('gaussian/twoinfall/plateau_width10/diskmodel')
    mzs_nomig = MultizoneStars.from_output('nomigration/twoinfall/plateau_width10/diskmodel')
    
    for i, ax in enumerate(axs.flatten()):
        galr_lim = (_globals.GALR_BINS[i], _globals.GALR_BINS[i+1])
        # ax.text(0.5, 0.95, r'$%s \leq R_{\rm gal} < %s$ kpc' % galr_lim, 
        #         transform=ax.transAxes, ha='center', va='top')
        ax.set_title(r'$%s \leq R_{\rm gal} < %s$ kpc' % galr_lim)
        mig_subset = mzs_mig.region(galr_lim, absz_lim=(0, 2))
        mig_mdf, mdf_bins = mig_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                           smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), mig_mdf, c='k', ls='-',
                label='With migration')
        nomig_subset = mzs_nomig.region(galr_lim, absz_lim=(0, 2))
        nomig_mdf, mdf_bins = nomig_subset.mdf('[fe/h]', bins=nbins, range=xlim,
                                               smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), nomig_mdf, c='k', ls='--',
                label='Without migration')
        apogee_subset = apogee_region(apogee_data, galr_lim=galr_lim, absz_lim=(0, 2))
        data_mdf, mdf_bins = apogee_mdf(apogee_subset, col='FE_H', bins=nbins,
                                        range=xlim, smoothing=smooth_width)
        ax.plot(get_bin_centers(mdf_bins), data_mdf, c='r', ls='-',
                label='APOGEE DR17')
    
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim((0, 2.5))
    for ax in axs[-1]:
        ax.set_xlabel('[Fe/H]')
    for ax in axs[:,0]:
        ax.set_ylabel('PDF')
    
    axs[0,-1].legend(frameon=False)
    fig.savefig(paths.figures / 'migration_comparison')


if __name__ == '__main__':
    main()
