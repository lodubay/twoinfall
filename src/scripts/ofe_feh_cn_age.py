"""
This script plots [O/Fe] vs [Fe/H] density histograms in bins of Rgal
and age, using the [C/N]-derived ages for APOGEE stars.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from apogee_sample import APOGEESample
from scatter_plot_grid import setup_colorbar
from _globals import TWO_COLUMN_WIDTH
import paths

GALR_BINS = [3, 5, 7, 9, 11, 13]
AGE_BINS = [1, 2, 4, 6, 8, 10]
# AGE_BINS = [1, 3, 5, 7, 9, 13]
# AGE_BINS = [0, 2, 4, 6, 8, 10, 13]
FEH_LIM = (-0.9, 0.5)
OFE_LIM = (-0.15, 0.55)
ABSZ_LIM = (0, 2)
GRIDSIZE = 20
AGE_COL = 'CN_AGE'
# VMAX = 0.11


def main(style='paper', cmap='binary'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        len(AGE_BINS)-1, len(GALR_BINS)-1,
        figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # cbar = setup_colorbar(fig, cmap=cmap, vmin=0, vmax=VMAX, 
    #                       label='Density',
    #                       width=0.015, pad=0.015)

    sample = APOGEESample.load()
    for j in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[j:j+2]
        axs[0,j].set_title(r'$%s \leq R_{\rm gal} < %s$ kpc' % tuple(galr_lim))
        region_subset = sample.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
        for i in range(len(AGE_BINS)-1):
            age_lim = (AGE_BINS[-(i+2)], AGE_BINS[-(i+1)])
            age_subset = region_subset.filter({AGE_COL: tuple(age_lim)})
            pcm = axs[i,j].hexbin(
                age_subset('FE_H'), age_subset('O_FE'),
                C=1 / age_subset.nstars * np.ones(age_subset.nstars),
                reduce_C_function=np.sum,
                # c=age_subset('ABSZ'), reduce_C_function=np.median,
                gridsize=GRIDSIZE, cmap=cmap, linewidths=0.2, 
                # vmin=0, vmax=VMAX,
                extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
                marginals=True
            )
            # Plot reference contours of entire sample
            region_subset.plot_kde2D_contours(axs[i,j], 'FE_H', 'O_FE')
            # print(pcm.get_clim())
            if j == 0:
                axs[i,j].text(
                    0.05, 0.95, 
                    r'$%s \leq \tau_{\rm [C/N]} < %s$ Gyr' % tuple(age_lim),
                    va='top', transform=axs[i,j].transAxes,
                    bbox={
                        'facecolor': 'w',
                        'edgecolor': 'none',
                        'boxstyle': 'round',
                        'pad': 0.15,
                        'alpha': 1.,
                    }
                )
            if AGE_COL == 'CN_AGE':
                # Indicate cut below [Fe/H] < -0.4 for URGB and RC stars
                axs[i,j].axvline(-0.4, color='gray', ls='--')
    
    for ax in axs[-1]:
        ax.set_xlabel('[Fe/H]')
    for ax in axs[:,0]:
        ax.set_ylabel('[O/Fe]')
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0,0].set_xlim(FEH_LIM)
    axs[0,0].set_ylim(OFE_LIM)
    
    fig.savefig(paths.figures / 'ofe_feh_cn_age')
    plt.close()


if __name__ == '__main__':
    main()
