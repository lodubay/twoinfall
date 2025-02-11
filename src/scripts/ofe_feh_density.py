"""
This script plots 2D density histograms of multi-zone models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from _globals import ONE_COLUMN_WIDTH
import paths

OUTPUT_NAMES = [
    'yZ1/mass_loading/bespoke/diskmodel',
    'yZ2/mass_loading/bespoke/diskmodel'
]
LABELS = [
    r'(a) $y/Z_\odot=1$',
    r'(b) $y/Z_\odot=2$',
    '(c) APOGEE'
]
FEH_LIM = (-1.1, 0.6)
OFE_LIM = (-0.1, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0.5, 1)
GRIDSIZE = 50


def main(style='paper', cmap='Blues'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(len(OUTPUT_NAMES)+1, 1,
                            figsize=(0.67 * ONE_COLUMN_WIDTH, 2 * ONE_COLUMN_WIDTH),
                            sharex=True, sharey=True)
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.08, top=0.95,
                        wspace=0., hspace=0.)
    
    apogee_sample = APOGEESample.load()
    for i, ax in enumerate(axs[:-1]):
        mzs = MultizoneStars.from_output(OUTPUT_NAMES[i])
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        ax.set_title(LABELS[i], y=0.85)
        subset = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        pcm = ax.hexbin(
            subset('[fe/h]'), subset('[o/fe]'),
            # C=subset('galr_origin'), reduce_C_function=np.median,
            # C=subset('age'), reduce_C_function=np.median,
            C=subset('mstar') / 1e6,
            reduce_C_function=np.sum, #bins='log',
            gridsize=GRIDSIZE, cmap=cmap, linewidths=0.1,
            extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]]
        )
        cax = axs[i].inset_axes([1.05, 0.05, 0.05, 0.9])
        fig.colorbar(pcm, cax=cax, orientation='vertical',
                     label=r'Stellar mass [$\times10^6$ M$_\odot$]')

    axs[-1].set_title(LABELS[-1], y=0.85)
    subset = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    pcm = axs[-1].hexbin(
        subset('FE_H'), subset('O_FE'),
        #    C=subset('L23_AGE'), reduce_C_function=np.median,
        C=np.ones(subset.nstars),
        reduce_C_function=np.sum, #bins='log',
        gridsize=GRIDSIZE, cmap=cmap, linewidths=0.2,
        extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]]
    )
    cax = axs[-1].inset_axes([1.05, 0.05, 0.05, 0.9])
    fig.colorbar(pcm, cax=cax, orientation='vertical',
                 label='Number of stars')
    
    # Axes limits
    axs[0].set_xlim(FEH_LIM)
    axs[0].set_ylim(OFE_LIM)
    # Axis labels
    axs[-1].set_xlabel('[Fe/H]', labelpad=2)
    for i, ax in enumerate(axs):
        ax.set_ylabel('[O/Fe]', labelpad=2)
    # Set x-axis ticks
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.savefig(paths.figures / 'ofe_feh_density')
    plt.close()


if __name__ == '__main__':
    main()
