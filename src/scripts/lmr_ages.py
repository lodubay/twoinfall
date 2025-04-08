"""
This script compares the age distributions of locally metal-rich stars
between a two-infall VICE output and APOGEE.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
from utils import kde2D, contour_levels_2D
from scatter_plot_grid import setup_colorbar
from _globals import TWO_COLUMN_WIDTH
import paths

OUTPUT_NAME = 'yZ2/best/cgm07_ratio025_eta18_migr36/diskmodel'
AGE_COL = 'L23_AGE'
LMR_CUT = 0.1 # lower bound of locally metal-rich (LMR) stars
CONTOUR_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]
PLOT_EXTENT = [-1.2, 0.7, -0.25, 0.55]


def main(style='paper', cmap='Spectral_r'):
    # Load data and model outputs
    full_sample = APOGEESample.load()
    local_sample = full_sample.region(galr_lim=(7, 9), absz_lim=(0, 2))
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    mzs.model_uncertainty(full_sample.data, inplace=True, age_col=AGE_COL)
    mzs_local = mzs.region(galr_lim=(7, 9), absz_lim=(0, 2))

    # Set up plot
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(2, 2, 
                            figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH), 
                            sharex='row', sharey='row', 
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
    age_bins = np.arange(0, 13.1, 2)
    cbar = setup_colorbar(fig, cmap=cmap, bounds=age_bins, extend='max',
                        label='Median stellar age [Gyr]', bottom=0.52)
    # VICE
    # Problem: right now the median age isn't weighted by mass, so a number of
    # low-mass populations could skew the result in a bin
    axs[0,0].set_title(r'Two-Infall Model ($y/Z_\odot=2$)')
    weighted_median = lambda x: np.quantile(x, 0.5, weights=mzs_local('mstar'))
    mzs_local.filter({'mstar': (0.1, None)}, inplace=True)
    pcm0 = axs[0,0].hexbin(
        mzs_local('[fe/h]'), mzs_local('[o/fe]'),
        C=mzs_local('age'),
        reduce_C_function=np.median,
        gridsize=50, cmap=cmap, norm=cbar.norm, linewidths=0.2,
        extent=PLOT_EXTENT,
    )
    # Plot contour lines
    bandwidth = 0.03
    xx, yy, logz = kde2D(mzs_local.stars['[fe/h]'], mzs_local.stars['[o/fe]'], 
                        bandwidth, xbins=200j, ybins=200j)
    scaled_density = np.exp(logz) / np.max(np.exp(logz))
    levels = contour_levels_2D(scaled_density, enclosed=CONTOUR_LEVELS)
    axs[0,0].contour(xx, yy, scaled_density, levels, colors='k',
                linewidths=0.5, linestyles='-')
    axs[0,0].axvline(LMR_CUT, ls='--', c='k')
    # APOGEE
    axs[0,1].set_title('APOGEE (NN ages)')
    pcm1 = axs[0,1].hexbin(
        local_sample('FE_H'), local_sample('O_FE'),
        C=local_sample(AGE_COL),
        reduce_C_function=np.median,
        gridsize=50, cmap=cmap, norm=cbar.norm, linewidths=0.2,
        extent=PLOT_EXTENT,
    )
    local_sample.plot_kde2D_contours(axs[0,1], 'FE_H', 'O_FE',
                                    c='k', lw=0.5, ls='-', 
                                    enclosed=CONTOUR_LEVELS,
                                    bandwidth=bandwidth
                                    )
    axs[0,1].axvline(LMR_CUT, ls='--', c='k')
    axs[0,0].set_xlabel('[Fe/H]')
    axs[0,1].set_xlabel('[Fe/H]')
    axs[0,0].set_ylabel('[O/Fe]')
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0,0].set_xlim(PLOT_EXTENT[:2])
    axs[0,0].set_ylim(PLOT_EXTENT[2:])

    # Age distributions
    age_bins = np.arange(0, 14.1, 1)
    # VICE
    axs[1,0].hist(mzs_local('age'), bins=age_bins, weights=mzs_local('mstar'), 
                density=True, label='All stars',
                histtype='step', color='k', ls='-',)
    mzs_lmr = mzs_local.filter({'[fe/h]': (LMR_CUT, None)})
    axs[1,0].hist(mzs_lmr('age'), bins=age_bins, weights=mzs_lmr('mstar'), 
                density=True, 
                label=r'${\rm [Fe/H]} > %s$' % LMR_CUT,
                histtype='bar', color='gray', rwidth=0.8,)
    # APOGEE
    axs[1,1].hist(local_sample(AGE_COL), bins=age_bins, 
                density=True, label='All stars', 
                histtype='step', color='k', ls='-',)
    apogee_lmr = local_sample.filter({'FE_H': (LMR_CUT, None)})
    axs[1,1].hist(apogee_lmr(AGE_COL), bins=age_bins, 
                density=True, label=r'${\rm [Fe/H]} > %s$' % LMR_CUT,
                histtype='bar', color='gray', rwidth=0.8,)
    axs[1,1].legend()
    axs[1,0].set_xlabel('Age [Gyr]')
    axs[1,1].set_xlabel('Age [Gyr]')
    axs[1,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[1,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1,0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1,0].set_xlim((-1, 15))
    axs[1,0].set_ylabel('Density')

    fig.savefig(paths.figures / 'lmr_ages')
    plt.close()


if __name__ == '__main__':
    main()
