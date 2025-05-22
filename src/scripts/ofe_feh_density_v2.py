"""
This script plots 2D density histograms of multi-zone models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from track_and_mdf import setup_axes, plot_mdf_curve
import distribution_functions as dfs
from _globals import ONE_COLUMN_WIDTH, ZONE_WIDTH
import paths
from colormaps import paultol
from utils import get_bin_centers

OUTPUT_NAMES = [
    'yZ1/fiducial/diskmodel',
    'yZ2/fiducial/diskmodel'
]
LABELS = [
    r'(a) $y/Z_\odot=1$',
    r'(b) $y/Z_\odot=2$',
    '(c) APOGEE'
]
FEH_LIM = (-1.1, 0.6)
OFE_LIM = (-0.15, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 2)
GRIDSIZE = 30


def main(style='paper'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2.*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(22, 7, wspace=0., hspace=0.)
    subfigs = [
        fig.add_subfigure(gs[i:i+w,:]) for i, w in zip((0, 7, 14), (7, 7, 8))
    ]
    inset_axes_bounds = [1.28, 0., 0.07, 1.]
    
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    ofe_kwargs = {'bins': 100, 'range': OFE_LIM, 'smoothing': 0.05}
    feh_kwargs = {'bins': 100, 'range': FEH_LIM, 'smoothing': 0.2}
    cmap_name = 'Blues'
    for i, subfig in enumerate(subfigs[:-1]):
        axs = setup_axes(
            subfig, xlabel='[Fe/H]', xlim=FEH_LIM, ylim=OFE_LIM, 
            show_xlabel=False,
        )
        axs[0].text(
            0.5, 0.95, LABELS[i], 
            ha='center', va='top', 
            size=plt.rcParams['axes.titlesize'], transform=axs[0].transAxes
        )
        axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
        axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
        mzs = MultizoneStars.from_output(OUTPUT_NAMES[i])
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        subset = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        subset.resample_zheight(20000, apogee_data=local_sample.data)
        pcm = axs[0].hexbin(
            subset('[fe/h]'), subset('[o/fe]'),
            C=subset('mstar') / subset('mstar').sum(),
            reduce_C_function=np.sum,
            gridsize=GRIDSIZE, cmap=cmap_name, linewidths=0.1,
            extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
        )
        cax = axs[0].inset_axes(inset_axes_bounds)
        cbar = subfig.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Stellar mass fraction', labelpad=4)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(0.01))
        # Gas abundance track
        galr_mean = (GALR_LIM[1] + GALR_LIM[0]) / 2.
        zone = int(galr_mean / ZONE_WIDTH)
        multioutput = vice.output(str(subset.fullpath))
        hist = multioutput.zones[f'zone{zone}'].history
        axs[0].plot(hist['[fe/h]'], hist['[o/fe]'], color='k', marker='none', linewidth=2)
        # Mark every 1 Gyr of lookback time
        for tstart in np.arange(0.2, 15, 2):
            istart = int(100 * tstart)
            iend = int(min(100*(tstart+1), len(hist['[o/fe]'])-1))
            axs[0].plot(
                hist['[fe/h]'][istart:iend], hist['[o/fe]'][istart:iend], 
                color='w', marker='none', linewidth=1
            )
        # Marginal distributions
        color = plt.get_cmap(cmap_name)(0.6)
        feh_df, bin_edges = subset.mdf(col='[fe/h]', **feh_kwargs)
        axs[1].fill_between(
            get_bin_centers(bin_edges), feh_df / max(feh_df), y2=0, color=color,
        )
        ofe_df, bin_edges = subset.mdf(col='[o/fe]', **ofe_kwargs)
        axs[2].fill_betweenx(
            get_bin_centers(bin_edges), ofe_df / max(ofe_df), x2=0, color=color,
        )

    # APOGEE panel
    cmap_name = 'Reds'
    axs = setup_axes(
        subfigs[-1], xlabel='[Fe/H]', xlim=FEH_LIM, ylim=OFE_LIM, 
    )
    axs[0].text(
        0.5, 0.95, LABELS[-1], 
        ha='center', va='top', 
        size=plt.rcParams['axes.titlesize'], transform=axs[0].transAxes
    )
    axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    subset = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    pcm = axs[0].hexbin(
        subset('FE_H'), subset('O_FE'),
        C=np.ones(subset.nstars),
        reduce_C_function=np.sum,
        gridsize=GRIDSIZE, cmap=cmap_name, linewidths=0.2,
        extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
    )
    cax = axs[0].inset_axes(inset_axes_bounds)
    cbar = subfig.colorbar(pcm, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('Number of stars')
    # Marginal distributions
    color = plt.get_cmap(cmap_name)(0.6)
    feh_df, bin_edges = subset.mdf(col='FE_H', **feh_kwargs)
    axs[1].fill_between(
        get_bin_centers(bin_edges), feh_df / max(feh_df), y2=0, color=color
    )
    ofe_df, bin_edges = subset.mdf(col='O_FE', **ofe_kwargs)
    axs[2].fill_betweenx(
        get_bin_centers(bin_edges), ofe_df / max(ofe_df), x2=0, color=color
    )
    
    fig.subplots_adjust(top=0.98, right=0.75)

    fname = 'ofe_feh_density_v2'
    if style == 'poster':
        plt.savefig(paths.extra / 'poster' / fname)
    else:
        plt.savefig(paths.figures / fname)
    plt.close()


if __name__ == '__main__':
    main()
