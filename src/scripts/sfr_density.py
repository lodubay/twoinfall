"""
This script plots stellar abundance distributions from a mult-zone model
along with an illustration of the local SFR.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import vice

from track_and_mdf import setup_figure
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
import paths
from utils import get_bin_centers
from _globals import ONE_COLUMN_WIDTH, ZONE_WIDTH
from colormaps import paultol

GALR_LIM = (7, 9)
ABSZ_LIM = (0, 2)
FEH_LIM = (-1.4, 0.6)
OFE_LIM = (-0.14, 0.499)
GRIDSIZE = 20

def main(style='paper', cmap='Greys'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig, axs = setup_figure(xlim=FEH_LIM, ylim=OFE_LIM)

    mzs = MultizoneStars.from_output('yields/yZ2/diskmodel')
    mzs.model_uncertainty(inplace=True)
    subset = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)

    # 2D hexbin density
    pcm = axs[0].hexbin(subset('[fe/h]'), subset('[o/fe]'),
                        C=subset('mstar') / subset('mstar').sum(),
                        reduce_C_function=np.sum, #bins='log',
                        gridsize=GRIDSIZE, cmap=cmap, linewidths=0.1,
                        extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]])
    cax = axs[0].inset_axes([0.05, 0.05, 0.05, 0.8])
    fig.colorbar(pcm, cax=cax, orientation='vertical', 
                 label='Stellar mass fraction')
    
    # Marginal abundance distributions
    feh_df, bin_edges = subset.mdf(col='[fe/h]', range=FEH_LIM, smoothing=0.2)
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), 
                color='gray', linestyle='-', marker=None)
    ofe_df, bin_edges = subset.mdf(col='[o/fe]', range=OFE_LIM, smoothing=0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
                color='gray', linestyle='-', marker=None)
    
    # Gas abundance track, weighted by SFR
    galr_mean = (GALR_LIM[1] + GALR_LIM[0]) / 2.
    zone = int(galr_mean / ZONE_WIDTH)
    multioutput = vice.output(str(subset.fullpath))
    hist = multioutput.zones[f'zone{zone}'].history
    # sfr = np.array(hist['sfr'])
    # linewidths = 100 * sfr[1:]
    # points = np.array([hist['[fe/h]'], hist['[o/fe]']]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, linewidths=linewidths)
    # axs[0].add_collection(lc)

    model_color = paultol.bright.colors[0]
    axs[0].plot(hist['[fe/h]'], hist['[o/fe]'], color=model_color, marker='none')
    axs[0].scatter(hist['[fe/h]'][::10], hist['[o/fe]'][::10], 
                   s=[10*h/max(hist['sfr']) for h in hist['sfr'][::10]],
                   c=model_color)
    # Mark every Gyr
    axs[0].scatter(hist['[fe/h]'][::100], hist['[o/fe]'][::100], 
                   s=[2*h/max(hist['sfr']) for h in hist['sfr'][::100]], 
                   c='w', zorder=10)
    
    plt.savefig(paths.figures / 'sfr_density')
    plt.close()
    

if __name__ == '__main__':
    main()
