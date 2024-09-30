"""
Plot metallicity distribution functions (MDFs) of [Fe/H] binned by radius.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import paths
from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from utils import get_color_list
import distribution_functions as dfs
from _globals import ONE_COLUMN_WIDTH, GALR_BINS, ABSZ_BINS

NBINS = 100
FEH_LIM = (-1.2, 0.7)
SMOOTH_WIDTH = 0.2

def main(output_name, uncertainties=True, **kwargs):
    apogee_sample = APOGEESample.load()
    mzs = MultizoneStars.from_output(output_name)
    if uncertainties:
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
    plot_feh_distribution(mzs, apogee_sample, **kwargs)
    
    
def plot_feh_distribution(mzs, apogee_sample, nbins=NBINS, xlim=FEH_LIM,
                          smoothing=SMOOTH_WIDTH, cmap='plasma_r', style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Set up plot
    fig, axs = dfs.setup_axes(ncols=2, figure_width=ONE_COLUMN_WIDTH, 
                              cmap=cmap, xlabel='[Fe/H]', xlim=xlim, 
                              major_tick_spacing=0.5)
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    # plot
    mdf_kwargs = {'bins': nbins, 'range': xlim, 'smoothing': smoothing}
    dfs.plot_multizone_mdfs(mzs, axs[:,0], '[fe/h]', colors, **mdf_kwargs)
    dfs.plot_multizone_mdfs(apogee_sample, axs[:,1], 'FE_H', colors, **mdf_kwargs)
    for ax in axs[:,0]:
        ax.set_ylim((0, None))
    fig.suptitle(mzs.name)
    plt.subplots_adjust(top=0.85)
    # Save
    fname = mzs.name.replace('diskmodel', 'feh_df.png')
    fullpath = paths.extra / fname
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='feh_distribution.py',
        description='Plot the [Fe/H] MDF from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('--nbins', metavar='NBINS', type=int, default=NBINS,
                        help='Number of histogram bins (default: 100)')
    parser.add_argument('--xlim', metavar='XLIM', nargs=2, type=list, 
                        default=FEH_LIM,
                        help='Lower and upper bounds of the MDF ' + \
                             '(default: [-1.1, +0.6])')
    parser.add_argument('--smoothing', metavar='WIDTH', type=float,
                        default=SMOOTH_WIDTH,
                        help='Width of boxcar smoothing (default: 0.2)')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='plasma_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: plasma_r)')
    args = parser.parse_args()
    main(**vars(args))
