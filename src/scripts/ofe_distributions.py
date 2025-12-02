"""
This script compares [O/Fe] distributions across the disk between models
with different parameters, plus APOGEE data.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import distribution_functions as dfs
from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from utils import get_color_list, highlight_panels
import paths
from _globals import TWO_COLUMN_WIDTH, GALR_BINS, ABSZ_BINS

OUTPUT_NAMES = [
    'yZ2-earlyonset/diskmodel',
    'yZ2-powerlaw/diskmodel',
    'yZ2-diskratio/diskmodel',
    'yZ2-preenrich/diskmodel',
]
LABELS = [
    '(a)\nFiducial',
    '(b)\nPower-law DTD',
    '(c)\n' + r'$f_\Sigma(R_\odot)=0.5$',
    '(d)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.5$',
    'APOGEE\n(e)'
]
NBINS = 100
OFE_LIM = (-0.15, 0.55)
SMOOTH_WIDTH = 0.05
CMAP = 'viridis_r'


def main(style='paper', cmap=CMAP, verbose=False):
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = dfs.setup_axes(
        ncols=len(OUTPUT_NAMES)+1, 
        figure_width=TWO_COLUMN_WIDTH,
        cmap=cmap, 
        xlabel='[O/Fe]', 
        xlim=OFE_LIM,
        major_tick_spacing=0.2, 
        major_minor_tick_ratio=4.,
        cbar_width=0.4
    )
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    apogee_sample = APOGEESample.load()
    apogee_index = len(OUTPUT_NAMES)
    mdf_kwargs = {'bins': NBINS, 'range': OFE_LIM, 'smoothing': SMOOTH_WIDTH}
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        dfs.plot_multizone_mdfs(mzs, axs[:,i], '[o/fe]', colors, label=LABELS[i],
                                **mdf_kwargs)
    dfs.plot_multizone_mdfs(apogee_sample, axs[:,apogee_index], 'O_FE', colors, 
                            label=LABELS[apogee_index], titlepad=18, **mdf_kwargs)
    for ax in axs[:,0]:
        ax.set_ylim((0, None))
    highlight_panels(
        fig, axs, [(0,apogee_index), (1, apogee_index), (2, apogee_index)]
    )
    # Add figure title
    fig.suptitle(r'$y/Z_\odot=2$', x=0.44, y=1.06)
    fig.subplots_adjust(top=0.92)
    # Save
    plt.savefig(paths.figures / 'ofe_distributions')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ofe_distributions.py',
        description='Compare stellar [O/Fe] distributions across the disk \
between multi-zone models with different parameters and APOGEE data.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '--style', 
        metavar='STYLE', 
        type=str,
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default=CMAP,
        help='Name of colormap for color-coding model output (default: %s).' % CMAP
    )
    args = parser.parse_args()
    main(**vars(args))
