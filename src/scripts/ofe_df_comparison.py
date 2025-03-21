"""
This script compares [O/Fe] distributions across the disk between models
with low and high yields, plus APOGEE data.
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
    'yZ1/fiducial/diskmodel',
    'yZ2/fiducial/diskmodel',
    'yZ2/dtd/powerlaw/diskmodel',
    'yZ2/pre_enrichment/mh07_alpha00/diskmodel'
]
LABELS = [
    '(a)\n' + r'$y/Z_\odot=1$',
    '(b)\n' + r'$y/Z_\odot=2$',
    '(c)\n' + r'$y/Z_\odot=2$ &' + '\nPower-law DTD',
    '(d)\n' + r'$y/Z_\odot=2$ &' + '\n' + r'${\rm [X/H]}_{\rm CGM}=-0.7$',
    '(e)\nAPOGEE'
]
LABEL_PADS = [2, 2, -8, -9, 3]
NBINS = 100
OFE_LIM = (-0.15, 0.55)
SMOOTH_WIDTH = 0.05


def main(style='paper', cmap='plasma_r'):
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = dfs.setup_axes(ncols=5, figure_width=TWO_COLUMN_WIDTH,
                              cmap=cmap, xlabel='[O/Fe]', xlim=OFE_LIM,
                              major_tick_spacing=0.2, major_minor_tick_ratio=4.,
                              cbar_width=0.4)
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    apogee_sample = APOGEESample.load()
    apogee_index = len(OUTPUT_NAMES)
    mdf_kwargs = {'bins': NBINS, 'range': OFE_LIM, 'smoothing': SMOOTH_WIDTH}
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        dfs.plot_multizone_mdfs(mzs, axs[:,i], '[o/fe]', colors, label=LABELS[i],
                                titlepad=LABEL_PADS[i], **mdf_kwargs)
    dfs.plot_multizone_mdfs(apogee_sample, axs[:,apogee_index], 'O_FE', colors, 
                            label=LABELS[apogee_index], **mdf_kwargs)
    for ax in axs[:,0]:
        ax.set_ylim((0, None))
    highlight_panels(fig, axs, 
                     [(0,apogee_index), (1, apogee_index), (2, apogee_index)])
    # Save
    plt.savefig(paths.figures / 'ofe_df_comparison')
    plt.close()


if __name__ == '__main__':
    main()
