"""
This script compares [Fe/H] distributions across the disk between models
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
    'sfe_tests/tmol/diskmodel',
    'sfe_tests/tstep/diskmodel',
    'sfe_tests/fiducial/diskmodel'
]
LABELS = [
    r'$\propto\tau_{\rm mol}$',
    'Step-function',
    'Both',
    'APOGEE'
]
NBINS = 100
FEH_LIM = (-1.2, 0.7)
SMOOTH_WIDTH = 0.2


def main(style='paper', cmap='plasma_r'):
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = dfs.setup_axes(
        ncols=len(OUTPUT_NAMES)+1, 
        figure_width=TWO_COLUMN_WIDTH,
        cmap=cmap, 
        xlabel='[Fe/H]', 
        xlim=FEH_LIM,
        major_tick_spacing=0.5, 
        cbar_width=0.4
    )
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    apogee_sample = APOGEESample.load()
    apogee_index = len(OUTPUT_NAMES)
    mdf_kwargs = {'bins': NBINS, 'range': FEH_LIM, 'smoothing': SMOOTH_WIDTH}
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        dfs.plot_multizone_mdfs(mzs, axs[:,i], '[fe/h]', colors, label=LABELS[i],
                                **mdf_kwargs)
    dfs.plot_multizone_mdfs(apogee_sample, axs[:,apogee_index], 'FE_H', colors, 
                            label=LABELS[apogee_index], **mdf_kwargs)
    for ax in axs[:,0]:
        ax.set_ylim((0, None))
    highlight_panels(
        fig, axs, [(0,apogee_index), (1, apogee_index), (2, apogee_index)]
    )
    # Save
    fullpath = paths.extra / 'sfe_tests' / 'feh_distribution.png'
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath)
    plt.close()


if __name__ == '__main__':
    main()
