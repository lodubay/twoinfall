"""
This script compares the abundance evolution of multi-zone models with 
different yield sets and outflow settings.
"""

import argparse
import matplotlib.pyplot as plt
from abundance_evolution_yields import plot_abundance_evolution_comparison
import paths

MODEL_LIST = ['pristine', 'mh07_alpha00']
LABEL_LIST = ['Pristine', r'${\rm [X/H]}_{\rm CGM}=-0.7$']


def main(verbose=False, style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plot_abundance_evolution_comparison(
        [f'pre_enrichment/{model}/diskmodel' for model in MODEL_LIST],
        labels=LABEL_LIST,
        uncertainties=True, verbose=verbose, cmap_name=cmap
    )
    fig.savefig(paths.figures / 'abundance_evolution_pre_enrichment')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='abundance_evolution_yields.py',
        description='Compare age-abundance relations between multi-zone \
outputs with different yield sets and APOGEE data.'
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
        default='winter_r',
        help='Name of colormap for color-coding model output (default: winter_r).'
    )
    args = parser.parse_args()
    main(**vars(args))
