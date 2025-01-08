"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

import matplotlib.pyplot as plt

from multizone.src.yields import J21
from onezone_params import vary_param
from _globals import TWO_COLUMN_WIDTH
from colormaps import paultol
import paths

XLIM = (-1.4, 0.8)
YLIM = (-0.18, 0.54)

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(7, 22, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))]
    print('\nFirst timescale')
    axs0 = vary_param(subfigs[0], first_timescale=[0.1, 0.3, 1, 3],
                      second_timescale=10, onset=3,
                      xlim=XLIM, ylim=YLIM, label_index=3, verbose=True)
    print('\nSecond timescale')
    axs1 = vary_param(subfigs[1], second_timescale=[3, 5, 10, 30],
                      first_timescale=1, onset=3,
                      xlim=XLIM, ylim=YLIM, show_ylabel=False,
                      label_index=0, verbose=True)
    print('\nOnset time')
    axs2 = vary_param(subfigs[2], onset=[1, 2, 3, 4, 5],
                      first_timescale=1, second_timescale=10,
                      xlim=XLIM, ylim=YLIM, show_ylabel=False,
                      label_index=0, verbose=True)
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5)
    fig.savefig(paths.figures / 'onezone_params_high_yields', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
