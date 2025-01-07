"""
This script plots abundance diagrams from one-zone models with different
SN Ia delay time distributions (DTDs).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import vice

from multizone.src.yields import J21
from multizone.src import dtds
from multizone.src.models import twoinfall_sf_law
from track_and_mdf import setup_axes, plot_vice_onezone
from utils import twoinfall_onezone
from colormaps import paultol
from _globals import ONEZONE_DEFAULTS, END_TIME, DT, TWO_COLUMN_WIDTH
import paths

RADIUS = 8.
ZONE_WIDTH = 2.
XLIM = (-1.3, 0.4)
YLIM = (-0.14, 0.499)
FIRST_INFALL = 1.
SECOND_INFALL = 15.
ONSET = 3.2


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.bright.colors)
    
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.5*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(5, 10, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,:5]),
               fig.add_subfigure(gs[:,5:])]

    dtd_list = [dtds.plateau(), dtds.exponential(), dtds.powerlaw()]
    plot_dtds(subfigs[0], dtd_list, labels = ['Plateau', 'Exponential', 'Power-law'])
    plot_onezone(subfigs[1], dtd_list, labels = ['Plateau', 'Exponential', 'Power-law'])
    
    fig.savefig(paths.figures / 'delay_time_distribution')
    plt.close()


def plot_dtds(fig, dtd_list, labels=[]):
    """Plot DTD models as a function of time."""
    missing_labels = len(dtd_list) - len(labels)
    if missing_labels:
        labels += [None] * missing_labels
    ax = fig.add_subplot()
    fig.subplots_adjust(hspace=0., wspace=0.)
    dt = 0.001
    tarr = np.arange(0.04, 13.2, dt)
    linestyles = ['-', '-.', '--']
    for i, dtd in enumerate(dtd_list):
        yvals = np.array([dtd(t) for t in tarr])
        ax.plot(tarr, yvals / np.sum(yvals * dt), label=labels[i],
                ls=linestyles[i])
        # indicate median delay times
        cdf = np.cumsum(yvals / np.sum(yvals))
        med_idx = np.where(cdf >= 0.5)[0][0]
        med = tarr[med_idx]
        ax.scatter(med, 2e-3, s=10, marker='o')
    # Label median delay times
    ax.text(1, 3e-3, 'Median delay times', ha='center')
    # Format axes
    ax.set_ylim((1e-3, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    log_formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
    ax.xaxis.set_major_formatter(log_formatter)
    ax.yaxis.set_major_formatter(log_formatter)
    ax.set_xlabel('Time after star formation [Gyr]')
    ax.set_ylabel('Normalized SN Ia rate')
    if not missing_labels:
        ax.legend(frameon=False, title='SN Ia DTD', loc='upper right')
    return ax


def plot_onezone(fig, dtd_list, labels=[]):
    """Plot abundance tracks of one-zone models with different DTDs."""
    missing_labels = len(dtd_list) - len(labels)
    if missing_labels:
        labels += [None] * missing_labels
    output_dir = paths.data / 'onezone' / 'dtd'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    axs = setup_axes(fig, xlim=XLIM, ylim=YLIM)
    fig.subplots_adjust(left=0.05, hspace=0., wspace=0.)
    linestyles = ['-', '-.', '--']
    for i, dtd in enumerate(dtd_list):
        run_model(dtd)
        plot_vice_onezone(str(output_dir / dtd.name), 
                          fig=fig, axs=axs, label=labels[i],
                          marker_labels=(i == 0),
                          markers=[0.3, 1, 3, 10],
                          ls=linestyles[i])
    if not missing_labels:
        axs[0].legend(frameon=False, title='SN Ia DTD', loc='lower left')
    return axs


def run_model(dtd, output_dir=paths.data / 'onezone' / 'dtd'):
    ifr = twoinfall_onezone(
        RADIUS, 
        first_timescale=FIRST_INFALL,
        second_timescale=SECOND_INFALL, 
        onset=ONSET,
        dr=ZONE_WIDTH
    )
    # Run one-zone model
    fullname = str(output_dir / dtd.name)
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    sz = vice.singlezone(name=fullname,
                         func=ifr,
                         mode='ifr',
                         **ONEZONE_DEFAULTS)
    sz.tau_star = twoinfall_sf_law(area, onset=ONSET)
    sz.RIa = dtd
    simtime = np.arange(0, END_TIME + DT, DT)
    sz.run(simtime, overwrite=True)


if __name__ == '__main__':
    main()
