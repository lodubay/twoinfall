"""
This script plots the outputs of one-zone models which illustrate the effect
of a variable star formation efficiency (SFE).
"""

from track_and_mdf import setup_axes, plot_vice_onezone
from colormaps import paultol
from _globals import END_TIME, ONEZONE_DEFAULTS, ONE_COLUMN_WIDTH, ZONE_WIDTH
from multizone.src import models, dtds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vice

from multizone.src.models import twoinfall_sf_law
import paths
from multizone.src.yields import W23
vice.yields.sneia.settings['fe'] *= (1.1/1.2)

RADIUS = 8.
MASS_LOADING = 0.21
XLIM = (-1.7, 0.7)
YLIM = (-0.14, 0.54)


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.vibrant.colors)

    output_dir = paths.data / 'onezone' / 'sfe'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH))
    axs = setup_axes(fig, title='')

    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)

    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    dtd = dtds.plateau(width=1, tmin=ONEZONE_DEFAULTS['delay'])

    tau_star_list = [twoinfall_sf_law(area, onset=4.),
                     twoinfall_sf_law(area, onset=4., factor=1.)]
    names = ['factor10', 'factor05']
    labels = ['Default', 'Variable']
    for i, tau_star in enumerate(tau_star_list):
        # Run one-zone model
        name = str(output_dir / names[i])
        ifr = models.twoinfall(RADIUS, dt=dt, first_timescale=1.,
                               second_timescale=10., onset=4.)
        sz = vice.singlezone(name=name,
                             RIa=dtd,
                             func=ifr,
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = tau_star
        sz.eta = MASS_LOADING
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name,
                          fig=fig, axs=axs,
                          linestyle='-',
                          color=None,
                          label=labels[i],
                          marker_labels=(i == 0),
                          markers=[0.1, 0.3, 1, 3, 10])

    # Adjust axis limits
    axs[1].set_ylim(bottom=0)
    axs[2].set_xlim(left=0)
    axs[0].legend(frameon=False, loc='lower left',
                  title='$%s$ [Gyr]' % LABELS[var])

    # Label other param values
    axs[0].text(0.95, 0.95, other_params, ha='right', va='top',
                transform=axs[0].transAxes)
    subfigs = [fig.add_subfigure(gs[:, i:i+w])
               for i, w in zip((0, 8, 15), (8, 7, 7))]
    # Outflow mass-loading factor
    eta = mass_loading_factor(tau_sfh=10., tau_star=2.)
    # eta = 1.
    print('Mass-loading factor: ', eta)
    print('First timescale')
    axs0 = vary_param(subfigs[0], first_timescale=[0.1, 0.3, 1, 3],
                      second_timescale=10, onset=4, eta=eta,
                      xlim=XLIM, ylim=YLIM, label_index=3)
    print('Second timescale')
    axs1 = vary_param(subfigs[1], second_timescale=[1, 3, 10, 30],
                      first_timescale=1, onset=4, eta=eta,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=1)
    print('Onset time')
    axs2 = vary_param(subfigs[2], onset=[1, 2, 3, 4, 5],
                      first_timescale=1, second_timescale=10, eta=eta,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=2)
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16,
                        right=0.98, wspace=0.5)
    fig.savefig(paths.figures / 'onezone_params', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
