"""
This script plots the outputs of one-zone models which illustrate the effect
of a variable star formation efficiency (SFE).
"""

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone.src import models, dtds
from multizone.src.models import twoinfall_sf_law
from multizone.src.models import equilibrium_mass_loading
from multizone.src.yields import W23
# vice.yields.sneia.settings['fe'] *= (1.1/1.2)
from track_and_mdf import setup_figure, plot_vice_onezone
from colormaps import paultol
import paths
from _globals import END_TIME, ONEZONE_DEFAULTS, ONE_COLUMN_WIDTH, ZONE_WIDTH

RADIUS = 8.
XLIM = (-1.6, 0.6)
YLIM = (-0.16, 0.48)


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.vibrant.colors)

    output_dir = paths.data / 'onezone' / 'sfe'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fig, axs = setup_figure(width=ONE_COLUMN_WIDTH, xlim=XLIM, ylim=YLIM)

    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)

    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    ifr = models.twoinfall(
        RADIUS, 
        first_timescale=1.,
        second_timescale=15., 
        onset=3.5
    )
    eta_func = equilibrium_mass_loading(
        tau_star=2., 
        tau_sfh=ifr.second.timescale, 
        alpha_h_eq=0.2
    )

    tau_star_list = [twoinfall_sf_law(area, onset=ifr.onset, factor=1.),
                     twoinfall_sf_law(area, onset=ifr.onset, factor=0.5),
                     twoinfall_sf_law(area, onset=ifr.onset, factor=0.2)]
    names = ['factor10', 'factor05', 'factor02']
    labels = ['Fiducial', 'Variable (2x)', 'Variable (5x)']
    for i, tau_star in enumerate(tau_star_list):
        # Run one-zone model
        name = str(output_dir / names[i])
        sz = vice.singlezone(name=name,
                             func=ifr,
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = tau_star
        sz.eta = eta_func(RADIUS)
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name,
                          fig=fig, axs=axs,
                          linestyle='-',
                          color=None,
                          label=labels[i],
                          marker_labels=(i == 2),
                          markers=[0.3, 1, 3, 5, 10])
        # Thick-to-thin ratio
        hist = vice.history(name)
        onset_idx = int(ifr.onset / dt)
        thick_disk_mass = hist['mstar'][onset_idx-1]
        thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
        print(thick_disk_mass / thin_disk_mass)

    # Adjust axis limits
    axs[1].set_ylim(bottom=0)
    axs[2].set_xlim(left=0)
    axs[0].legend(frameon=False, loc='lower left', title='SFE')
    
    fig.savefig(paths.figures / 'onezone_sfe', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
