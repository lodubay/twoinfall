"""
This script plots the abundance tracks from one-zone models with different
infall amplitude normalization.
"""

import math as m

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone.src.yields import J21
from multizone.src.models.gradient import gradient, thick_to_thin_ratio
from multizone.src.models.diskmodel import two_component_disk
from multizone.src.models import twoinfall_sf_law
from track_and_mdf import setup_figure, plot_vice_onezone
from utils import twoinfall_onezone
from _globals import ONEZONE_DEFAULTS, END_TIME
import paths
from colormaps import paultol

RADIUS = 8.
ZONE_WIDTH = 0.1
ONSET = 3.2

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)

    output_dir = paths.data / 'onezone' / 'ampratio'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    fig, axs = setup_figure(xlim=(-1.3, 0.4), ylim=(-0.1, 0.52))

    BHG16 = two_component_disk()

    # One-zone model settings
    eta_func = vice.milkyway.default_mass_loading
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    params = ONEZONE_DEFAULTS
    dt = params['dt']
    params['eta'] = eta_func(RADIUS)
    params['mode'] = 'ifr'
    params['tau_star'] = twoinfall_sf_law(area, onset=ONSET)
    simtime = np.arange(0, END_TIME + dt, dt)

    # Fiducial thick-to-thin ratio
    print('Fiducial model')
    print('Input T/t =', thick_to_thin_ratio(RADIUS))
    ifr = twoinfall_onezone(RADIUS, dr=ZONE_WIDTH, onset=ONSET, 
                            mass_loading=eta_func, dt=dt)
    name = str(output_dir / 'fiducial')
    sz = vice.singlezone(name=name, func=ifr, **params)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, fig=fig, axs=axs, label='Fiducial')
    # Calculate T/t, Mstar
    hist = vice.history(name)
    onset_idx = int(ifr.onset / dt)
    thick_disk_mass = hist['mstar'][onset_idx-1]
    thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
    print('Final T/t =', thick_disk_mass / thin_disk_mass)
    print('Final Mstar =', hist['mstar'][-1])

    # Re-compute with higher thick-to-thin ratio
    print('\nMore thick model')
    high_tt_ratio = lambda r: 2.0 * thick_to_thin_ratio(r)
    print('Input T/t =', high_tt_ratio(RADIUS))
    ifr = twoinfall_ampratio(RADIUS, high_tt_ratio, 
                             dr=ZONE_WIDTH, onset=ONSET, 
                             mass_loading=eta_func, dt=dt)
    name = str(output_dir / 'more_thick')
    sz = vice.singlezone(name=name, func=ifr, **params)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, fig=fig, axs=axs, label='2x')
    # Calculate T/t, Mstar
    hist = vice.history(name)
    onset_idx = int(ifr.onset / dt)
    thick_disk_mass = hist['mstar'][onset_idx-1]
    thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
    print('Final T/t =', thick_disk_mass / thin_disk_mass)
    print('Final Mstar =', hist['mstar'][-1])

    # Re-compute with lower thick-to-thin ratio
    print('\nLess thick model')
    low_tt_ratio = lambda r: 0.5 * thick_to_thin_ratio(r)
    print('Input T/t =', low_tt_ratio(RADIUS))
    ifr = twoinfall_ampratio(RADIUS, low_tt_ratio, 
                             dr=ZONE_WIDTH, onset=ONSET, 
                             mass_loading=eta_func, dt=dt)
    name = str(output_dir / 'less_thick')
    sz = vice.singlezone(name=name, func=ifr, **params)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, fig=fig, axs=axs, label='0.5x')
    # Calculate T/t, Mstar
    hist = vice.history(name)
    onset_idx = int(ifr.onset / dt)
    thick_disk_mass = hist['mstar'][onset_idx-1]
    thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
    print('Final T/t =', thick_disk_mass / thin_disk_mass)
    print('Final Mstar =', hist['mstar'][-1])

    axs[0].legend()
    fig.savefig(paths.figures / 'onezone_ampratio')
    plt.close()


class twoinfall_ampratio(twoinfall_onezone):
    def __init__(self, radius, thick_thin_function, 
                 mass_loading=vice.milkyway.default_mass_loading,
                 dt = 0.01, dr = 0.1, **kwargs):
        super().__init__(radius, **kwargs)
        # Re-compute normalization with different thick-to-thin ratio
        self.ratio = self.ampratio(radius, thick_thin_function, 
                                   mass_loading=mass_loading, 
                                   dt=dt, dr=dr)
        prefactor = self.normalize(radius, gradient, 
                                   mass_loading = mass_loading, 
                                   dt = dt,  dr = dr)
        area = m.pi * ((radius + dr/2)**2 - (radius - dr/2)**2)
        self.first.norm *= prefactor * area * gradient(radius)
        self.second.norm *= prefactor * area * gradient(radius)


if __name__ == '__main__':
    main()
