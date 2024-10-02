"""
This script plots one-zone model results with pre-enriched infall.
"""

import math as m

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone.src.yields import W23
from multizone.src.models import twoinfall, twoinfall_sf_law, equilibrium_mass_loading
from track_and_mdf import setup_axes, plot_vice_onezone
from apogee_sample import APOGEESample
import paths
from utils import get_bin_centers
from _globals import ONEZONE_DEFAULTS, ZONE_WIDTH, ONE_COLUMN_WIDTH, END_TIME
from colormaps import paultol

XLIM = (-2.2, 0.4)
YLIM = (-0.1, 0.54)
RADIUS = 8. # kpc


def main():
    # Set up figure: two panels, vertically stacked
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 1.85*ONE_COLUMN_WIDTH))
    # TODO figure out how to make subfigures exactly equal
    gs = fig.add_gridspec(13, 7, hspace=0., wspace=0.)
    subfigs = [fig.add_subfigure(gs[i:i+w,:]) for i, w in zip((0, 6), (6, 7))]
    
    # One-zone model settings
    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    eta_func = equilibrium_mass_loading()
    eta = eta_func(RADIUS)
    ifr = twoinfall(RADIUS, mass_loading=eta_func, dt=dt, 
                    dr=ZONE_WIDTH)
    tau_star = twoinfall_sf_law(area, onset=ifr.onset)
    parent_dir = paths.onezone / 'enriched_infall'
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
    
    # First panel: constant [O/Fe]
    axs0 = setup_axes(subfigs[0], xlim=XLIM, ylim=YLIM, xlabel=False)
    # Pristine infall
    name = str(parent_dir / 'ZinNone')
    sz = vice.singlezone(name=name,
                         func=ifr, 
                         mode='ifr',
                         **ONEZONE_DEFAULTS)
    sz.tau_star = tau_star
    sz.eta = eta
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color='gray', 
                      linewidth=2,
                      label='None', 
                      marker_labels=False)
    # hist = vice.history(name)
    # mstar_onset = hist['mstar'][int(ifr.onset/dt)-1]
    # print(mstar_onset / (hist['mstar'][-1] - mstar_onset))

    # Pre-enriched infall with constant metallicity
    ofe_in = 0.3
    for i, feh_in in enumerate([-2.0, -1.5, -1.0]):
        Zfe_in = xh_to_z(feh_in)
        Zo_in = xh_to_z(ofe_in + feh_in, element='o')
        name = str(parent_dir / f'ZinConst{i+1:d}')
        sz = vice.singlezone(name=name,
                             func=ifr, 
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = tau_star
        sz.eta = eta
        sz.Zin = {'fe': Zfe_in, 'o': Zo_in}
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name, 
                          fig=subfigs[0], axs=axs0, 
                          linestyle='-', 
                          color=None,
                          label=r'$%s$' % feh_in, 
                          marker_labels=False)
    
    # Exponentially increasing Zin
    name = str(parent_dir / 'ZinExp')
    feh_in = -1.0
    tau_Zin = 2 # exponential timescale of infall enrichment
    Zfe_in = lambda t: xh_to_z(feh_in) * (1 - m.exp(-t/tau_Zin))
    Zo_in = lambda t: xh_to_z(ofe_in + feh_in, 'o') * (1 - m.exp(-t/tau_Zin))
    sz = vice.singlezone(name=name,
                         func=ifr, 
                         mode='ifr',
                         **ONEZONE_DEFAULTS)
    sz.tau_star = tau_star
    sz.eta = eta
    sz.Zin = {'fe': Zfe_in, 'o': Zo_in}
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color=None,
                      label=r'Exp (2 Gyr) $\to -1.0$', 
                      marker_labels=False)
    # Axes labels
    axs0[0].text(0.95, 0.95, r'[O/Fe]$_{\rm in}=+%s$' % ofe_in, 
                 ha='right', va='top', transform=axs0[0].transAxes)
    axs0[0].legend(title=r'[Fe/H]$_{\rm in}$', frameon=False)
    
    # Second panel: variable [O/Fe]
    axs1 = setup_axes(subfigs[1], xlim=XLIM, ylim=YLIM)
    # Pristine infall for reference
    plot_vice_onezone(str(parent_dir / 'ZinNone'), 
                      fig=subfigs[1], axs=axs1, 
                      linestyle='-', 
                      color='gray', 
                      linewidth=2,
                      label='None', 
                      marker_labels=False)
    # Exponentially increasing Zin
    name = str(parent_dir / 'ZinExp_VaryOFe')
    feh_in = -1.0
    oh_in = -1.0
    tau_Zfe = 5 # exponential timescale of infall enrichment
    tau_Zo = 2
    Zfe_in = lambda t: xh_to_z(feh_in) * (1 - m.exp(-t/tau_Zfe))
    Zo_in = lambda t: xh_to_z(oh_in, 'o') * (1 - m.exp(-t/tau_Zo))
    sz = vice.singlezone(name=name,
                         func=ifr, 
                         mode='ifr',
                         **ONEZONE_DEFAULTS)
    sz.tau_star = tau_star
    sz.eta = eta
    sz.Zin = {'fe': Zfe_in, 'o': Zo_in}
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[1], axs=axs1, 
                      linestyle='-', 
                      color=None,
                      label='Exp', 
                      marker_labels=False)
    
    # Axes labels
    axs1[0].text(0.95, 0.95, r'Variable [O/Fe]$_{\rm in}$', 
                 ha='right', va='top', transform=axs1[0].transAxes)
    axs1[0].legend(title=r'[Fe/H]$_{\rm in}$', frameon=False)
    
    fig.savefig(paths.figures / 'enriched_infall')
    plt.close()

def xh_to_z(xh, element='fe'):
    """Convert logarithmic abundance [X/H] to linear abundance Z."""
    return vice.solar_z[element] * 10 ** xh


if __name__ == '__main__':
    main()
