"""
This script plots one-zone model results with pre-enriched infall.
"""

import math as m

import numpy as np
import matplotlib.pyplot as plt
import vice

# from multizone.src.yields import W24
from multizone.src.yields import J21
from multizone.src.models import twoinfall_sf_law, equilibrium_mass_loading
from track_and_mdf import setup_axes, plot_vice_onezone
from apogee_sample import APOGEESample
import paths
from utils import get_bin_centers, twoinfall_onezone
from _globals import ONEZONE_DEFAULTS, ZONE_WIDTH, ONE_COLUMN_WIDTH, END_TIME
from colormaps import paultol

XLIM = (-1.4, 0.4)
YLIM = (-0.1, 0.54)
RADIUS = 8. # kpc
ONSET = 3.2


def main():
    # Set up figure: two panels, vertically stacked
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2*ONE_COLUMN_WIDTH))
    # TODO figure out how to make subfigures exactly equal
    # gs = fig.add_gridspec(13, 7, hspace=0., wspace=0.)
    # subfigs = [fig.add_subfigure(gs[i:i+w,:]) for i, w in zip((0, 6), (6, 7))]
    subfigs = fig.subfigures(2, 1, hspace=-0.1)
    
    # One-zone model settings
    params = ONEZONE_DEFAULTS
    params['Mg0'] = 1e6
    dt = params['dt']
    simtime = np.arange(0, END_TIME + dt, dt)
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    # eta_func = equilibrium_mass_loading()
    eta_func = vice.milkyway.default_mass_loading
    params['eta'] = eta_func(RADIUS)
    ifr = twoinfall_onezone(RADIUS, onset=ONSET, mass_loading=eta_func, dt=dt, 
                            dr=ZONE_WIDTH)
    params['func'] = ifr
    params['mode'] = 'ifr'
    params['tau_star'] = twoinfall_sf_law(area, onset=ifr.onset)
    parent_dir = paths.onezone / 'enriched_infall'
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
    
    # First panel: constant [O/Fe]
    axs0 = setup_axes(subfigs[0], xlim=XLIM, ylim=YLIM, xlabel=True)
    # Pristine infall
    name = str(parent_dir / 'ZinNone')
    sz = vice.singlezone(name=name, **params)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color='gray', 
                      linewidth=2,
                      label='None', 
                      marker_labels=False)

    # Pre-enriched infall with constant metallicity
    ofe_in = 0.3
    for i, feh_in in enumerate([-1.0]):
        name = str(parent_dir / f'ZinConst{int(-10*feh_in):d}')
        sz = vice.singlezone(name=name, **params)
        sz.Zin = {
            'fe': xh_to_z(feh_in), 
            'o': xh_to_z(ofe_in + feh_in, element='o'),
        }
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name, 
                          fig=subfigs[0], axs=axs0, 
                          linestyle='-', 
                          color=None,
                          label=r'$%s$' % feh_in, 
                          marker_labels=False)
    
    # Second infall only
    name = str(parent_dir / 'ZinSecond10')
    feh_in = -1.0
    sz = vice.singlezone(name=name, **params)
    sz.Zin = {
        'fe': step(ifr.onset, feh_in, 'fe'), 
        'o': step(ifr.onset, ofe_in + feh_in, 'o')
    }
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color=None,
                      label=r'Second Infall [Fe/H]$=-1.0$', 
                      marker_labels=False)
    
    # Exponentially increasing Zin to [Fe/H]=-1.0
    name = str(parent_dir / 'ZinExp2')
    feh_in = -1.0
    sz = vice.singlezone(name=name, **params)
    sz.Zin = {
        'fe': exponential(2, feh_in, 'fe'), 
        'o': exponential(2, ofe_in + feh_in, 'o')
    }
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color=None,
                      label=r'Exp (2 Gyr) $\to-1.0$', 
                      marker_labels=False)

    # Exponentially increasing Zin to [Fe/H]=-1
    name = str(parent_dir / 'ZinExp5')
    sz = vice.singlezone(name=name, **params)
    sz.Zin = {
        'fe': exponential(5, feh_in, 'fe'), 
        'o': exponential(5, ofe_in + feh_in, 'o')
    }
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[0], axs=axs0, 
                      linestyle='-', 
                      color=None,
                      label=r'Exp (5 Gyr) $\to-1.0$', 
                      marker_labels=False)
    
    # Axes labels
    axs0[0].text(0.95, 0.95, r'[O/Fe]$_{\rm in}=+%s$' % ofe_in, 
                 ha='right', va='top', transform=axs0[0].transAxes)
    axs0[0].legend(title=r'[Fe/H]$_{\rm in}$', frameon=False, loc='lower left')
    
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
    
    # Constant [O/H], variable [O/Fe]
    name = str(parent_dir / 'Zin10_VaryOFe')
    oh_in = -1.0
    Zo_in = xh_to_z(feh_in, 'o')
    tau_ofe = 5
    ofe_in = lambda t: 0.45 * m.exp(-t/tau_ofe)
    Zfe_in = lambda t: xh_to_z(oh_in - ofe_in(t), 'fe')
    sz = vice.singlezone(name=name, **params)
    sz.Zin = {'fe': Zfe_in, 'o': Zo_in}
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, 
                      fig=subfigs[1], axs=axs1, 
                      linestyle='-', 
                      color=None,
                      label='-1.0', 
                      marker_labels=False)
    
    # Exponentially increasing Zin
    # name = str(parent_dir / 'ZinExp_VaryOFe')
    # oh_in = -1.0
    # tau_Zo = 2
    # Zo_in = lambda t: xh_to_z(oh_in, 'o') * (1 - m.exp(-t/tau_Zo))
    # Zfe_in = lambda t: xh_to_z(z_to_xh(Zo_in(t), 'o') - ofe_in(t))
    # sz = vice.singlezone(name=name,
    #                      func=ifr, 
    #                      mode='ifr',
    #                      **ONEZONE_DEFAULTS)
    # sz.tau_star = tau_star
    # sz.eta = eta
    # sz.Zin = {'fe': Zfe_in, 'o': Zo_in}
    # sz.run(simtime, overwrite=True)
    # plot_vice_onezone(name, 
    #                   fig=subfigs[1], axs=axs1, 
    #                   linestyle='-', 
    #                   color=None,
    #                   label='Exp', 
    #                   marker_labels=False)
    
    # Axes labels
    axs1[0].text(0.95, 0.95, r'Variable [O/Fe]$_{\rm in}$', 
                 ha='right', va='top', transform=axs1[0].transAxes)
    axs1[0].legend(title=r'[O/H]$_{\rm in}$', frameon=False)
    
    fig.savefig(paths.figures / 'enriched_infall')
    plt.close()


def exponential(timescale, xh, element):
    """Returns an exponential function which asymptotes to the given [X/H]."""
    return lambda t: xh_to_z(xh, element) * (1 - m.exp(-t/timescale))

def step(onset, xh2, element, xh1=-float('inf')):
    """Step function to the given [X/H] at the given onset time."""
    return lambda t: xh_to_z(xh2, element) if t >= onset else xh_to_z(xh1, element)

def xh_to_z(xh, element='fe'):
    """Convert logarithmic abundance [X/H] to linear abundance Z."""
    return vice.solar_z[element] * 10 ** xh

def z_to_xh(z, element='fe'):
    """Convert linear abundance Z to logarithmic abundance [X/H]"""
    return m.log10(z / vice.solar_z[element])


if __name__ == '__main__':
    main()
