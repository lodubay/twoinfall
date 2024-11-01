"""
This script plots one-zone models which track the evolution of just 
CCSN products.
"""

import numpy as np
import matplotlib.pyplot as plt
import vice

from apogee_sample import APOGEESample
from multizone.src.models import twoinfall_sf_law, equilibrium_mass_loading
from utils import twoinfall_onezone
from track_and_mdf import plot_mdf_curve
from _globals import ONEZONE_DEFAULTS, END_TIME, ONE_COLUMN_WIDTH
import paths
from colormaps import paultol

RADIUS = 8.
ZONE_WIDTH = 0.1
XLIM = (-1.6, 0.6)
YLIM = (-0.18, 0.48)
FIRST_INFALL = 1.
SECOND_INFALL = 10.
ONSET = 4.2


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.vibrant.colors)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 4), wspace=0., hspace=0.)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='y', labelleft=False)
    axs = [ax0, ax1]

    params = ONEZONE_DEFAULTS
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    params['tau_star'] = twoinfall_sf_law(area)
    params['eta'] = 0. # dummy value

    # J21 yields
    from multizone.src.yields import J21
    # Eta depends on yields
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_oh_history(axs, params, 'J21')
    
    # W24 x2 yields
    from multizone.src.yields import W24
    vice.yields.ccsne.settings['o'] *= 2.
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_oh_history(axs, params, 'W24x2')
    
    # W24 yields
    vice.yields.ccsne.settings['o'] *= 0.5
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_oh_history(axs, params, 'W24')
    
    # W24 x2 outflows
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)/2.
    plot_oh_history(axs, params, 'W24etax0.5')

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 0.5))
    
    ax0.set_xlabel('P([O/H])', size='small')
    ax0.set_xlim((1.1, 0))
    ax0.set_ylabel('[O/H]')

    ax1.set_xlabel('Lookback Time [Gyr]')
    ax1.set_xlim((-1, 13.7))
    ax1.set_ylim((-1, 0.3))

    ax1.text(0.05, 0.95, r'$\tau_2=%s$ Gyr' % SECOND_INFALL, 
             transform=ax1.transAxes, va='top')
    ax1.legend(frameon=False)

    fig.savefig(paths.figures / 'ccsn_yield')
    plt.close()


def plot_oh_history(axs, params, name, label=''):
    output_dir = paths.data / 'onezone' / 'ccsn_yield'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if label == '':
        ycco = vice.yields.ccsne.settings['o']
        label = r'$y^{\rm cc}_{\rm O} = %s$, $\eta=%s$' % (round(ycco, 4), round(params['eta'], 2))
    
    eta_func = lambda r: params['eta']
    ifr = twoinfall_onezone(
        RADIUS, 
        first_timescale=FIRST_INFALL,
        second_timescale=SECOND_INFALL, 
        onset=ONSET,
        mass_loading=eta_func,
        dr=ZONE_WIDTH
    )
    # Run one-zone model
    fullname = str(output_dir / name)
    sz = vice.singlezone(name=fullname,
                         func=ifr,
                         mode='ifr',
                         **params)
    simtime = np.arange(0, END_TIME + params['dt'], params['dt'])
    sz.run(simtime, overwrite=True)
    # Plot [O/H] vs age
    hist = vice.history(fullname)
    axs[1].plot(hist['lookback'], hist['[o/h]'], label=label)
    mdf = vice.mdf(fullname)
    mdf_bins = mdf['bin_edge_left'] + mdf['bin_edge_right'][-1:]
    plot_mdf_curve(axs[0], mdf['dn/d[o/h]'], mdf_bins, smoothing=0.01,
                   orientation='horizontal')


if __name__ == '__main__':
    main()
