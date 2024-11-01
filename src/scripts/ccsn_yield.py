"""
This script plots one-zone models which track the evolution of just 
CCSN products.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from apogee_sample import APOGEESample
from multizone.src.models import twoinfall_sf_law, equilibrium_mass_loading
from multizone.src import dtds
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

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 4), wspace=0., hspace=0.1)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='y', labelleft=False)
    ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax3 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax2)
    ax3.tick_params(axis='y', labelleft=False)
    axs = [[ax0, ax1], [ax2, ax3]]

    params = ONEZONE_DEFAULTS
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    params['tau_star'] = twoinfall_sf_law(area)
    params['eta'] = 0. # dummy value
    params['RIa'] = dtds.plateau()

    # 3x yields, similar to J21
    scale_yields(3)
    # Eta depends on yields
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_abundance_history(axs[0], params, '3xSol')
    plot_abundance_history(axs[1], params, '3xSol', element='fe')
    
    # 2x yields
    scale_yields(2)
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_abundance_history(axs[0], params, '2xSol')
    plot_abundance_history(axs[1], params, '2xSol', element='fe')
    
    # Solar yields, similar to W24
    scale_yields(1)
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    plot_abundance_history(axs[0], params, '1xSol')
    plot_abundance_history(axs[1], params, '1xSol', element='fe')
    
    # Solar yields, x2 outflows
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)/2.
    plot_abundance_history(axs[0], params, '1xSol_0.5xEta')
    plot_abundance_history(axs[1], params, '1xSol_0.5xEta', element='fe')

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 0.5))

    ax0.set_ylabel('[O/H]')
    ax0.set_ylim((-1.2, 0.3))
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.text(0.05, 0.95, r'$\tau_2=%s$ Gyr' % SECOND_INFALL, 
             transform=ax1.transAxes, va='top')
    ax1.legend(frameon=False)
    
    ax2.set_xlabel('P([X/H])', size='small')
    ax2.set_xlim((1.2, 0))
    ax2.set_ylabel('[Fe/H]')
    ax2.set_ylim((-1.4, 0.4))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax3.set_xlabel('Lookback Time [Gyr]')
    ax3.set_xlim((-1, 14))
    ax3.xaxis.set_major_locator(MultipleLocator(5))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.text(0.05, 0.95, r'$f^{\rm CC}_{\rm Fe}=0.35$, Plateau DTD', 
             transform=ax3.transAxes, va='top')
    ax3.legend(frameon=False)

    fig.savefig(paths.figures / 'ccsn_yield')
    plt.close()


def plot_abundance_history(axs, params, name, element='o', label=''):
    output_dir = paths.data / 'onezone' / 'ccsn_yield'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if label == '':
        ycc = vice.yields.ccsne.settings[element]
        label = r'$\eta=%s$, $y^{\rm CC}_{\rm %s} = %s$' % (
            round(params['eta'], 1), element.capitalize(), f'{ycc:.02g}'
        )
    
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
    axs[1].plot(hist['lookback'], hist['[%s/h]' % element], label=label)
    mdf = vice.mdf(fullname)
    mdf_bins = mdf['bin_edge_left'] + mdf['bin_edge_right'][-1:]
    plot_mdf_curve(axs[0], mdf['dn/d[%s/h]' % element], mdf_bins, smoothing=0.01,
                   orientation='horizontal')
    

def scale_yields(scale, fe_cc_frac=0.35):
    r"""
    Adopt yields scaled according to the Solar metallicity.

    The O yield of CC SNe is computed:
    $ y^{\rm cc}_{\rm O} = S\times Z_{\rm O,\odot} $
    where $S$ is the given scale factor. For Fe, the yields are:
    $ y^{\rm cc}_{\rm Fe} = S\times f_{\rm cc} \times Z_{\rm Fe,\odot} $
    and
    $ y^{\rm Ia}_{\rm Fe} = S \times (1 - f_{\rm cc}) \times 
    Z_{\rm Fe,\odot} $,
    where $f_{\rm cc}$ is the fraction of Solar Fe produced in CC SNe.
    
    Parameters
    ----------
    scale : float
        Scale of total yields relative to the Solar abundance.
    fe_cc_frac : float [default: 0.35]
        Fraction of Solar Fe produced in CC SNe.
    
    """
    vice.yields.ccsne.settings['o'] = scale * vice.solar_z['o']
    vice.yields.ccsne.settings['fe'] = scale * fe_cc_frac * vice.solar_z['fe']
    vice.yields.sneia.settings['o'] = 0.
    vice.yields.sneia.settings['fe'] = scale * (1 - fe_cc_frac) * vice.solar_z['fe']


if __name__ == '__main__':
    main()
