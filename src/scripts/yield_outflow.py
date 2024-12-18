"""
This script plots the evolution of gas abundance in one-zone models which 
compare several factors of yields and outflows.
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

FE_CC_FRAC = 0.35
SNIA_FE_YIELD = 0.7 # Msun
RADIUS = 8.
ZONE_WIDTH = 0.1
FIRST_INFALL = 1.
SECOND_INFALL = 15.
ONSET = 4.2


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.vibrant.colors)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 3*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 4), wspace=0., hspace=0.1)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='y', labelleft=False)
    ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax3 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax2)
    ax3.tick_params(axis='y', labelleft=False)
    ax4 = fig.add_subplot(gs[2,0], sharex=ax0)
    ax5 = fig.add_subplot(gs[2,1], sharex=ax1, sharey=ax4)
    ax5.tick_params(axis='y', labelleft=False)
    axs = [[ax0, ax1], [ax2, ax3], [ax4, ax5]]

    params = ONEZONE_DEFAULTS
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    params['tau_star'] = twoinfall_sf_law(area)
    params['eta'] = 0. # dummy value
    params['RIa'] = dtds.plateau()

    # 3x yields, similar to J21
    run_plot_model(axs, 3, params, eta_scale=1)
    # 2x yields
    run_plot_model(axs, 2, params, eta_scale=1)
    # Solar yields, similar to W24
    run_plot_model(axs, 1, params, eta_scale=1)
    # Solar yields, x0.5 outflows
    run_plot_model(axs, 1, params, eta_scale=0.5)
    # Higher yIa
    yia_scale = 1.3
    run_plot_model(axs, 1, params, eta_scale=0.5, yia_scale=yia_scale, 
                   c=paultol.vibrant.colors[3], ls='--', 
                   label=r'$y_{\rm Ia}\times%s$' % yia_scale)
    # Solar yields, no outflows
    # run_plot_model(axs, 1, params, eta_scale=0.)

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 0.5))

    ax0.set_ylabel('[O/H]')
    ax0.set_xlim((1.2, 0))
    ax0.set_ylim((-1.2, 0.3))
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.set_xlim((-1, 14))
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.text(0.05, 0.95, r'$\tau_2=%s$ Gyr' % SECOND_INFALL, 
             transform=ax1.transAxes, va='top')
    ax1.legend(frameon=False)
    
    ax2.set_ylabel('[Fe/H]')
    ax2.set_ylim((-1.4, 0.4))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax3.text(0.05, 0.95, r'$f^{\rm CC}_{\rm Fe}=%s$, Plateau DTD' % FE_CC_FRAC, 
             transform=ax3.transAxes, va='top')
    ax3.legend(frameon=False)

    ax4.set_ylabel('[O/Fe]')
    ax4.set_xlabel('P([X/H])', size='small')
    ax4.set_ylim((-0.15, 0.5))
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax5.set_xlabel('Lookback Time [Gyr]')
    ax5.legend(frameon=False)

    fig.savefig(paths.figures / 'yield_outflow')
    plt.close()


def run_plot_model(axs, scale, params, eta_scale=1., yia_scale=1., 
                   label='', c=None, ls='-'):
    output_dir = paths.data / 'onezone' / 'yield_outflow'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    scale_yields(scale, fe_cc_frac=FE_CC_FRAC, yia_scale=yia_scale)
    params['eta'] = equilibrium_mass_loading(tau_sfh=SECOND_INFALL)(RADIUS)
    params['eta'] *= eta_scale
    RIa = vice.yields.sneia.settings['fe'] / SNIA_FE_YIELD
    if label == '':
        label = r'$\eta=%s$, $y/Z_\odot = %s$' % (
            round(params['eta'], 1), scale
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
    name = '%sxSol' % scale
    if eta_scale != 1:
        name += '_%sxEta' % eta_scale
    fullname = str(output_dir / name)
    sz = vice.singlezone(name=fullname,
                         func=ifr,
                         mode='ifr',
                         **params)
    simtime = np.arange(0, END_TIME + params['dt'], params['dt'])
    sz.run(simtime, overwrite=True)

    # Plots
    plot_abundance_history(axs[0], fullname, '[o/h]', label=label, c=c, ls=ls)
    plot_abundance_history(axs[1], fullname, '[fe/h]', label=label, c=c, ls=ls)
    plot_abundance_history(axs[2], fullname, '[o/fe]', label=label, c=c, ls=ls)


def plot_abundance_history(axs, fullname, col, label='', c=None, ls='-'):
    hist = vice.history(fullname)
    axs[1].plot(hist['lookback'], hist[col], label=label, color=c, ls=ls)
    mdf = vice.mdf(fullname)
    mdf_bins = mdf['bin_edge_left'] + mdf['bin_edge_right'][-1:]
    plot_mdf_curve(axs[0], mdf['dn/d%s' % col], mdf_bins, smoothing=0.02,
                   orientation='horizontal', color=c, ls=ls)
    

def scale_yields(scale, fe_cc_frac=0.35, yia_scale=1.):
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
    vice.yields.sneia.settings['fe'] = yia_scale * scale * (1 - fe_cc_frac) * vice.solar_z['fe']


if __name__ == '__main__':
    main()
