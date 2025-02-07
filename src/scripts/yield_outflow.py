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
from utils import twoinfall_onezone, get_bin_centers
from track_and_mdf import plot_mdf_curve
from _globals import ONEZONE_DEFAULTS, END_TIME, ONE_COLUMN_WIDTH
import paths
from colormaps import paultol

AFE_CC = 0.45
SNIA_FE_YIELD = 0.7 # Msun
RADIUS = 8.
ZONE_WIDTH = 2
FIRST_INFALL = 1
SECOND_INFALL = 15.
ONSET = 4.2

OH_LIM = (-0.6, 0.8)
FEH_LIM = (-0.8, 0.6)
OFE_LIM = (-0.15, 0.5)


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.bright.colors)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2.2*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 4), wspace=0., hspace=0.)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.tick_params(axis='x', labelbottom=False)
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax2)
    ax3.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax4 = fig.add_subplot(gs[2,0], sharex=ax2)
    ax5 = fig.add_subplot(gs[2,1], sharex=ax3, sharey=ax4)
    ax5.tick_params(axis='y', labelleft=False)
    axs = [[ax0, ax1], [ax2, ax3], [ax4, ax5]]

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 0.5))
    age_bin_width = 1. # Gyr
    age_bins = np.arange(0, 13 + age_bin_width, age_bin_width)
    age_bin_centers = get_bin_centers(age_bins)
    
    # Plot [C/N] ages
    # data_color = 'purple'
    # oh_bins = local_sample.binned_modes('O_H', 'CN_AGE', age_bins)
    # ax1.errorbar(age_bin_centers, oh_bins['mode'], 
    #              xerr=age_bin_width/2, yerr=oh_bins['error'],
    #              linestyle='none', c=data_color, capsize=1, marker='.',
    #              zorder=0, label='[C/N] modes')
    # feh_bins = local_sample.binned_modes('FE_H', 'CN_AGE', age_bins)
    # ax3.errorbar(age_bin_centers, feh_bins['mode'], 
    #              xerr=age_bin_width/2, yerr=feh_bins['error'],
    #              linestyle='none', c=data_color, capsize=1, marker='.',
    #              zorder=0, label='[C/N] modes')
    # ofe_bins = local_sample.binned_modes('O_FE', 'CN_AGE', age_bins)
    # ax5.errorbar(age_bin_centers, ofe_bins['mode'], 
    #              xerr=age_bin_width/2, yerr=ofe_bins['error'],
    #              linestyle='none', c=data_color, capsize=1, marker='.',
    #              zorder=0, label='[C/N] modes')
    
    # Plot L23 ages
    data_color = '0.6'
    oh_bins = local_sample.binned_modes('O_H', 'L23_AGE', age_bins)
    ax1.errorbar(age_bin_centers, oh_bins['mode'], 
                 xerr=age_bin_width/2, yerr=oh_bins['error'],
                 linestyle='none', c=data_color, capsize=1, marker='.',
                 zorder=0, label='L23 modes')
    feh_bins = local_sample.binned_modes('FE_H', 'L23_AGE', age_bins)
    ax3.errorbar(age_bin_centers, feh_bins['mode'], 
                 xerr=age_bin_width/2, yerr=feh_bins['error'],
                 linestyle='none', c=data_color, capsize=1, marker='.',
                 zorder=0, label='L23 modes')
    ofe_bins = local_sample.binned_modes('O_FE', 'L23_AGE', age_bins)
    ax5.errorbar(age_bin_centers, ofe_bins['mode'], 
                 xerr=age_bin_width/2, yerr=ofe_bins['error'],
                 linestyle='none', c=data_color, capsize=1, marker='.',
                 zorder=0, label='L23 modes')
    
    # Plot APOGEE abundance distributions in marginal panels
    oh_df, bin_edges = local_sample.mdf(col='O_H', range=OH_LIM, 
                                        smoothing=0.1)
    ax0.plot(oh_df / max(oh_df), get_bin_centers(bin_edges),
             color=data_color, linestyle='-', linewidth=2, marker=None)
    feh_df, bin_edges = local_sample.mdf(col='FE_H', range=FEH_LIM, 
                                         smoothing=0.1)
    ax2.plot(feh_df / max(feh_df), get_bin_centers(bin_edges),
             color=data_color, linestyle='-', linewidth=2, marker=None)
    ofe_df, bin_edges = local_sample.mdf(col='O_FE', range=OFE_LIM, 
                                         smoothing=0.05)
    ax4.plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
             color=data_color, linestyle='-', linewidth=2, marker=None)

    params = ONEZONE_DEFAULTS
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    params['tau_star'] = twoinfall_sf_law(area, onset=ONSET)
    params['eta'] = 0. # dummy value
    params['RIa'] = dtds.plateau()

    # 3x yields, similar to J21
    params['eta'] = 2.4
    run_plot_model(axs, 3, params, yia_scale=1.0)
    # 2x yields
    params['eta'] = 1.4
    run_plot_model(axs, 2, params, yia_scale=1.1)
    # Solar yields, similar to W24
    params['eta'] = 0.2
    run_plot_model(axs, 1, params, yia_scale=1.3)

    ax0.set_ylabel('[O/H]')
    ax0.set_xlim((1.2, 0))
    ax0.set_ylim(OH_LIM)
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.set_xlim((-1, 14))
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    # ax1.text(0.05, 0.95, r'$\tau_1=%s$ Gyr' % FIRST_INFALL, 
    #          transform=ax1.transAxes, va='top')
    # ax1.text(0.05, 0.89, r'$\tau_2=%s$ Gyr' % SECOND_INFALL, 
    #          transform=ax1.transAxes, va='top')
    ax1.legend(frameon=False, loc='upper left')
    
    ax2.set_ylabel('[Fe/H]')
    ax2.set_ylim(FEH_LIM)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ax3.text(0.05, 0.95, r'$y^{\rm CC}_{\rm Fe} / y^{\rm CC}_{\rm O}=%s$' % FE_CC_FRAC, 
    #          transform=ax3.transAxes, va='top')
    # ax3.legend(frameon=False)

    ax4.set_ylabel('[O/Fe]')
    ax4.set_xlabel('P([X/H])', size='small')
    ax4.set_ylim(OFE_LIM)
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.05))

    # ax5.text(0.05, 0.95, r'Plateau DTD', transform=ax5.transAxes, va='top')
    ax5.set_xlabel('Lookback Time [Gyr]')
    # ax5.legend(frameon=False)

    fig.savefig(paths.figures / 'yield_outflow')
    plt.close()


def run_plot_model(axs, scale, params, yia_scale=1., 
                   label='', c=None, ls='-'):
    output_dir = paths.data / 'onezone' / 'yield_outflow'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    scale_yields(scale, afe_cc=AFE_CC, yia_scale=yia_scale)
    # params['eta'] = equilibrium_mass_loading(tau_star=0.)(RADIUS)
    # params['eta'] *= eta_scale
    RIa = vice.yields.sneia.settings['fe'] / SNIA_FE_YIELD
    if label == '':
        label = r'$y/Z_\odot = %s$, $\eta=%s$' % (
            scale, round(params['eta'], 2)
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
    name = '%sxSol_Eta%s' % (scale, params['eta'])
    fullname = str(output_dir / name)
    sz = vice.singlezone(name=fullname,
                         func=ifr,
                         mode='ifr',
                         **params)
    simtime = np.arange(0, END_TIME + params['dt'], params['dt'])
    sz.run(simtime, overwrite=True)

    # Plots
    plot_abundance_history(axs[0], fullname, '[o/h]', c=c, ls=ls, label=label)
    plot_abundance_history(axs[1], fullname, '[fe/h]', c=c, ls=ls, label=label)
    plot_abundance_history(axs[2], fullname, '[o/fe]', c=c, ls=ls, label=label)


def plot_abundance_history(axs, fullname, col, label='', c=None, ls='-'):
    hist = vice.history(fullname)
    axs[1].plot(hist['lookback'], hist[col], label=label, color=c, ls=ls)
    mdf = vice.mdf(fullname)
    mdf_bins = mdf['bin_edge_left'] + mdf['bin_edge_right'][-1:]
    plot_mdf_curve(axs[0], mdf['dn/d%s' % col], mdf_bins, smoothing=0.02,
                   orientation='horizontal', color=c, ls=ls)
    

def scale_yields(scale, afe_cc=0.45, yia_scale=1.):
    r"""
    Adopt yields scaled according to the Solar metallicity.

    The O yield of CC SNe is computed:
    $ y^{\rm cc}_{\rm O} = S\times Z_{\rm O,\odot} $
    where $S$ is the given scale factor. For Fe, the yields are:
    $ y^{\rm cc}_{\rm Fe} = S\times f_{\rm cc} \times Z_{\rm Fe,\odot} $
    and
    $ y^{\rm Ia}_{\rm Fe} = S \times (1 - f_{\rm cc}) \times 
    Z_{\rm Fe,\odot} $,
    where $f_{\rm cc}$ is the ratio of Fe to O produced in CC SNe.
    
    Parameters
    ----------
    scale : float
        Scale of total yields relative to the Solar abundance.
    afe_cc : float [default: 0.45]
        Logarithmic ratio of O to Fe produced by CC SNe (the CC SN plateau).
    
    """
    vice.yields.ccsne.settings['o'] = scale * vice.solar_z['o']
    vice.yields.ccsne.settings['fe'] = scale * 10**-afe_cc * vice.solar_z['fe']
    vice.yields.sneia.settings['o'] = 0.
    vice.yields.sneia.settings['fe'] = yia_scale * scale * (1 - 10**-afe_cc) * vice.solar_z['fe']


if __name__ == '__main__':
    main()
