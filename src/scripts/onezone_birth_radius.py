"""
This script plots a one-zone VICE model and compares it to birth radius estimates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import vice

from multizone.src.yields import W24mod
from multizone.src.models import twoinfall_sf_law, equilibrium_mass_loading
from multizone.src.dtds import powerlaw, exponential
from track_and_mdf import setup_figure, plot_vice_onezone
# from apogee_sample import APOGEESample
import paths
from utils import get_bin_centers, twoinfall_onezone, box_smooth
from _globals import ONEZONE_DEFAULTS
from colormaps import paultol

RADIUS = 8.
ZONE_WIDTH = 2.
ONSET = 4.2
FIRST_TIMESCALE = 1.
SECOND_TIMESCALE = 10.
FEH_LIM = (-1.4, 0.6)
OFE_LIM = (-0.08, 0.48)

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig, axs = setup_figure(xlim=FEH_LIM, ylim=OFE_LIM)

    # Plot abundances within birth radius bin
    Rb_data = pd.read_csv(paths.data / 'Rb_liam' / 'Rb_LAMOST.csv')
    Rb_solar = Rb_data[(Rb_data['Rb'] >= 7) & (Rb_data['Rb'] < 9) & (Rb_data['ZMAX'] < 2)]
    pcm = axs[0].hexbin(Rb_solar['FEH'], Rb_solar['ALPHA_FE'],
                  gridsize=50, bins=None,
                  extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
                  cmap='Greys', linewidths=0.2)
    cax = axs[0].inset_axes([0.05, 0.05, 0.05, 0.8])
    fig.colorbar(pcm, cax=cax, orientation='vertical')
    
    # abundance distributions
    bins = 100
    feh_df, bin_edges = np.histogram(Rb_solar['FEH'], bins=bins, range=FEH_LIM, 
                                  density=True)
    feh_df = box_smooth(feh_df, bin_edges, 0.2)
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), 
                color='gray', linestyle='-', marker=None)
    
    ofe_df, bin_edges = np.histogram(Rb_solar['ALPHA_FE'], bins=bins, range=OFE_LIM, 
                                  density=True)
    ofe_df = box_smooth(ofe_df, bin_edges, 0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
                color='gray', linestyle='-', marker=None)
    
    # Set up output directory
    output_dir = paths.data / 'onezone' / 'birth_radius'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    name = str(output_dir / 'equilibrium')
    
    # One-zone model parameters
    simtime = np.arange(0, 13.21, 0.01)
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)
    eta_func = equilibrium_mass_loading(
        tau_star=0.,
        # tau_sfh=SECOND_TIMESCALE, 
        equilibrium=0.
    )
    # eta_func = vice.milkyway.default_mass_loading
    ifr = twoinfall_onezone(
        RADIUS, 
        first_timescale=FIRST_TIMESCALE, 
        second_timescale=SECOND_TIMESCALE, 
        onset=ONSET,
        mass_loading=eta_func,
        dr=ZONE_WIDTH
    )
    
    sz = vice.singlezone(
        name = name,
        func = ifr,
        mode = 'ifr',
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta_func(RADIUS)
    sz.tau_star = twoinfall_sf_law(area, onset=ifr.onset)
    # sz.RIa = exponential()
    sz.run(simtime, overwrite=True)
    
    model_color = paultol.bright.colors[0]
    plot_vice_onezone(name, fig=fig, axs=axs, markers=[], 
                      color=model_color)
    # Weight by SFR
    hist = vice.history(name)
    axs[0].scatter(hist['[fe/h]'][::10], hist['[o/fe]'][::10], 
                   s=[20*h/max(hist['sfr']) for h in hist['sfr'][::10]],
                   c=model_color)
    # Mark every Gyr
    axs[0].scatter(hist['[fe/h]'][::100], hist['[o/fe]'][::100], 
                   s=[5*h/max(hist['sfr']) for h in hist['sfr'][::100]], 
                   c='w', zorder=10)
    
    plt.savefig(paths.figures / 'onezone_birth_radius')
    plt.close()
    

if __name__ == '__main__':
    main()
