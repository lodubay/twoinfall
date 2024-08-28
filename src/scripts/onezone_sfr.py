"""
This script plots a one-zone VICE model with the equilibrium-calibrated
mass-loading factor.
"""

import numpy as np
import matplotlib.pyplot as plt

import vice

from multizone.src.yields import W23
from multizone.src.disks import equilibrium_mass_loading
from multizone.src.models import twoinfall, twoinfall_sf_law
from track_and_mdf import setup_figure, plot_vice_onezone
from apogee_sample import APOGEESample
import paths
from utils import get_bin_centers
from _globals import ONEZONE_DEFAULTS, ZONE_WIDTH

RADIUS = 8.
ONSET = 3.5 # Gyr
FEH_LIM = (-1.3, 0.6)
OFE_LIM = (-0.12, 0.48)

def main():
    plt.style.use(paths.styles / "paper.mplstyle")
    fig, axs = setup_figure(xlim=FEH_LIM, ylim=OFE_LIM)

    # Plot underlying APOGEE contours
    apogee_data = APOGEESample.load()
    apogee_solar = apogee_data.region(galr_lim=(7, 9), absz_lim=(0, 2))
    apogee_solar.plot_kde2D_contours(axs[0], 'FE_H', 'O_FE', c='k', lw=1,
                                     plot_kwargs={'zorder': 1})
    feh_df, bin_edges = apogee_solar.mdf(col='FE_H', range=FEH_LIM, 
                                         smoothing=0.2)
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), 'k-')
    ofe_df, bin_edges = apogee_solar.mdf(col='O_FE', range=OFE_LIM, 
                                         smoothing=0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 'k-')
    
    output_dir = paths.data / "onezone" / "eta"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    name = str(output_dir / "equilibrium")
    
    simtime = np.arange(0, 13.21, 0.01)
    
    eta_func = equilibrium_mass_loading(tau_star=2., tau_sfh=10., alpha_h_eq=0.2)
    
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    
    sz = vice.singlezone(
        name = name,
        func = twoinfall(
            RADIUS, 
            first_timescale=1., 
            second_timescale=10., 
            onset=ONSET),
        mode = "ifr",
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta_func(RADIUS)
    print(eta_func(RADIUS))
    sz.tau_star = twoinfall_sf_law(area, onset=ONSET)
    sz.run(simtime, overwrite=True)
    
    plot_vice_onezone(name, fig=fig, axs=axs, markers=[])
    # Weight by SFR
    hist = vice.history(name)
    axs[0].scatter(hist['[fe/h]'][::10], hist['[o/fe]'][::10], 
                   s=[20*h for h in hist['sfr'][::10]])
    # Mark every Gyr
    axs[0].scatter(hist['[fe/h]'][::100], hist['[o/fe]'][::100], 
                   s=[5*h for h in hist['sfr'][::100]], c='w', zorder=10)
    
    plt.savefig(paths.figures / 'onezone_sfr')
    plt.close()
    

if __name__ == "__main__":
    main()
