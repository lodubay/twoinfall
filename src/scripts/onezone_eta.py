"""
This script plots a one-zone VICE model with the equilibrium-calibrated
mass-loading factor.
"""

import numpy as np
import matplotlib.pyplot as plt

import vice

from multizone.src.yields import W24
from multizone.src import models
from track_and_mdf import setup_figure, plot_vice_onezone
import paths
from _globals import ONEZONE_DEFAULTS, ZONE_WIDTH

RADIUS = 8.
ONSET = 4. # Gyr


def main():
    plt.style.use(paths.styles / "paper.mplstyle")
    fig, axs = setup_figure()
    
    output_dir = paths.data / "onezone" / "eta"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    name = str(output_dir / "equilibrium")
    
    simtime = np.arange(0, 13.21, 0.01)
    
    eta_func = models.equilibrium_mass_loading(
        tau_star=2., tau_sfh=10., alpha_h_eq=0.1
    )
    
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    
    sz = vice.singlezone(
        name = name,
        func = models.twoinfall(
            RADIUS, 
            first_timescale=1., 
            second_timescale=10., 
            onset=ONSET),
        mode = "ifr",
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta_func(RADIUS)
    print(eta_func(RADIUS))
    sz.tau_star = models.twoinfall_sf_law(area, onset=ONSET)
    sz.run(simtime, overwrite=True)
    
    plot_vice_onezone(name, fig=fig, axs=axs)
    
    plt.savefig(paths.figures / 'onezone_eta')
    plt.close()
    

if __name__ == "__main__":
    main()
