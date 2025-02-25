"""
This script plots one-zone model abundance tracks with my parameters
versus parameters from Spitoni et al. (2021), as closely as I can replicate.
"""

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone.src.yields import yZ1
from multizone.src.models import twoinfall_sf_law, twoinfall_expvar
from multizone.src import outflows
from multizone.src.models.gradient import thick_to_thin_ratio, gradient
from multizone.src.models.normalize import normalize_ifrmode
from track_and_mdf import setup_figure, plot_vice_onezone
from colormaps import paultol
import paths
from _globals import END_TIME, ONEZONE_DEFAULTS, ONE_COLUMN_WIDTH
from utils import twoinfall_onezone

XLIM = (-2.1, 0.8)
YLIM = (-0.2, 0.499)
ONSET = 4.2
ZONE_WIDTH = 4
RADII = [4, 8, 12]
S21_TAU1 = [0.115, 0.103, 0.449]
S21_TAU2 = [3.756, 4.110, 10.986]
S21_TMAX = [4.647, 4.085, 3.059]
S21_THICK_THIN_RATIO = [1/3.805, 1/5.635, 1/10.348]
S21_SFE1 = [3., 2., 2.]
S21_SFE2 = [1.5, 1, 0.5]


def main(style='paper'):
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', reversed(paultol.highcontrast.colors))

    output_dir = paths.data / 'onezone' / 'spitoni_comparison'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fig, axs = setup_figure(width=ONE_COLUMN_WIDTH, xlim=XLIM, ylim=YLIM)

    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)

    # My parameters
    eta_func = outflows.yZ1()
    for i, radius in enumerate(RADII):
        name = str(output_dir / ('fiducial_%skpc' % int(radius)))
        area = np.pi * ((radius + ZONE_WIDTH/2)**2 - (radius - ZONE_WIDTH/2)**2)
        tau_star = twoinfall_sf_law(area, onset=ONSET)
        # Initialize infall rate
        ifr = twoinfall_onezone(
            radius, 
            first_timescale=1.,
            second_timescale=twoinfall_expvar.timescale(radius), 
            onset=ONSET,
            mass_loading=eta_func,
            dr=ZONE_WIDTH
        )
        # Run one-zone model
        sz = vice.singlezone(name=name,
                             func=ifr,
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = tau_star
        sz.eta = eta_func(radius)
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name,
                          fig=fig, axs=axs,
                          linestyle='-',
                        #   color=color_list[i],
                          label='Me (R=%s kpc)' % radius,
                        #   marker_labels=(i == 1),
                          markers=[0.3, 1, 3, 10])
        # Thick-to-thin ratio
        print(thick_to_thin_ratio(radius))
        hist = vice.history(name)
        onset_idx = int(ifr.onset / dt)
        thick_disk_mass = hist['mstar'][onset_idx-1]
        thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
        print(thick_disk_mass / thin_disk_mass)
    
    # Spitoni's parameters
    for i, radius in enumerate(RADII):
        name = str(output_dir / ('spitoni_%skpc' % int(radius)))
        area = np.pi * ((radius + ZONE_WIDTH/2)**2 - (radius - ZONE_WIDTH/2)**2)
        tau_star = twoinfall_sf_law(area, onset=S21_TMAX[i], 
                                    sfe1=S21_SFE1[i], sfe2=S21_SFE2[i])
        # Initialize infall rate
        ifr = twoinfall_onezone(
            radius, 
            first_timescale=S21_TAU1[i],
            second_timescale=S21_TAU2[i], 
            onset=S21_TMAX[i],
            mass_loading=outflows.no_outflows,
            dr=ZONE_WIDTH,
            sfe1=tau_star.sfe1,
            sfe2=tau_star.sfe2,
        )
        # print(ifr.ratio)
        # ifr.ratio = ifr.ampratio(radius, thick_to_thin_ratio, 
        #     eta = 0., vgas = 0., dt = dt
        # )
        # print(ifr.ratio)
        # # Normalize infall rate
        # prefactor = normalize_ifrmode(
        #     ifr, gradient, ifr.tau_star, radius, 
        #     eta = 0., vgas = 0., dt = dt, dr = ZONE_WIDTH, recycling = 0.4
        # )
        # ifr.first.norm *= prefactor
        # ifr.second.norm *= prefactor
        # area = np.pi * ((radius + ZONE_WIDTH/2)**2 - (radius - ZONE_WIDTH/2)**2)
        # ifr.first.norm *= area * gradient(radius)
        # ifr.second.norm *= area * gradient(radius)
        # Run one-zone model
        sz = vice.singlezone(name=name,
                             func=ifr,
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = tau_star
        sz.eta = 0.
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name,
                          fig=fig, axs=axs,
                          linestyle='--',
                        #   color=color_list[i],
                          label='Spitoni (R=%s kpc)' % radius,
                        #   marker_labels=(i == 1),
                          markers=[0.3, 1, 3, 10])
        # Thick-to-thin ratio
        print(thick_to_thin_ratio(radius))
        hist = vice.history(name)
        onset_idx = int(ifr.onset / dt)
        thick_disk_mass = hist['mstar'][onset_idx-1]
        thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
        print(thick_disk_mass / thin_disk_mass)

    # Adjust axis limits
    axs[1].set_ylim(bottom=0)
    axs[2].set_xlim(left=0)
    axs[0].legend(frameon=False, loc='lower left')
    
    fig.savefig(paths.figures / 'spitoni_comparison', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
