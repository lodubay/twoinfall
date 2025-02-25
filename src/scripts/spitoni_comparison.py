"""
This script plots one-zone model abundance tracks with my parameters
versus parameters from Spitoni et al. (2021), as closely as I can replicate.
"""

import numpy as np
import matplotlib.pyplot as plt
import vice

from apogee_sample import APOGEESample
from multizone.src.yields import yZ1
from multizone.src.models import twoinfall_sf_law, twoinfall_expvar
from multizone.src import outflows
from multizone.src.models.gradient import thick_to_thin_ratio
from track_and_mdf import setup_axes, plot_vice_onezone
from colormaps import paultol
import paths
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH
from utils import twoinfall_onezone, get_bin_centers

XLIM = (-1.8, 0.8)
YLIM = (-0.22, 0.499)
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
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(5, 16, wspace=0.)
    subfigs = [
        fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 6, 11), (6, 5, 5))
    ]
    axs_titles = ['(a)', '(b)', '(c)']
    axs_list = [
        setup_axes(
            subfigs[i], xlim=XLIM, ylim=YLIM, show_ylabel=(i==0), 
            title=axs_titles[i],
        ) for i in range(3)
    ]
    colors = list(reversed(paultol.highcontrast.colors))

    output_dir = paths.data / 'onezone' / 'spitoni_comparison'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)

    # Plot APOGEE distributions
    for i, radius in enumerate(RADII):
        axs = axs_list[i]
        axs[0].text(
            0.95, 0.95, r'$R_{\rm gal}=%s$ kpc' % radius,
            ha='right', va='top', transform=axs[0].transAxes, size=plt.rcParams['axes.labelsize'])
        apogee_data = APOGEESample.load()
        apogee_region = apogee_data.region(
            galr_lim=(radius - ZONE_WIDTH/2, radius + ZONE_WIDTH/2), 
            absz_lim=(0, 2)
        )
        pcm = axs[0].hexbin(
            apogee_region('FE_H'), apogee_region('O_FE'),
            gridsize=50, extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
            cmap='binary', linewidths=0.2
        )
        cax = axs[0].inset_axes([0.05, 0.05, 0.05, 0.75])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
        if i == 0:
            cbar.set_label('# APOGEE stars')
        
        # APOGEE abundance distributions
        feh_df, bin_edges = apogee_region.mdf(
            col='FE_H', range=XLIM, smoothing=0.2
        )
        axs[1].plot(
            get_bin_centers(bin_edges), feh_df / max(feh_df), 
            color='gray', linestyle='-', marker=None
        )
        ofe_df, bin_edges = apogee_region.mdf(
            col='O_FE', range=YLIM, smoothing=0.05
        )
        axs[2].plot(
            ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
            color='gray', linestyle='-', marker=None
        )

    # My parameters
    eta_func = outflows.yZ1()
    for i, radius in enumerate(RADII):
        axs = axs_list[i]
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
                          color=colors[i],
                          label='This work',
                          marker_labels=False,
                          markers=[0.3, 1, 3, 6, 10])
        # Thick-to-thin ratio
        # print(thick_to_thin_ratio(radius))
        # hist = vice.history(name)
        # onset_idx = int(ifr.onset / dt)
        # thick_disk_mass = hist['mstar'][onset_idx-1]
        # thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
        # print(thick_disk_mass / thin_disk_mass)
    
    # Spitoni's parameters
    custom_thick_thin_ratio = vice.toolkit.interpolation.interp_scheme_1d(
        RADII, S21_THICK_THIN_RATIO
    )
    for i, radius in enumerate(RADII):
        axs = axs_list[i]
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
            disk_ratio=custom_thick_thin_ratio,
        )
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
                          color=colors[i],
                          label='Spitoni et al. (2021)',
                          marker_labels=False,
                          markers=[0.3, 1, 3, 6, 10])
        # Thick-to-thin ratio
        # print(custom_thick_thin_ratio(radius))
        # hist = vice.history(name)
        # onset_idx = int(ifr.onset / dt)
        # thick_disk_mass = hist['mstar'][onset_idx-1]
        # thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
        # print(thick_disk_mass / thin_disk_mass)

    # Adjust axis limits
    axs[1].set_ylim(bottom=0)
    axs[2].set_xlim(left=0)
    axs[0].legend(frameon=False, loc='lower right', handletextpad=0.5)
    plt.subplots_adjust(
        bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5
    )
    
    fig.savefig(paths.figures / 'spitoni_comparison', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
