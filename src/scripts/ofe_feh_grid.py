"""
Plot a grid of [O/Fe] vs [Fe/H] at varying Galactic radii and z-heights.
"""

# import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
import numpy as np
import vice
from multizone_stars import MultizoneStars
from apogee_tools import import_apogee, gen_kde
# from mwm_tools import import_mwm
from scatter_plot_grid import setup_axes, setup_colorbar
# from utils import kde2D
from _globals import GALR_BINS, ABSZ_BINS, ZONE_WIDTH, MAX_SF_RADIUS
import paths

FEH_LIM = (-1.3, 0.7)
OFE_LIM = (-0.15, 0.65)

def main(output_name, uncertainties=True, **kwargs):
    # Import APOGEE data
    apogee_data = import_apogee()
    # Import multioutput stars data
    mzs = MultizoneStars.from_output(output_name)
    # Model observational uncertainties
    if uncertainties:
        mzs.model_uncertainty(inplace=True)
    plot_ofe_feh_grid(mzs, apogee_data, **kwargs)


def plot_ofe_feh_grid(mzs, apogee_data, tracks=True, apogee_contours=True,
                      style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_axes(xlim=FEH_LIM, ylim=OFE_LIM, xlabel='[Fe/H]', 
                          ylabel='[O/Fe]', row_label_pos=(0.07, 0.85),
                          title=mzs.name, width=8, galr_bins=GALR_BINS)
    cbar = setup_colorbar(fig, cmap=cmap, vmin=0, vmax=MAX_SF_RADIUS, 
                          label=r'Birth $R_{\rm{Gal}}$ [kpc]')
    cbar.ax.yaxis.set_major_locator(MultipleLocator(2))
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ism_track_color = 'k'
    ism_track_width = 0.5
        
    for i, row in enumerate(axs):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            subset = mzs.region(galr_lim, absz_lim)
            subset.scatter_plot(ax, '[fe/h]', '[o/fe]', color='galr_origin',
                                cmap=cmap, norm=cbar.norm)
            if tracks:
                zone = int(0.5 * (galr_lim[0] + galr_lim[1]) / ZONE_WIDTH)
                zone_path = str(mzs.fullpath / ('zone%d' % zone))
                hist = vice.history(zone_path)
                ax.plot(hist['[fe/h]'], hist['[o/fe]'], c=ism_track_color, ls='-', 
                        linewidth=ism_track_width)
            if apogee_contours:
                xx, yy, logz = gen_kde(apogee_data, bandwidth=0.02,
                                       galr_lim=galr_lim, absz_lim=absz_lim)
                # scale the linear density to the max value
                scaled_density = np.exp(logz) / np.max(np.exp(logz))
                # contour levels at 1, 2, and 3 sigma
                levels = np.exp(-0.5 * np.array([2, 1])**2)
                ax.contour(xx, yy, scaled_density, levels, colors='r',
                           linewidths=0.5, linestyles=['--', '-'])
    
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    # Custom legend
    custom_lines = [Line2D([0], [0], color=ism_track_color, linestyle='-', 
                           linewidth=ism_track_width),
                    Line2D([0], [0], color='r', linestyle='-', linewidth=0.5),
                    Line2D([0], [0], color='r', linestyle='--', linewidth=0.5)]
    legend_labels = ['Gas abundance', 'APOGEE 30% cont.', 'APOGEE 80% cont.']
    axs[2,-1].legend(custom_lines, legend_labels, frameon=False, 
                     loc='upper right', handlelength=0.6, handletextpad=0.4)
    
    # Save
    fname = mzs.name.replace('diskmodel', 'ofe_feh_grid.png')
    fullpath = paths.extra / fname
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ofe_feh_grid.py',
        description='Generate a grid of [O/Fe] vs [Fe/H] scatterplots ' + \
            'from a VICE multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output')
    parser.add_argument('-u', '--uncertainties', action='store_true',
                        help='Model APOGEE uncertainties in VICE output')
    parser.add_argument('-t', '--tracks', action='store_true',
                        help='Plot ISM tracks in addition to stellar abundances')
    parser.add_argument('-a', '--apogee-contours', action='store_true',
                        help='Plot contour lines from APOGEE data')
    parser.add_argument('--cmap', metavar='COLORMAP', type=str,
                        default='winter_r',
                        help='Name of colormap for color-coding VICE ' + \
                             'output (default: winter_r)')
    parser.add_argument('-s', '--style', 
                        choices=['paper', 'poster'],
                        default='paper', 
                        help='Plot style to use (default: paper)')
    args = parser.parse_args()
    main(**vars(args))
