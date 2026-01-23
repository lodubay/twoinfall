"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import vice

from apogee_sample import APOGEESample
from utils import twoinfall_onezone, get_bin_centers
from multizone.src import models, outflows
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH, ZONE_WIDTH
from track_and_mdf import setup_axes, plot_vice_onezone
import paths

RADIUS = 8.
LOCAL_DISK_RATIO = 0.12 # local thick-to-thin disk mass ratio
# Fiducial parameter values
FIDUCIAL = {
    'first_timescale': 0.3,
    'second_timescale': 15.,
    'onset': 3.2
}
# Full range of parameter values to explore
ALTERNATIVES = {
    'first_timescale': [0.1, 0.3, 1, 3],
    'second_timescale': [5, 10, 15, 30],
    'onset': [1.2, 2.2, 3.2, 4.2]
}
# convert between parameter keyword names and fancy labels
LABELS = {
    'first_timescale': '\\tau_1',
    'second_timescale': '\\tau_2',
    'onset': 't_{\\rm max}'
}
XLIM = (-1.6, 0.7)
YLIM = (-0.12, 0.499)
GRIDSIZE = 30
SMOOTH_WIDTH = 0.05

def main(verbose=False, style='paper'):
    # Set up figure and subfigures
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.72*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(15, 22, wspace=0., hspace=0.)

    # First row of sub-figures: y/Zsun=1 yield scale
    if verbose:
        print('\ny/Zsun=1 yields')
    from multizone.src.yields import yZ1
    eta = outflows.yZ1(RADIUS)
    subfigs1 = [
        fig.add_subfigure(gs[:7,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))
    ]
    # First panel: vary tau_1
    if verbose:
        print('\nFirst timescale')
    f1axs0 = vary_param(
        subfigs1[0], 
        first_timescale=ALTERNATIVES['first_timescale'], 
        second_timescale=FIDUCIAL['second_timescale'], 
        onset=FIDUCIAL['onset'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        label_index=None, 
        cmap_name='autumn', 
        title='(a)',
        verbose=verbose
    )
    # Second panel: vary tau_2
    if verbose:
        print('\nSecond timescale')
    f1axs1 = vary_param(
        subfigs1[1], 
        second_timescale=ALTERNATIVES['second_timescale'],
        first_timescale=FIDUCIAL['first_timescale'], 
        onset=FIDUCIAL['onset'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        show_ylabel=False,
        label_index=0, 
        cmap_name='summer', 
        title='(b)',
        verbose=verbose
    )
    # Third panel: vary t_on
    if verbose:
        print('\nOnset time')
    f1axs2 = vary_param(
        subfigs1[2], 
        onset=ALTERNATIVES['onset'],
        first_timescale=FIDUCIAL['first_timescale'], 
        second_timescale=FIDUCIAL['second_timescale'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        show_ylabel=False,
        label_index=None, 
        cmap_name='winter', 
        title='(c)',
        verbose=verbose
    )
    subfigs1[1].suptitle(r'$y/Z_\odot=1$', y=1, va='bottom')

    # Second row of sub-figures: y/Zsun=2 yield scale
    if verbose:
        print('\ny/Zsun=2 yields')
    from multizone.src.yields import yZ2
    eta = outflows.yZ2(RADIUS)
    subfigs2 = [
        fig.add_subfigure(gs[8:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))
    ]
    FIDUCIAL['onset'] = 2.2
    # First panel: vary tau_1
    if verbose:
        print('\nFirst timescale')
    f2axs0 = vary_param(
        subfigs2[0], 
        first_timescale=ALTERNATIVES['first_timescale'], 
        second_timescale=FIDUCIAL['second_timescale'], 
        onset=FIDUCIAL['onset'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        label_index=None, 
        cmap_name='autumn', 
        title='(d)',
        verbose=verbose
    )
    # Second panel: vary tau_2
    if verbose:
        print('\nSecond timescale')
    f2axs1 = vary_param(
        subfigs2[1], 
        second_timescale=ALTERNATIVES['second_timescale'],
        first_timescale=FIDUCIAL['first_timescale'], 
        onset=FIDUCIAL['onset'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        show_ylabel=False,
        label_index=0, 
        cmap_name='summer', 
        title='(e)',
        verbose=verbose
    )
    # Third panel: vary t_on
    if verbose:
        print('\nOnset time')
    f2axs2 = vary_param(
        subfigs2[2], 
        onset=ALTERNATIVES['onset'],
        first_timescale=FIDUCIAL['first_timescale'], 
        second_timescale=FIDUCIAL['second_timescale'],
        eta=eta,
        xlim=XLIM, ylim=YLIM, 
        show_ylabel=False,
        label_index=None, 
        cmap_name='winter', 
        title='(f)',
        verbose=verbose
    )
    subfigs2[1].suptitle(r'$y/Z_\odot=2$', y=1, va='bottom')

    # Plot APOGEE data in each panel
    apogee_sample = APOGEESample.load()
    apogee_solar = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 2))
    cmap_name = 'binary'
    data_color = '0.6'
    for axs in [f1axs0, f1axs1, f1axs2, f2axs0, f2axs1, f2axs2]:
        pcm = axs[0].hexbin(
            apogee_solar('FE_H'), apogee_solar('O_FE'),
            gridsize=GRIDSIZE,
            extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
            cmap=cmap_name, linewidths=0.2, zorder=0
        )
        # Plot APOGEE abundance distributions in marginal panels
        feh_df, feh_bin_edges = apogee_solar.mdf(
            col='FE_H', range=XLIM, smoothing=SMOOTH_WIDTH
        )
        axs[1].plot(
            get_bin_centers(feh_bin_edges), feh_df / max(feh_df), 
            color=data_color, linestyle='-', linewidth=2, marker=None, zorder=0
        )
        ofe_df, ofe_bin_edges = apogee_solar.mdf(
            col='O_FE', range=YLIM, smoothing=SMOOTH_WIDTH
        )
        axs[2].plot(
            ofe_df / max(ofe_df), get_bin_centers(ofe_bin_edges),
            color=data_color, linestyle='-', linewidth=2, marker=None, zorder=0
        )
    
    plt.subplots_adjust(
        bottom=0.13, top=0.98, left=0.16, right=0.98
    )
    fig.savefig(paths.figures / 'infall_parameters')
    plt.close()


def vary_param(subfig, first_timescale=1., second_timescale=10., onset=4.,
               eta=0.2, local_disk_ratio=LOCAL_DISK_RATIO, label_index=None, 
               cmap_name=None, verbose=False, **kwargs):
    """
    Plot a series of onezone model outputs, varying one parameter of the 
    two-infall model while holding the others fixed.
    
    Parameters
    ----------
    subfig : matplotlib.figure.Figure
        Figure or subfigure in which to generate the axes.
    first_timescale : float, optional
        Timescale of the first infall in Gyr. If a list is passed, assumes
        this is the variable parameter and the others should be held fixed.
        The default is 0.1.
    second_timescale : float, optional
        Timescale of the second infall in Gyr. The default is 3.
    onset : float, optional
        Onset time of the second infall in Gyr. The default is 3.
    eta : float, optional
        Outflow mass-loading factor. The default is 0.2.
    label_index : int, optional
        Index of track to add time labels to. If None, no time labels are added.
        The default is None.
    cmap_name : str, optional
        Name of colormap to draw line colors from. If None, line colors are
        drawn from the default prop color cycle. The default is None.
    verbose : bool, optional
        Whether to print verbose output to terminal.
    **kwargs passed to track_and_mdf.setup_axes()
    """
    param_dict = {
        'first_timescale': first_timescale, 
        'second_timescale': second_timescale, 
        'onset': onset
    }
    var = None
    other_params = ''
    for i, param in enumerate(param_dict.keys()):
        value = param_dict[param]
        if isinstance(value, list):
            if var is not None:
                raise ValueError('Too many variable parameters.')
            values = value
            var = param
        else:
            other_params += '$%s=%s$ Gyr\n' % (LABELS[param], value)
    if var is None:
        raise ValueError('Please specify one variable parameter.')
    axs = setup_axes(subfig, xlabel='[Fe/H]', **kwargs)

    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    # Prescription for disk surface density as a function of radius
    diskmodel = models.diskmodel.two_component_disk.from_local_ratio(
        local_ratio = local_disk_ratio
    )

    for i, val in enumerate(values):
        param_dict[var] = val
        # Line color
        if val == FIDUCIAL[var]:
            color = 'k'
        elif cmap_name is not None:
            cmap = plt.get_cmap(cmap_name)
            color = cmap((i+0.5) / len(values))
        else:
            color = None
        # Run one-zone model
        name = output_name(*param_dict.values())
        ifr = twoinfall_onezone(
            RADIUS, 
            diskmodel=diskmodel,
            mass_loading=eta, 
            dt=dt, 
            dr=ZONE_WIDTH, 
            **param_dict
        )
        sz = vice.singlezone(name=name,
                             func=ifr, 
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = models.twoinfall_sf_law(area, onset=param_dict['onset'])
        sz.eta = eta
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name, 
                          xcol='[fe/h]',
                          fig=subfig, axs=axs, 
                          linestyle='-', 
                          color=color, 
                          label=f'{val:.1f}', 
                          marker_labels=(i==label_index),
                          markers=[0.3, 1, 3, 6, 10])
        if verbose:
            print('Value:', val)
            print('Eta:', eta)
            # Thick-to-thin ratio
            hist = vice.history(name)
            onset_idx = int(ifr.onset / dt)
            thick_disk_mass = hist['mstar'][onset_idx-1]
            thin_disk_mass = hist['mstar'][-1] - thick_disk_mass
            print('Thick-to-thin ratio:', thick_disk_mass / thin_disk_mass)

    # Adjust axis limits
    axs[1].set_ylim(bottom=0)
    axs[2].set_xlim(left=0)
    axs[0].legend(frameon=False, loc='lower left', 
                  title='$%s$ [Gyr]' % LABELS[var])
    
    # Label other param values
    axs[0].text(0.95, 0.95, other_params, ha='right', va='top',
                transform=axs[0].transAxes)

    return axs


def output_name(tau1, tau2, onset, parent_dir=paths.data/'onezone'/'params'):
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
    name = f'first{int(tau1*10):02d}_second{int(tau2)}_onset{int(onset)}'
    return str(parent_dir / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='infall_parameters.py',
        description='Plot the effect of different infall timescales and onset \
time in one-zone models.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '--style', 
        type=str,
        default='paper',
        choices=['paper', 'poster', 'presentation'],
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
