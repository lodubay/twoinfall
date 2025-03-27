"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone.src.yields import yZ1
from utils import twoinfall_onezone
from multizone.src import models, outflows
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH
from track_and_mdf import setup_axes, plot_vice_onezone
import paths

RADIUS = 8.
ZONE_WIDTH = 2.
FIDUCIAL = {
    'first_timescale': 1.,
    'second_timescale': 10.,
    'onset': 4.
}
# convert between parameter keyword names and fancy labels
LABELS = {
    'first_timescale': '\\tau_1',
    'second_timescale': '\\tau_2',
    'onset': 't_{\\rm max}'
}
XLIM = (-1.9, 0.7)
YLIM = (-0.14, 0.499)

def main(fiducial=FIDUCIAL, xlim=XLIM, ylim=YLIM, fname='onezone_params', 
         verbose=False, style='paper'):
    # Set up figure and subfigures
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(5, 16, wspace=0.)
    subfigs = [
        fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 6, 11), (6, 5, 5))
    ]
    # First panel: vary tau_1
    if verbose:
        print('\nFirst timescale')
    axs0 = vary_param(
        subfigs[0], 
        first_timescale=[0.1, 0.3, 1, 3], 
        second_timescale=fiducial['second_timescale'], 
        onset=fiducial['onset'],
        xlim=xlim, ylim=ylim, 
        label_index=None, 
        cmap_name='autumn', 
        title='(a)',
        verbose=verbose
    )
    # Second panel: vary tau_2
    if verbose:
        print('\nSecond timescale')
    axs1 = vary_param(
        subfigs[1], 
        second_timescale=[3, 5, 10, 30],
        first_timescale=fiducial['first_timescale'], 
        onset=fiducial['onset'],
        xlim=xlim, ylim=ylim, 
        show_ylabel=False,
        label_index=0, 
        cmap_name='summer', 
        title='(b)',
        verbose=verbose
    )
    # Third panel: vary t_on
    if verbose:
        print('\nOnset time')
    axs2 = vary_param(
        subfigs[2], 
        onset=[1, 2, 3, 4, 5],
        first_timescale=fiducial['first_timescale'], 
        second_timescale=fiducial['second_timescale'],
        xlim=xlim, ylim=ylim, 
        show_ylabel=False,
        label_index=None, 
        cmap_name='winter', 
        title='(c)',
        verbose=verbose
    )
    plt.subplots_adjust(
        bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5
    )
    fig.savefig(paths.figures / fname, dpi=300)
    plt.close()


def vary_param(subfig, first_timescale=1., second_timescale=10., onset=4.,
               label_index=None, cmap_name=None, verbose=False, **kwargs):
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
    area = np.pi * ((RADIUS + ZONE_WIDTH/2)**2 - (RADIUS - ZONE_WIDTH/2)**2)

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
        # Outflow mass-loading factor
        eta_func = outflows.yZ1
        eta = eta_func(RADIUS)
        # Run one-zone model
        name = output_name(*param_dict.values())
        ifr = twoinfall_onezone(RADIUS, mass_loading=eta_func, dt=dt, 
                                 dr=ZONE_WIDTH, **param_dict)
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
        prog='onezone_params.py',
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
