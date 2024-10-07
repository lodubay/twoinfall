"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vice
# from vice.toolkit import J21_sf_law
from multizone.src.yields import J21
# from vice.yields.presets import JW20
# from multizone.src.yields import W23
# vice.yields.sneia.settings['fe'] *= (1.1/1.2)
# from multizone.src.yields import F04
from multizone.src import models
from multizone.src.models.gradient import gradient
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH, ZONE_WIDTH
from colormaps import paultol
from track_and_mdf import setup_axes, plot_vice_onezone
import paths

RADIUS = 8.
PARAM_DEFAULTS = {
    'first_timescale': 1.,
    'second_timescale': 10.,
    'onset': 4.
}
# convert between parameter keyword names and fancy labels
LABELS = {
    'first_timescale': '\\tau_1',
    'second_timescale': '\\tau_2',
    'onset': 't_{\\rm on}'
}
XLIM = (-1.7, 0.8)
YLIM = (-0.18, 0.54)

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(7, 22, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))]
    print('\nFirst timescale')
    axs0 = vary_param(subfigs[0], first_timescale=[0.1, 0.3, 1, 3],
                      second_timescale=10, onset=4,
                      xlim=XLIM, ylim=YLIM, label_index=0, verbose=True)
    print('\nSecond timescale')
    axs1 = vary_param(subfigs[1], second_timescale=[3, 5, 10, 30],
                      first_timescale=1, onset=4,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=1, verbose=True)
    print('\nOnset time')
    axs2 = vary_param(subfigs[2], onset=[1, 2, 3, 4, 5],
                      first_timescale=1, second_timescale=10,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=2, verbose=True)
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5)
    fig.savefig(paths.figures / 'onezone_params', dpi=300)
    plt.close()


def vary_param(subfig, first_timescale=0.1, second_timescale=3, onset=3,
               label_index=0, verbose=False, **kwargs):
    """
    Plot a series of onezone model outputs, varying one parameter of the 
    two-infall model while holding the others fixed.
    
    Parameters
    ----------
    subfig : matplotlib.figure.Figure
    first_timescale : float, optional
        Timescale of the first infall in Gyr. If a list is passed, assumes
        this is the variable parameter and the others should be held fixed.
        The default is 0.1.
    second_timescale : float, optional
        Timescale of the second infall in Gyr. The default is 3.
    onset : float, optional
        Onset time of the second infall in Gyr. The default is 3.
    label_index : int, optional
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
    multiline_title = '$\\tau_2=%s$ Gyr,' % second_timescale + '\n' + \
        '$t_{\\rm on}=%s$ Gyr' % onset
    axs = setup_axes(subfig, title='', **kwargs)

    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)

    for i, val in enumerate(values):
        param_dict[var] = val
        # Outflow mass-loading factor
        # eta_func = models.equilibrium_mass_loading(
        #     alpha_h_eq=0.2, 
        #     tau_sfh=param_dict['second_timescale'], 
        #     tau_star=2.
        # )
        eta_func = vice.milkyway.default_mass_loading
        eta = eta_func(RADIUS)
        # Run one-zone model
        name = output_name(*param_dict.values())
        ifr = twoinfall_gradient(RADIUS, mass_loading=eta_func, dt=dt, 
                                 dr=ZONE_WIDTH, **param_dict)
        sz = vice.singlezone(name=name,
                             func=ifr, 
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = models.twoinfall_sf_law(area, onset=param_dict['onset'])
        sz.eta = eta
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name, 
                          fig=subfig, axs=axs, 
                          linestyle='-', 
                          color=None, 
                          label=f'{val:.1f}', 
                          marker_labels=(i==label_index),
                          markers=[0.3, 1, 3, 10])
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


class twoinfall_gradient(models.twoinfall):
    def __init__(self, radius, **kwargs):
        super().__init__(radius, **kwargs)
        self.first.norm *= gradient(radius)
        self.second.norm *= gradient(radius)


def output_name(tau1, tau2, onset, parent_dir=paths.data/'onezone'/'params'):
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
    name = f'first{int(tau1*10):02d}_second{int(tau2)}_onset{int(onset)}'
    return str(parent_dir / name)


if __name__ == '__main__':
    main()
