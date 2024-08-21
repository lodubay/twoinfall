"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vice
from vice.toolkit import J21_sf_law
import paths
# from multizone.src.yields import J21
# from vice.yields.presets import JW20
from multizone.src.yields import W23
from multizone.src import models, dtds
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH, ZONE_WIDTH
from colormaps import paultol
from track_and_mdf import setup_axes, plot_vice_onezone

RADIUS = 8.
PARAM_DEFAULTS = {
    'first_timescale': 0.1,
    'second_timescale': 3,
    'onset': 3
}
# convert between parameter keyword names and fancy labels
LABELS = {
    'first_timescale': '\\tau_1',
    'second_timescale': '\\tau_2',
    'onset': 't_{\\rm on}'
}
XLIM = (-2, 0.8)
YLIM = (-0.16, 0.54)

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(7, 22, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))]
    print('First timescale')
    axs0 = vary_param(subfigs[0], first_timescale=[0.1, 0.3, 1, 3],
                      second_timescale=4, onset=3,
                      xlim=XLIM, ylim=YLIM)
    print('Second timescale')
    axs1 = vary_param(subfigs[1], second_timescale=[1, 3, 10, 30],
                      first_timescale=1, onset=3,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=1)
    print('Onset time')
    axs2 = vary_param(subfigs[2], onset=[1, 2, 3, 4],
                      first_timescale=1, second_timescale=4,
                      xlim=XLIM, ylim=YLIM, ylabel=False,
                      label_index=2)
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5)
    fig.savefig(paths.figures / 'onezone_params', dpi=300)
    plt.close()


def vary_param(subfig, first_timescale=0.1, second_timescale=3, onset=3,
               label_index=0, **kwargs):
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
    
    # Star formation law
    # Single power-law with k=1.5 and high-mass cutoff
    area = np.pi * ((RADIUS + ZONE_WIDTH)**2 - RADIUS**2)
    sf_law = J21_sf_law(area, mode='ifr', index1=1.5, index2=1.5, Sigma_g2=1e8)
    
    dtd = dtds.plateau(width=1, tmin=ONEZONE_DEFAULTS['delay'])

    for i, val in tqdm(enumerate(values)):
        param_dict[var] = val
        # Run one-zone model
        name = output_name(*param_dict.values())
        # Note: the amplitude ratio calculation assumes the J21 star formation
        # law, but this should only affect it at the 1% level
        ifr = models.twoinfall(RADIUS, dt=dt, **param_dict)
        sz = vice.singlezone(name=name,
                             RIa=dtd,
                             func=ifr, 
                             mode='ifr',
                             **ONEZONE_DEFAULTS)
        sz.tau_star = sf_law
        # sz.schmidt = True
        # sz.schmidt_index = 0.5
        # sz.MgSchmidt = 1e8
        sz.run(simtime, overwrite=True)
        plot_vice_onezone(name, 
                          fig=subfig, axs=axs, 
                          linestyle='-', 
                          color=None, 
                          label=f'{val:.1f}', 
                          marker_labels=(i==label_index))

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
    main()
