"""
This script plots the radial metallicity gradient for models with and without
radial gas flows.
"""

import numpy as np
import matplotlib.pyplot as plt
import vice
import paths
import _globals

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    
    fig, axs = plt.subplots(3, 1, figsize=(_globals.ONE_COLUMN_WIDTH, 
                                           2 * _globals.ONE_COLUMN_WIDTH),
                            tight_layout=True)
                            
    # plot_radial_gradient('no_outflow/diskmodel', axs, 
    #                      label=r'0 km/s | $\eta=0$',
    #                      color='k', linestyle='-')
    # plot_radial_gradient('gasflow_1kms_no_outflow/diskmodel', axs,
    #                      label=r'1 km/s | $\eta=0$',
    #                      color='r', linestyle='-')
    # plot_radial_gradient('gasflow_1kms_maxsf_13kpc/diskmodel', axs,
    #                      label=r'1 km/s | $\eta=0$ | $R_{\rm max}=13$ kpc',
    #                      color='r', linestyle=':')
    # plot_radial_gradient('gasflow_2kms_no_outflow/diskmodel', axs,
    #                      label=r'2 km/s | $\eta=0$',
    #                      color='b', linestyle='-')
    # plot_radial_gradient('gaussian/diskmodel', axs,
    #                      label=r'0 km/s | $\eta\propto e^R$',
    #                      color='k', linestyle='--')
    # plot_radial_gradient('gasflow_1kms/diskmodel', axs,
    #                      label=r'1 km/s | $\eta\propto e^R$',
    #                      color='r', linestyle='--')
    # plot_radial_gradient('gasflow_2kms/diskmodel', axs,
    #                      label=r'2 km/s | $\eta\propto e^R$',
    #                      color='b', linestyle='--')
    plot_radial_gradient('gaussian/no_outflow/gasflow_in_1kms/J21/insideout/diskmodel',
                         axs, label='SFR mode')
    plot_radial_gradient('gaussian/no_outflow/gasflow_in_1kms/J21/twoinfall/diskmodel',
                         axs, label='IFR mode')
    
    # mout = vice.output(str(paths.multizone / 'gasflow_1kms_maxsf_13kpc/diskmodel'))
    # axs[0].plot(radial_gradient(mout, '[o/h]'), 'r:', label='1 km/s | $\eta=0$')
    # axs[1].plot(radial_gradient(mout, '[fe/h]'), 'r:')
    # axs[2].plot(radial_gradient(mout, '[o/fe]'), 'r:')
    
    axs[0].set_ylabel('[O/H]')
    axs[1].set_ylabel('[Fe/H]')
    axs[2].set_ylabel('[O/Fe]')
    axs[2].set_xlabel('Radius [kpc]')
    
    axs[0].legend()
    
    fig.savefig(paths.figures / 'gasflows')
    plt.close()


def plot_radial_gradient(name, axs, label='', color=None, linestyle='-'):
    mout = vice.output(str(paths.multizone / name))
    xarr = np.arange(0, _globals.MAX_SF_RADIUS, _globals.ZONE_WIDTH)
    axs[0].plot(xarr, radial_gradient(mout, '[o/h]'), 
                marker=None, linestyle=linestyle, color=color,
                label=label)
    axs[1].plot(xarr, radial_gradient(mout, '[fe/h]'), 
                marker=None, linestyle=linestyle, color=color)
    axs[2].plot(xarr, radial_gradient(mout, '[o/fe]'), 
                marker=None, linestyle=linestyle, color=color)


def radial_gradient(multioutput, parameter, index=-1, 
                    Rmax=_globals.MAX_SF_RADIUS,
                    zone_width=_globals.ZONE_WIDTH):
    """
    Return the value of the given model parameter at all zones.
    
    Parameters
    ----------
    multioutput : vice.multioutput
        VICE multi-zone output instance for the desired model.
    parameter : str
        Name of parameter in vice.history dataframe.
    index : int, optional
        Index to select for each zone. The default is -1, which corresponds
        to the last simulation timestep or the present day.
    Rmax : float, optional
        Maximum radius in kpc. The default is 15.5.
    zone_width : float, optional
        Annular zone width in kpc. The default is 0.1.
        
    Returns
    -------
    list
        Parameter values at each zone at the given time index.
    """
    return [multioutput.zones['zone%i' % z].history[index][parameter] 
            for z in range(int(Rmax/zone_width))]


if __name__ == '__main__':
    main()
