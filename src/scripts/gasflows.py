"""
This script plots the radial metallicity gradient for models with and without
radial gas flows.
"""

import numpy as np
import matplotlib.pyplot as plt
import vice
import paths
import _globals
from utils import radial_gradient

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    
    fig, axs = plt.subplots(3, 1, figsize=(_globals.ONE_COLUMN_WIDTH, 
                                           2 * _globals.ONE_COLUMN_WIDTH),
                            tight_layout=True)
    
    # Gradients
    # rlist = [i*0.1 for i in range(155)]
    # axs[0].plot(rlist, [-0.08 * (r - 8) for r in rlist], 'k--')
                            
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
    plot_radial_gradient('nomigration/no_outflow/gasflow_in_1kms/J21/static/diskmodel',
                         axs, label='Static SFR mode (gas flows, no outflow)')
    plot_radial_gradient('nomigration/no_outflow/gasflow_in_1kms/J21/static_infall/diskmodel',
                         axs, label='Static IFR mode (gas flows, no outflow')
    # plot_radial_gradient('nomigration/no_outflow/gasflow_in_1kms/J21/twoinfall/diskmodel',
    #                      axs, label='Two-Infall')
    plot_radial_gradient('nomigration/outflow/no_gasflow/J21/twoinfall/diskmodel',
                         axs, linestyle='--', label='Two-Infall (outflows, no gas flow)')
    
    # mout = vice.output(str(paths.multizone / 'gasflow_1kms_maxsf_13kpc/diskmodel'))
    # axs[0].plot(radial_gradient(mout, '[o/h]'), 'r:', label='1 km/s | $\eta=0$')
    # axs[1].plot(radial_gradient(mout, '[fe/h]'), 'r:')
    # axs[2].plot(radial_gradient(mout, '[o/fe]'), 'r:')
    
    axs[0].set_ylabel('[O/H]')
    axs[1].set_ylabel('[Fe/H]')
    axs[2].set_ylabel('[O/Fe]')
    axs[2].set_xlabel('Radius [kpc]')
    
    axs[0].set_ylim((-1, 1))
    axs[1].set_ylim((-1, 1))
    
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


if __name__ == '__main__':
    main()
