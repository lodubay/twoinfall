"""
This script plots the radial metallicity gradient for models with and without
radial gas flows.
"""

import numpy as np
import matplotlib.pyplot as plt
# from multizone_stars import MultizoneStars
import vice
import paths
import _globals
from utils import get_bin_centers

def main(style='paper'):
    # mzs_flows = vice.output(str(paths.simulation_outputs / 'gasflow/diskmodel'))
    # mzs_noflows = vice.output(str(paths.simulation_outputs / 'gaussian/diskmodel'))
    plt.style.use(paths.styles / f'{style}.mplstyle')
    
    fig, axs = plt.subplots(3, 1, figsize=(_globals.ONE_COLUMN_WIDTH, 
                                           2 * _globals.ONE_COLUMN_WIDTH),
                            tight_layout=True)
    
    mout_flow = vice.output(str(paths.simulation_outputs / 'gasflow/diskmodel'))
    axs[0].plot(radial_gradient(mout_flow, '[o/h]'), 'r-', label='Radial flows')
    axs[1].plot(radial_gradient(mout_flow, '[fe/h]'), 'r-')
    axs[2].plot(radial_gradient(mout_flow, '[o/fe]'), 'r-')
    
    mout_noflow = vice.output(str(paths.simulation_outputs / 'gaussian/diskmodel'))
    axs[0].plot(radial_gradient(mout_noflow, '[o/h]'), 'k--', label='No flows')
    axs[1].plot(radial_gradient(mout_noflow, '[fe/h]'), 'k--')
    axs[2].plot(radial_gradient(mout_noflow, '[o/fe]'), 'k--')
    
    axs[0].set_ylabel('[O/H]')
    axs[1].set_ylabel('[Fe/H]')
    axs[2].set_ylabel('[O/Fe]')
    axs[2].set_xlabel('Zone')
    
    axs[0].legend()
    
    fig.savefig(paths.figures / 'gasflows')
    plt.close()


def radial_gradient(multioutput, parameter, index=-1, Rmax=15.5,
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
