"""
This script plots the evolution of the radial gas density gradient with time.
"""

import math as m

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vice

import paths
from _globals import MAX_SF_RADIUS, ZONE_WIDTH

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')

    output_name = 'gaussian/outflow/no_gasflow/pristine/J21/twoinfall/diskmodel'
    # list of time indices
    times = list(range(0, 1321, 10))
    baseline = 8.0 # kpc
    Rsun = 8.0 # kpc

    multioutput = vice.output(str(paths.multizone / output_name))
    slopes = []
    for t in times:
        gas_densities = []
        for r in [Rsun - baseline/2, Rsun + baseline/2]:
            area = m.pi * ((r + ZONE_WIDTH)**2 - r**2)
            gas_densities.append(
                multioutput.zones['zone%s' % int(r/ZONE_WIDTH)].history[t]['mgas'] / area
            )
        slopes.append((m.log(gas_densities[1]) - m.log(gas_densities[0])) / baseline)
    
    fig, ax = plt.subplots()
    ax.plot([t*0.01 for t in times], slopes)
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'$\Delta \ln( \Sigma_g) / \Delta R$')
    fig.savefig(paths.figures / 'gas_gradient_evolution')
    plt.close()

if __name__ == '__main__':
    main()
