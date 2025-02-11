"""
This script plots the time-invariant mass-loading factor eta as a function of 
radius.
"""

import numpy as np
import matplotlib.pyplot as plt

from multizone.src.yields import yZ1
from multizone.src import outflows
import paths


def main():
    plt.style.use(paths.styles / "paper.mplstyle")
    
    fig, ax = plt.subplots(tight_layout=True)
    
    radii = np.arange(0, 15.5, 0.1)
    eta_func = outflows.equilibrium()
    ax.plot(radii, [eta_func(r) for r in radii], "k-")
    
    ax.axvline(8, color="r", ls="--")
    ax.axhline(eta_func(8), color="r", ls="--")
    
    ax.set_xlabel(r"$R_{\rm gal}$ [kpc]")
    ax.set_ylabel(r"$\eta\equiv\dot\Sigma_{\rm out}/\dot\Sigma_\star$")
    
    plt.savefig(paths.figures / "mass_loading")
    plt.close()
    

if __name__ == "__main__":
    main()
