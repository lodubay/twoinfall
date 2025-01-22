"""
This script plots the outputs of one-zone models which illustrate the effect
of the different two-infall model parameters - timescales and onset time.
"""

from onezone_params import main
from multizone.src.yields import yZ2

XLIM = (-1.7, 0.7)
YLIM = (-0.18, 0.54)
FIDUCIAL = {
    'first_timescale': 1.,
    'second_timescale': 10.,
    'onset': 3.
}

if __name__ == '__main__':
    main(fiducial=FIDUCIAL, xlim=XLIM, ylim=YLIM, fname='onezone_params_high_yields')
