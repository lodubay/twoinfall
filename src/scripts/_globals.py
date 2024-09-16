"""
Global variables for the multizone simulations and plotting scripts.
"""

# Simulation globals are located at src/scripts/simulations/_globals.py
from multizone._globals import *
from multizone.src.dtds import plateau

# Plot bins in galactocentric radius and absolute z-height in kpc
GALR_BINS = [3, 5, 7, 9, 11, 13, 15]
ABSZ_BINS = [0, 0.5, 1, 2]

# AASTeX plot widths in inches
ONE_COLUMN_WIDTH = 3.25
TWO_COLUMN_WIDTH = 7.

# Default parameters for one-zone simulations
ONEZONE_DEFAULTS = {
    'elements': ELEMENTS,
    'eta': 2.15,
    'recycling': 'continuous',
    'RIa': plateau(width=1., slope=-1.1, tmin=MIN_RIA_DELAY),
    'delay': MIN_RIA_DELAY,
    'tau_star': 2,
    'dt': 0.01,
    'bins': [i*0.01 - 3 for i in range(401)],
    'Mg0': 1e6
}
