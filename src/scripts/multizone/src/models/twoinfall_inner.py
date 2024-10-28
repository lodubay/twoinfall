"""
This file declares the time-dependence of the infall rate for a variant of
the two-infall model where the first infall is constrained to the inner
galaxy (<9 kpc).
"""

import math as m
from ..._globals import MAX_SF_RADIUS
from .twoinfall import twoinfall, FIRST_TIMESCALE

FIRST_MAX_RADIUS = 9 # kpc, maximum radius of the first infall

class twoinfall_inner(twoinfall):
    """
    Variant of the two-infall SFH where the first infall occurs only within the
    inner galaxy.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(radius, **kwargs)
        if radius > FIRST_MAX_RADIUS:
            self.first.norm = 1e-6
            self.first.timescale = 100
        else:
            # Compensate for lost thick-disk mass in outer galaxy
            self.first.norm *= (
                (m.exp(-1 * MAX_SF_RADIUS / FIRST_TIMESCALE) - 1) / 
                (m.exp(-1 * FIRST_MAX_RADIUS / FIRST_TIMESCALE) - 1)
            )
