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
            self.first.norm = 0
