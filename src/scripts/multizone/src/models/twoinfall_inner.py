"""
This file declares the time-dependence of the infall rate for a variant of
the two-infall model where the first infall is constrained to the inner
galaxy (<9 kpc).
"""

from .twoinfall_expvar import twoinfall_expvar
from ..._globals import THICK_DISK_SCALE_RADIUS, MAX_SF_RADIUS
import math as m

FIRST_MAX_RADIUS = 3 # kpc, maximum radius of the first infall

class twoinfall_inner(twoinfall_expvar):
    """
    Variant of the two-infall SFH where the first infall occurs only within the
    inner galaxy.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(radius, **kwargs)
        if radius > FIRST_MAX_RADIUS:
            self.first.norm = 0
        else:
            # Increase inner normalization to maintain overall mass ratio
            thick_disk_total = 1 - m.exp(- MAX_SF_RADIUS / THICK_DISK_SCALE_RADIUS)
            thick_disk_inner = 1 - m.exp(- FIRST_MAX_RADIUS / THICK_DISK_SCALE_RADIUS)
            self.first.norm *= thick_disk_total / thick_disk_inner
