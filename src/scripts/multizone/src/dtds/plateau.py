"""
This file defines the exponential delay-time distribution (DTD) of Type Ia
supernovae.
"""

from .broken_powerlaw import broken_powerlaw
from ..._globals import END_TIME

class plateau(broken_powerlaw):
    """
    A normalized delay-time distribution which consists of a flat plateau
    followed by a declining power-law.
    """
    def __init__(self, width=1.0, slope=-1.1, tmin=0.04, tmax=END_TIME):
        super().__init__(tsplit=width+tmin, slope1=0, slope2=slope, tmin=tmin,
                         tmax=tmax)
        self._name = 'plateau_width{:03d}_slope{:02d}'.format(
            int(width * 1000), int(abs(slope) * 10))

    @property
    def name(self):
        return self._name
