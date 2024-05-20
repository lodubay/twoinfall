"""
This file defines the exponential delay-time distribution (DTD) of Type Ia
supernovae.
"""

from .utils import exponential as generic_exponential
from ..._globals import END_TIME

class exponential(generic_exponential):
    """
    A normalized, exponentially-declining Type Ia supernova delay time
    distribution.

    Attributes
    ----------
    timescale : float [default: 1.5]
        The exponential timescale in units of Gyr.

    """
    def __init__(self, timescale=1.5, tmin=0.04, tmax=END_TIME):
        super().__init__(timescale=timescale)
        # Normalize to 1 between the minimum delay and maximum simulation time
        self.norm *= 1e-9 / (timescale * (self.__call__(tmin) - self.__call__(tmax)))
        self._name = f'exponential_timescale{int(10 * timescale)}'

    @property
    def name(self):
        return self._name
