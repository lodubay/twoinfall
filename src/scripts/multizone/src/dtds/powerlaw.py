"""
This file defines the power-law delay-time distribution (DTD) of Type Ia
supernovae.
"""

from ..._globals import END_TIME

class powerlaw:
    """
    A normalized power-law delay time distribution.

    Attributes
    ----------
    slope : float [default: -1.1]
        The slope of the power-law.
    coeff : float [default: 1e-9]
        The post-normalization coefficient. The default is 1e-9 to
        convert between the timescale (in Gyr) and the rate (in yr^-1).
    norm : float
        The normalization coefficient, determined by integrating over the
        given range.

    """
    def __init__(self, slope=-1.1, coeff=1e-9, tmin=0.04, tmax=END_TIME):
        """
        Initialize the power-law function.

        Parameters
        ----------
        tmin : float [default: 0.04]
            The lower bound in Gyr of range over which to normalize.
        tmax : float [default: 13.2]
            The upper bound in Gyr of range over which to normalize.

        """
        self.slope = slope
        self.coeff = coeff
        self.tmin = tmin
        self.tmax = tmax
        self.norm = self.normalize(tmin, tmax)
        self._name = 'powerlaw_slope{:02d}'.format(int(10 * abs(slope)))

    def __call__(self, time):
        if time >= self.tmin and time < self.tmax:
            return self.coeff * self.norm * (time ** self.slope)
        else:
            return 0

    def normalize(self, tmin, tmax):
        intslope = self.slope + 1 # The slope of the integral
        if intslope == 0:
            # Prevent divide-by-zero case
            return 1 / ((self.integral(self.slope+1e-3, tmin, tmax) +
                         self.integral(self.slope-1e-3, tmin, tmax)) / 2)
        else:
            return 1 / self.integral(self.slope, tmin, tmax)

    @staticmethod
    def integral(slope, tmin, tmax):
        """
        Calculate the analytic integral of the power-law between two bounds.
        """
        intslope = slope + 1
        return (tmax ** intslope - tmin ** intslope) / intslope

    @property
    def name(self):
        return self._name
