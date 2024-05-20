# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:48:42 2022

@author: dubay.11
"""

from .powerlaw import powerlaw
from ..._globals import END_TIME

class broken_powerlaw:
    """
    A two-part broken power-law delay-time distribution of SNe Ia. The default
    setting is a flat distribution (slope of 0) before the time of separation
    and the standard -1.1 slope after.

    Attributes
    ----------
    tsplit : float
        Time in Gyr separating the two power-law components.
    plaw1 : <function>
        The first power-law function, called when t < tsplit.
    plaw2 : <function>
        The second power-law function, called when t >= tsplit.

    """
    def __init__(self, tsplit=0.2, slope1=0, slope2=-1.1, tmin=0.04,
                 tmax=END_TIME):
        """
        Initialize the broken power-law.

        Parameters
        ----------
        tsplit : float [default: 0.2]
            Time in Gyr separating the two components.
        slope1 : float [default: 0]
            The slope of the first power-law.
        slope2 : float [default: -1.1]
            The slope of the second power-law.
        tmin : float [default: 0.04]
            The minimum delay time and lower limit of integration.
        tmax : float [default: 13.2]
            The maximum simulation time and upper limimt of integration.

        """
        self.tsplit = tsplit
        # Initialize both power-law functions
        plaw1 = powerlaw(slope=slope1, tmin=tmin, tmax=tsplit, coeff=1e-9)
        plaw2 = powerlaw(slope=slope2, tmin=tsplit, tmax=tmax, coeff=1e-9)
        # Calculate new normalization coefficients
        norm1 = (plaw1.norm**-1 + tsplit**(slope1-slope2) * plaw2.norm**-1)**-1
        plaw1.norm = norm1
        norm2 = tsplit**(slope1-slope2) * norm1
        plaw2.norm = norm2
        self.plaw1 = plaw1
        self.plaw2 = plaw2

    def __call__(self, time):
        """
        Calculate the normalized SN Ia rate at a given time.

        Parameters
        ----------
        time : float
            Time since starburst in Gyr.

        Returns
        -------
        RIa : float
            Normalized SN Ia rate per solar mass per year.

        """
        if time < self.tsplit:
            RIa = self.plaw1(time)
        else:
            RIa = self.plaw2(time)
        return RIa
