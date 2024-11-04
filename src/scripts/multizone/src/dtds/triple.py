"""
This file defines a delay time distribution which approximates estimates from
triple system evolution by Rajamuthukumar et al. (2022).
"""

import math as m
from ..._globals import END_TIME
from .plateau import plateau

class triple(plateau):
    """
    An approximation of the Rajamuthukumar et al. (2022) triple-system DTD.
    
    This class defines a delay time distribution characterized by an initial
    (low) plateau, an instantaneous rise to a plateau between 500 Myr and 1 Gyr, 
    and a t^-1.1 declining power-law after 1 Gyr.
    
    Inherits from plateau.
    
    Parameters
    ----------
    early_rate : float, optional
        Initial Ia rate between tmin and rise_time as a fraction of peak rate.
        The default is 0.05.
    rise_time : float, optional
        Time of instantaneous rise in the Ia rate in Gyr. The default is 0.5.
    width : float, optional
        Length of time in Gyr to spend at the peak rate. The default is 0.5.
    slope : float, optional
        Slope of the power law at long delay times. The default is -1.1.
    tmin : float, optional
        Minimum Ia delay time in Gyr. The default is 0.04.
    tmax : float, optional
        Maximum Ia delay time in Gyr. The default is 13.2, the length of the 
        simulation.
    """
    def __init__(self, early_rate=0.05, rise_time=0.5, width=0.5, slope=-1.1, 
                 tmin=0.04, tmax=END_TIME):
        self.rise_time = rise_time
        super().__init__(width=width+rise_time, tmin=rise_time, tmax=tmax, 
                         slope=slope)
        self.early_rate = early_rate * super().__call__(rise_time)
        self.norm = 1
        # Normalize over full time range
        self.norm *= 1e-9 * self.normalize(tmin, tmax)
        self._name = 'triple_rise{:03d}_width{:03d}_slope{:02d}'.format(
            int(rise_time * 1000), int(width * 1000), int(abs(slope) * 10))
    
    def __call__(self, time):
        if time < self.rise_time:
            # Constant, low Ia rate before rise time
            return self.norm * self.early_rate
        else:
            # Same as plateau DTD after rise time
            return self.norm * super().__call__(time) 

    def normalize(self, tmin, tmax, dt=1e-3):
        """
        Calculate the normalization coefficient over the whole distribution.
    
        Parameters
        ----------
        tmin : float
            Lower limit of integration in Gyr.
        tmax : float
            Upper limit of integration in Gyr.
        dt : float [default: 1e-3]
            The numerical integration timestep in Gyr.
    
        Returns
        -------
        norm : float
            Normalization coefficient for which the total integral is unity.
    
        """
        integral = 0
        time = tmin
        while time < tmax:
            integral += self.__call__(time) * dt
            time += dt
        return 1 / integral

    @property
    def name(self):
        return self._name
    