"""
This file contains the greggio05_approximate class and related classes and
functions.
"""

import math as m
from numbers import Number
import numpy as np
from scipy.optimize import curve_fit
from ..._globals import END_TIME
from .greggio05_double import greggio05_double

# Best-fit analytic parameters for WIDE DD prescription
WIDE_PARAMS = {
    'tsplit': 1.,
    'slope1': -0.19,
    'slope2': -0.91,
    'rise_strength': 1.7,
    'rise_timescale': 0.084,
    'tail_strength': 0.85,
    'tail_timescale': 0.088,
}
# Best-fit analytic parameters for CLOSE DD prescription
CLOSE_PARAMS = {
    'tsplit': 1.,
    'slope1': -0.53,
    'slope2': -1.1,
    'rise_strength': 3.0,
    'rise_timescale': 0.042,
    'tail_strength': 0.62,
    'tail_timescale': 0.11,
}

class greggio05_approximate:
    """
    An analytic approximation of the double-degenerate DTD from Greggio 2005

    This DTD model consists of two parts: the first, prior to 1 Gyr, is an
    exponential rise into a power-law; the second, after 1 Gyr, is an
    exponential decline into a second (usually steeper) power-law.
    """
    def __init__(self, tsplit=1., slope1=-0.19, slope2=-0.91,
                 rise_strength=1.7, rise_timescale=8.4e-2,
                 tail_strength=0.85, tail_timescale=8.8e-2, normalize=True,
                 name='greggio05_approx', **kwargs):
        """
        Parameters
        ----------
        tsplit : float, optional
            Break time in Gyr of the piecewise function. The default is 1.
        slope1 : float, optional
            The slope of the first power-law. The default is -0.2
        slope2 : float, optional
            The slope of the second power-law. The default is
        rise_strength : float, optional
            Coefficient of the exponential rise. The default is
        rise_timescale : float, optional
            Timescale in Gyr of the exponential rise. The default is
        tail_strength : float, optional
            Coefficient of the exponential tail. The default is
        tail_timescale : float, optional
            Timescale in Gyr of the exponential tail. The default is
        normalize : bool, optional
            Whether to normalize the DTD. The default is True.
        name : str, optional
            Identifying name for chemical evolution model runs. The default is
            'greggio05_approx'.
        **kwargs passed to self.normalize()
        """
        self._name = name
        self.tsplit = tsplit
        self.slope1 = slope1
        self.slope2 = slope2
        self.rise = exponential_scaling(rise_strength, rise_timescale)
        self.tail = exponential_scaling(-tail_strength, tail_timescale,
                                        offset=self.tsplit)
        self.norm = 1
        if normalize:
            self.norm = self.normalize(**kwargs)

    def __call__(self, time):
        """
        Parameters
        ----------
        time : float
            Time since star formation event in Gyr

        Returns
        -------
        float
            Value of the DTD at the given time
        """
        if isinstance(time, Number):
            if time < 0:
                raise ValueError('Time must be non-negative.')
            elif time <= self.tsplit:
                RIa = self.norm * self.part1(time)
            else:
                RIa = self.norm * self.part2(time) * \
                    (self.part1(self.tsplit) / self.part2(self.tsplit))
            if RIa < 0:
                return 0
            else:
                return RIa
        else:
            raise TypeError('Parameter "time" must be numeric. Got: %s' \
                            % type(time))

    @classmethod
    def from_defaults(cls, scheme, **kwargs):
        params = {
            'wide': WIDE_PARAMS,
            'close': CLOSE_PARAMS
        }[scheme]
        return cls(name='greggio05_approx_%s' % scheme, **params, **kwargs)

    @classmethod
    def fit_to_model(cls, scheme, tmin=0.04, tmax=END_TIME, tstep=0.01,
                     **kwargs):
        """
        Fit the analytic function to the greggio05_double model from scratch.

        Warning! Takes a long time.
        """
        model = greggio05_double(scheme, **kwargs)
        tarr = np.arange(tmin, tmax, tstep)
        yarr = np.array([model(t) for t in tarr])
        return cls.fit_to_data(tarr, yarr, name='greggio05_approx_%s' % scheme)

    @classmethod
    def fit_to_data(cls, tarr, yarr, name='greggio05_approx'):
        """
        Fit the analytic function to data from a pre-generated DTD.
        """
        popt, pcov = curve_fit(analytic_wrapper, tarr, yarr,
                               p0=(-0.5, -1, 1, 0.1, 1, 0.1, 1e-9))
        return cls(slope1=popt[0], slope2=popt[1],
                   rise_strength=popt[2], rise_timescale=popt[3],
                   tail_strength=popt[4], tail_timescale=popt[5],
                   name=name)
    
    @property
    def name(self):
        return self._name

    def normalize(self, tmin=0.04, tmax=END_TIME, dt=0.001):
        tarr = np.arange(tmin, tmax+dt, dt)
        integral = np.sum([self.__call__(t) * dt * 1e9 for t in tarr])
        return 1 / integral

    def part1(self, time):
        return self.rise(time) * time ** self.slope1

    def part2(self, time):
        return self.tail(time) * time ** self.slope2

class exponential_scaling:
    """An exponential scaling function of time.

    At small times, this function approaches a declining exponential. At large
    times, the function approaches an asymptote of 1.
    """
    def __init__(self, strength, timescale, offset=0):
        """
        Parameters
        ----------
        strength : float
            The strength of the exponential scaling. If positive, the function
            increases over time; if negative, it decreases over time
        timecale : float
            The exponential timescale in the same units of time
        offset : float, optional
            Horizontal (temporal) offset of the exponential. The default is 0
        """
        self.strength = strength
        self.timescale = timescale
        self.offset = offset

    def __call__(self, time):
        return (1 - self.strength * m.exp(-(time - self.offset) / self.timescale))


def analytic_wrapper(time, slope1, slope2, rise_strength, rise_timescale,
                     tail_strength, tail_timescale, norm):
    """
    A wrapper function for initializing the greggio05_approximate class which
    takes time and all fit parameters as arguments.
    """
    cl = greggio05_approximate(slope1=slope1, slope2=slope2,
                               rise_strength=rise_strength,
                               rise_timescale=rise_timescale,
                               tail_strength=tail_strength,
                               tail_timescale=tail_timescale,
                               normalize=False)
    if isinstance(time, np.ndarray):
        return np.array([norm * cl(t) for t in time])
    else:
        return norm * cl(time)
