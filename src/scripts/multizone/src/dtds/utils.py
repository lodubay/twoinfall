"""
This file contains utility classes and functions for the DTD models
"""

import math as m
import vice

class exponential:
    """
    A generic normalized exponential function of time

    Attributes
    ----------
    timescale : float [default: 1]
        The exponential timescale in units of time.
    coeff : float [default: 1]
        The post-normalization coefficient.

    """
    def __init__(self, timescale=1, coeff = 1):
        self.timescale = timescale
        self.coeff = coeff
        self.norm = coeff / timescale

    def __call__(self, time):
        return self.norm * m.exp(-time / self.timescale)

class gaussian:
    """
    A generic normalized Gaussian function of time.

    Attributes
    ----------
    center : float
        The location of the peak of the Gaussian function.
    stdev : float
        The standard deviation of the Gaussian function.
    norm : float
        The normalization of the Gaussian function.

    """
    def __init__(self, center=1, stdev=1, coeff=1, normalize=True):
        """
        Initialize the Gaussian.

        Parameters
        ----------
        center : float [default: 1]
            The location of the peak of the Gaussian function.
        stdev : float [default: 1]
            The standard deviation of the Gaussian function.
        coeff : float [default: 1]
            The post-normalization coefficient.

        """
        self.center = center
        self.stdev = stdev
        self.coeff = coeff
        if normalize:
            self.norm = 1 / (stdev * m.sqrt(2 * m.pi))
        else:
            self.norm = 1

    def __call__(self, time):
        C = self.coeff * self.norm
        return C * m.exp(-(time-self.center)**2 / (2*self.stdev**2))


def minimum_wd_mass(secondary_mass, efficiency=1.):
    """
    The minimum acceptable white dwarf mass for a successful SN Ia event.

    Parameters
    ----------
    secondary_mass : float
        Mass of the secondary in solar masses
    efficiency : float, optional
        Efficiency of secondary mass transfer. In practice, any value
        above 0.3 will not affect the final DTD. The default is 1.

    Returns
    -------
    float
        Minimum WD mass in solar masses
    """
    envelope_mass = secondary_mass - remnant_mass(secondary_mass)
    return 1.4 - efficiency * envelope_mass

def remnant_mass(initial_mass):
    """
    Relationship between a star's initial mass and remnant mass according
    to Case B Roche lobe overflow from Nelemans et al. 2001 (equation 18
    from Greggio 2005).

    Parameters
    ----------
    initial_mass : float
        Initial stellar mass in solar masses

    Returns
    -------
    float
        Remnant mass in solar masses
    """
    return max((0.3,
                0.3 + 0.1 * (initial_mass - 2),
                0.5 + 0.15 * (initial_mass - 4)))

def mlr_wrapper(qty, which='mass', model='larson1974', **kwargs):
    """
    Wrapper for the mass-lifetime relation which allows the selection of any
    MLR model available in VICE.

    Parameters
    ----------
    model : str
        Designator of a mass-lifetime relation in VICE
    All other arguments and keyword arguments are passed to vice.mlr.<model>

    Returns
    -------
    float
        Output of vice.mlr.<model>
    """
    return {
        'larson1974': vice.mlr.larson1974,
        'mm1989': vice.mlr.mm1989,
        'pm1993': vice.mlr.pm1993,
        'ka1997': vice.mlr.ka1997,
        'hpt2000': vice.mlr.hpt2000,
        'vincenzo2016': vice.mlr.vincenzo2016,
        'powerlaw': vice.mlr.powerlaw
    }[model.lower()](qty, which=which, **kwargs)
