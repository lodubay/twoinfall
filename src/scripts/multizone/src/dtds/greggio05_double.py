"""
This file contains a prescription for the DTD of SNe Ia following the double-
degenerate solution of Greggio 2005.
"""

from numbers import Number
import math as m
import numpy as np
from tqdm import tqdm
from ..._globals import END_TIME
from .utils import mlr_wrapper, minimum_wd_mass
from .greggio05_single import greggio05_single

class greggio05_double:
    """
    An numerical implementation of the double-degenerate DTD from Greggio 2005

    Attributes
    ----------
    scheme : str
        Keyword denoting assumptions about the evolution of a double-degenerate
        system, either 'wide' or 'close'
    beta : float
        The exponent of the power-law distribution of merger timescales
    efficiency : float
        The efficiency of secondary mass transfer
    dt : float
        The integration timestep in Gyr
    mlr : str
        Which of the VICE mass-lifetime relations to assume
    t_min : float
        Total minimum delay time in Gyr
    t_max : float
        Total maximum delay time in Gyr
    single_degenerate_distribution : greggio05_single
        Instance of the single-degenerate DTD from Greggio 2005
    times : numpy.ndarray
        Array of logarithmically-spaced times at which the DTD is calculated
    dtd : numpy.ndarray
        Array of pre-calculated normalized values of the DTD

    Methods
    -------
    integrate(time)
        Calculate the value of the DTD at the given time
    f_merge(self, time, t_nuc, case=1)
        The distribution function of DD merger timescales
    mass_dependence(self, m2, case=1)
        The term describing the dependence on the mass of the DD system
    asymptotic_nuclear_lifetime(self, time)
        The lower integration limit for the 'close' DD scheme
    maximum_gravitational_delay(t_nuc)
        Static method to calculate the maximum gravitational delay time
    """

    def __init__(self, scheme='wide', beta_sep=0., beta_grav=-0.75, efficiency=1.,
                 t_nuc_min=0.04, t_nuc_max=1., t_grav_min=1e-3, dt=1e-3, 
                 nsamples=100, mlr='larson1974', progress=True, **kwargs):
        """
        Parameters
        ----------
        scheme : str
            Keyword denoting assumptions about the evolution of a DD system,
            either 'wide' or 'close'.
        beta_sep : float, optional
            The exponent of the power law distribution adopted for the final
            separations in the 'wide' scheme; a flat distribution of
            of separations corresponds to beta_sep = 0. The default is 0.
        beta_grav : float, optional
            The exponent of the power law distribution adopted for the
            gravitational delays in the 'close' scheme; a flat distribution
            of gravitational delays corresponds to beta_grav = -0.75. The
            default is -0.75.
        efficiency : float, optional
            The mass-transfer efficiency represented by epsilon in Equation 17.
            The default is 1.
        t_nuc_min : float, optional
            The minimum nuclear lifetime of the secondary in Gyr. The default
            is 0.04, corresponding to the lifetime of an 8 solar mass star.
        t_nuc_max : float, optional
            The maximum nuclear lifetime of the secondary in Gyr. The default
            is 1, corresponding to the lifetime of a 2 solar mass star.
        t_grav_min : float, optional
            The minimum gravitational inspiral delay in Gyr. The default is
            1e-3.
        dt : float, optional
            Integration timestep in Gyr. The default is 1e-3.
        nsamples : int, optional
            Number of logarithmically-spaced points at which to pre-compute the
            DTD. The default is 100.
        mlr : str, optional
            Which model of mass-lifetime relation to use; must be one available
            in VICE. The default is 'larson1974'.
        progress : bool, optional
            If True, show a progress bar when computing the DTD. The default
            is True.
        kwargs : dict
            Keyword arguments passed to greggio05_single
        """

        self.scheme = scheme
        if scheme == 'wide':
            self.beta = beta_sep
        else:
            self.beta = beta_grav
        self.efficiency = efficiency
        self.t_nuc_min = t_nuc_min
        self.t_nuc_max = t_nuc_max
        self.t_grav_min = t_grav_min
        self.dt = dt
        self.mlr = mlr
        
        self._name = 'greggio05_double_%s' % self.scheme

        # Total minimum and maximum delay times
        self.t_min = self.t_nuc_min + self.t_grav_min
        self.t_max = self.t_nuc_max + self.maximum_gravitational_delay(self.t_nuc_max)

        self.single_degenerate_distribution = greggio05_single(
            efficiency=self.efficiency, mlr=self.mlr, **kwargs
        )

        # Pre-calculate the DTD to save on compute time
        if progress:
            print('Computing Greggio 2005 DD %s DTD...' % self.scheme.upper())
        self.times = np.logspace(np.log10(self.t_min),
                                 np.log10(min((END_TIME+self.dt, self.t_max))),
                                 num=nsamples, endpoint=True)
        self.dtd = np.zeros(self.times.shape)
        if progress:
            tlist = tqdm(self.times)
        else:
            tlist = self.times
        for i, t in enumerate(tlist):
            self.dtd[i] = self.integrate(t)

        # Normalize the DTD
        timesteps = self.times[1:] - self.times[:-1]
        self.dtd /= np.sum(self.dtd[:-1] * timesteps * 1e9)

    def __call__(self, time):
        """
        Linearly interpolate the DTD at the given time.

        Parameters
        ----------
        time : float
            Time after starburst in Gyr

        Returns
        -------
        float
            Interpolated value of the distribution function of DD SNe Ia,
            or NaN if time is outside the bounds of integration

        Raises
        ------
        ValueError
            If the time passed as a parameter is not positive.
        TypeError
            If the time passed as a parameter is not a float.
        """
        if isinstance(time, Number):
            if time >= 0.:
                if time < self.t_min:
                    return 0.
                elif time <= END_TIME:
                    return np.interp(time, self.times, self.dtd)
                else:
                    return 0.
            else:
                raise ValueError('Time must be positive.')
        else:
            raise TypeError('Parameter "time" must be numeric. Got: %s' \
                            % type(time))
                
    @property
    def name(self):
        return self._name

    @property
    def scheme(self):
        """
        Type: str
            Keyword denoting assumptions about the evolution of the
            double-degenerate system.
            Allowed values:

                - 'wide': there is a wide distribution of separation and total
                mass irrespective of the initial mass of the secondary
                - 'close': there is a tight correlation between the initial
                separation and the gravitational merge time
        """
        return self._scheme

    @scheme.setter
    def scheme(self, value):
        if isinstance(value, str):
            if value.lower() in ['wide', 'close']:
                self._scheme = value.lower()
            else:
                raise ValueError('DD scheme must be either "wide" or "close".')
        else:
            raise TypeError('Parameter "scheme" must be a string. Got: %s' % \
                            type(value))

    @property
    def beta(self):
        """
        Type: float
            The exponent of the power-law distribution of merger timescales.
            If scheme == 'wide', this is assumed to be the distribution of
            final separations; if scheme == 'close', this is assumed to be the
            distribution of gravitational delay times.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        if isinstance(value, Number):
            self._beta = float(value)
        else:
            raise TypeError('Parameter "beta" must be numeric. Got: %s' \
                            % type(value))

    @property
    def efficiency(self):
        """
        Type: float
            The single-degenerate mass-transfer efficiency represented by
            epsilon in Equation 17 from Greggio 2005.
        """
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        if isinstance(value, Number):
            if value >= 0 and value <= 1:
                self._efficiency = float(value)
            else:
                raise ValueError('Efficiency must be between 0 and 1.')
        else:
            raise TypeError('Parameter "efficiency" must be numeric. Got: %s'\
                            % type(value))

    @property
    def dt(self):
        """
        Type: float
            Integration timestep in Gyr
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        if isinstance(value, Number):
            if value > 0:
                self._dt = float(value)
            else:
                raise ValueError('Integration timestep must be positive.')
        else:
            raise TypeError('Parameter "dt" must be numeric. Got: %s' \
                            % type(value))

    @property
    def mlr(self):
        r"""
        Type: str
            Which formulation of the mass-lifetime relation (MLR) to assume.
            Allowed values:

                - 'larson1974'
                - 'mm1989'
                - 'pm1993'
                - 'ka1997'
                - 'hpt2000'
                - 'vincenzo2016'
                - 'powerlaw'
        """
        return self._mlr

    @mlr.setter
    def mlr(self, value):
        if isinstance(value, str):
            try:
                mlr_wrapper(1, model=value)
                self._mlr = value
            except:
                raise ValueError('Parameter "mlr" not in acceptable list.')
        else:
            raise TypeError('Parameter "mlr" must be a string. Got: %s' \
                            % type(value))

    def integrate(self, time):
        """
        Calculate the value of the DTD at the given time

        Integrate over all nuclear timescales up to the given time to obtain
        the distribution function of DD SNe Ia at that time.

        Parameters
        ----------
        time : float
            Time in Gyr since star formation event

        Returns
        -------
        float
            The value of the DTD at the given time
        """
        if time <= self.t_min or time >= self.t_max:
            return 0
        else:
            if self.scheme == 'wide':
                t_int_min = self.t_nuc_min
            else:
                t_int_min = self.asymptotic_nuclear_lifetime(time)
            t_int_max = min((self.t_nuc_max, time))
            integration_time = np.arange(t_int_min,
                                         t_int_max + self.dt,
                                         self.dt)
            # TODO this re-calculates the SD distribution every time, but it
            # doesn't change. figure out a way to speed this up
            integral = np.sum([
                self.single_degenerate_distribution(t_nuc) * \
                    self.f_merge(time, t_nuc, case=1) * \
                    self.dt \
                for t_nuc in integration_time])
            return integral

    def f_merge(self, time, t_nuc, case=1):
        """
        The distribution function of DD merger timescales.

        Parameters
        ----------
        time : float
            Time since starburst in Gyr
        t_nuc : float
            Nuclear timescale of the DD system in Gyr
        case : int [default: 1]
            Parameter controlling the form of the mass dependence for the
            'wide' scheme (see mass_dependence for more)

        Returns
        -------
        float
            Frequency of merger events
        """
        if self.scheme == 'wide':
            if t_nuc <= time - self.t_grav_min:
                m2 = mlr_wrapper(t_nuc, which='age', model=self.mlr)
                fW12 = self.mass_dependence(m2, case=case)
                return fW12 * (time - t_nuc) ** (-0.75 + 0.25 * self.beta)
            else:
                return 0
        else:
            if t_nuc <= time - self.t_grav_min:
                t_grav_max = self.maximum_gravitational_delay(t_nuc)
                t_diff = time - t_nuc
                return t_diff ** self.beta / (
                    t_grav_max ** (1 + self.beta) - \
                        self.t_grav_min ** (1 + self.beta))
            else:
                return 0

    def mass_dependence(self, m2, case=1):
        """
        The term describing the dependence on the mass of the DD system

        Equivalent to the f_(1,2)^W term in Greggio 2005, defined in
        Equation 29.

        Parameters
        ----------
        m2 : float
            Mass of the secondary in solar masses
        case : int [default: 1]
            Parameter controlling the relationship between the secondary mass
            m2 and the total double-degenerate mass MDD. If case = 1, there
            is a tight correlation between m2 and MDD; if case = 2, a wide
            distribution of MDD is obtained from systems with the same m2
        """
        if case == 1:
            MDD = 1.4 + (m2 - 2)/6
            return MDD ** (0.75 + 0.75 * self.beta)
        elif case == 2:
            m_wd_min = minimum_wd_mass(m2, efficiency=self.efficiency)
            m2R = max((2, 2 + 10 * (m_wd_min - 0.6)))
            MDDn = max((1.4, m2R + 0.6))
            MDDx = m2R + 1.2
            exp = 1.75 + 0.75 * self.beta
            return MDDx ** exp - MDDn ** exp
        else:
            raise ValueError('Parameter "case" must be 1 or 2.')

    def asymptotic_nuclear_lifetime(self, time):
        """
        The lower integration limit for the 'close' DD scheme

        Represented in Greggio 2005 as t_(n,inf) and defined in equation 38.

        Parameters
        ----------
        time : float
            A delay time in Gyr

        Returns
        -------
        float
            The lower limit of integration in Gyr
        """
        if time < self.t_nuc_min + \
            self.maximum_gravitational_delay(self.t_nuc_min):
            return self.t_nuc_min
        else:
            # Solution to the equation t - t_nuc = t_(gw,x)(t_nuc)
            t_nuc_arr = np.arange(self.t_nuc_min, self.t_nuc_max, self.dt)
            lhs = time - t_nuc_arr
            rhs = np.array([self.maximum_gravitational_delay(t_nuc) \
                            for t_nuc in t_nuc_arr])
            soln = t_nuc_arr[np.argmin(np.abs(lhs - rhs))]
            return soln

    @staticmethod
    def maximum_gravitational_delay(t_nuc):
        """
        Calculate the maximum gravitational delay time.

        The upper envelope of the delay due to gravitational inspiral,
        represented in Greggio 2005 as tau_(gw,x) in Equation 30. The
        gravitational delay is proportional to the binary separation ^4.

        Parameters
        ----------
        t_nuc : float
            Nuclear lifetime of the secondary in Gyr.
        """
        log_t_nuc = m.log10(t_nuc) + 9
        log_t_grav_max = min((-16.66 + 3.17 * log_t_nuc,
                              6.02 + 0.52 * log_t_nuc))
        return 10 ** (log_t_grav_max - 9)
