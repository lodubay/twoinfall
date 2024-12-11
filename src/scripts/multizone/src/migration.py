import random
import math as m
from numbers import Number
from vice.toolkit import hydrodisk
from .._globals import END_TIME, ZONE_WIDTH, RANDOM_SEED


class diskmigration(hydrodisk.hydrodiskstars):

    r"""
    A ``hydrodiskstars`` object which writes extra analog star particle data to
    an output file.

    Parameters
    ----------
    radbins : array-like
        The bins in galactocentric radius in kpc corresponding to each annulus.
    mode : ``str`` [default : "linear"]
        A keyword denoting the time-dependence of stellar migration.
        Allowed values:

        - "diffusion"
        - "linear"
        - "sudden"
        - "post-process"

    filename : ``str`` [default : "stars.out"]
        The name of the file to write the extra star particle data to.

    Attributes
    ----------
    write : ``bool`` [default : False]     
        A boolean describing whether or not to write to an output file when
        the object is called. The ``multizone`` object, and by extension the
        ``milkyway`` object, automatically switch this attribute to True at the
        correct time to record extra data.
    """

    def __init__(self, radbins, mode = "diffusion", filename = "stars.out",
        **kwargs):
        super().__init__(radbins, mode = mode, **kwargs)
        if isinstance(filename, str):
            self._file = open(filename, 'w')
            self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
        else:
            raise TypeError("Filename must be a string. Got: %s" % (
                type(filename)))

        # use only disk stars in these simulations
        self.decomp_filter([1, 2])

        # Multizone object automatically swaps this to True in setting up
        # its stellar population zone histories
        self.write = False

    def __call__(self, zone, tform, time):
        if tform == time:
            super().__call__(zone, tform, time) # reset analog star particle
            if self.write:
                if self.analog_index == -1:
                    # finalz = 100
                    finalz = 0
                    analog_id = -1
                else:
                    finalz = self.analog_data["zfinal"][self.analog_index]
                    analog_id = self.analog_data["id"][self.analog_index]
                self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
                    analog_id, finalz))
            else: pass
            return zone
        else:
            return super().__call__(zone, tform, time)

    def close_file(self):
        r"""
        Closes the output file - should be called after the multizone model
        simulation runs.
        """
        self._file.close()

    @property
    def write(self):
        r"""
        Type : bool

        Whether or not to write out to the extra star particle data output
        file. For internal use by the vice.multizone object only.
        """
        return self._write

    @write.setter
    def write(self, value):
        if isinstance(value, bool):
            self._write = value
        else:
            raise TypeError("Must be a boolean. Got: %s" % (type(value)))


class gaussian_migration:
    r"""
    A class which controls the Gaussian stellar migration scheme.
    
    The total migration distance $\Delta R$ is drawn from a Gaussian whose 
    width is determined by the star's age and birth radius. The final z-height 
    is drawn from a sech^2 distribution and written to an external file.
    
    Parameters
    ----------
    radbins : array-like
        The bins in galactocentric radius in kpc corresponding to each annulus.
    zone_width : float [default : 0.1]
        Width of each radial zone in kpc.
    end_time : float [default : 13.2]
        The final simulation timestep in Gyr.
    filename : ``str`` [default : "stars.out"]
        The name of the file to write the extra star particle data to.
    absz_max : float [default : 3]
        Maximum |z|-height above the midplane in kpc. The default corresponds
        to the maximum value in the h277 sample.
    time_power : float, optional [default: 0.33]
        Power on the time-dependence of the migration speed.
    sigma_rm8 : float, optional [default: 2.68]
        Measurement of the strength of radial migration in kpc for an 8 Gyr 
        old star.
    
    Attributes
    ----------
    write : ``bool`` [default : False]     
        A boolean describing whether or not to write to an output file when
        the object is called. The ``multizone`` object, and by extension the
        ``milkyway`` object, automatically switch this attribute to True at the
        correct time to record extra data.
        
    Calling
    -------
    Returns the star's current zone at the given time.
    zone : int
        Birth zone of the star.
    tform : float
        Formation time of the star in Gyr.
    time : float
        Current simulation time in Gyr.
    """
    def __init__(self, radbins, zone_width=ZONE_WIDTH, end_time=END_TIME,
                 filename="stars.out", absz_max=3., seed=RANDOM_SEED,
                 post_process=False, time_power=0.33, sigma_rm8=2.68):
        # Random number seed
        random.seed(seed)
        self.radial_bins = radbins
        self.zone_width = zone_width
        self.end_time = end_time
        self.absz_max = absz_max
        self.post_process = post_process
        self.time_power = time_power
        self.sigma_rm8 = sigma_rm8
        # super().__init__(radbins, mode=None, filename=filename, **kwargs)
        if isinstance(filename, str):
            self._file = open(filename, 'w')
            # Same format as above for compatibility
            self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
        else:
            raise TypeError("Filename must be a string. Got: %s" % (
                type(filename)))

        # Multizone object automatically swaps this to True in setting up
        # its stellar population zone histories
        self.write = False
        
    def __call__(self, zone, tform, time):
        Rform = self.zone_width * (zone + 0.5)
        age = self.end_time - tform
        if tform == time:
            if age > 0:
                # Randomly draw migration distance dR based on age & Rform
                while True: # ensure 0 < Rfinal <= Rmax
                    dR = random.gauss(
                        mu=0., 
                        sigma=self.migr_scale(
                            age, 
                            Rform, 
                            time_power=self.time_power, 
                            coeff=self.sigma_rm8
                        )
                    )
                    Rfinal = Rform + dR
                    if Rfinal > 0. and Rfinal <= self.radial_bins[-1]:
                        break
            else:
                # Stars born in the last simulation timestep won't migrate
                dR = 0.
                Rfinal = Rform
            self.dR = dR
            # Randomly draw final midplane distance and write to file
            if self.write:
                # vertical scale height based on age and Rfinal
                hz = self.scale_height(age, Rfinal)
                # draw from sech^2 distribution
                rng_max = self.sech2_cdf(self.absz_max, hz)
                rng_min = self.sech2_cdf(-self.absz_max, hz)
                rng = random.uniform(rng_min, rng_max)
                finalz = self.inverse_sech2_cdf(rng, hz)
                analog_id = -1
                self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
                    analog_id, finalz))
            else:
                pass
            return zone
        else: 
            if self.post_process:
                # Hold off on migrating stars until the end of the simulation
                if time < self.end_time:
                    return zone
                else:
                    R = Rform + self.dR
                    return int((R / self.zone_width))
            else:
                # Interpolate between Rform and Rfinal at current time
                R = self.interpolator(
                    Rform, Rform + self.dR, tform, time, power=self.time_power
                )
                return int(R / self.zone_width)

    def close_file(self):
        r"""
        Closes the output file - should be called after the multizone model
        simulation runs.
        """
        self._file.close()
    
    def interpolator(self, Rform, Rfinal, tform, time, power=0.33):
        r"""
        Interpolate between the formation and final radius following the fit
        to the h277 data, $\Delta R \propto t^{0.33}$.
        
        Parameters
        ----------
        Rform : float
            Radius of formation in kpc.
        Rfinal : float
            Final radius in kpc.
        tform : float
            Formation time in Gyr.
        time : float
            Simulation time in Gyr.
        time_power : float, optional [default: 0.33]
            Power on the time-dependence of the migration speed.
        
        Returns
        -------
        float
            Radius at the current simulation time in kpc.
        """
        tfrac = (time - tform) / (self.end_time - tform)
        return Rform + (Rfinal - Rform) * (tfrac ** power)
        
    @staticmethod
    def migr_scale(age, Rform, time_power=0.33, coeff=2.68):
        r"""
        A prescription for $\sigma_{\Delta R}$, the scale of the Gaussian 
        distribution of radial migration.
        
        $$ \sigma_{\Delta R} = 1.35 (R_\rm{form}/8\,\rm{kpc})^{0.61} 
        (\tau/1\,\rm{Gyr})^{0.33} $$
        
        Parameters
        ----------
        age : float or array-like
            Age of the stellar population in Gyr.
        Rform : float or array-like
            Formation radius of the stellar population in kpc.
        time_power : float, optional [default: 0.33]
            Power on the time-dependence of migration speed.
        coeff : float, optional [default: 1.35]
            Coefficient which scales the strength of radial migration for a
            1 Gyr old star (also known as $\sigma_{\rm RM8}$).
        
        Returns
        -------
        float or array-like
            Scale factor for radial migration $\sigma_{\Delta R}$.
        """
        return coeff * ((age / 8) ** time_power) * (Rform / 8) ** 0.61
    
    @staticmethod
    def scale_height(age, Rfinal):
        r"""
        The scale height $h_z$ as a function of age and final radius:
        
        $$ h_z = (0.25\,{\rm kpc}) 
        \exp\Big(\frac{\tau-5\,{\rm Gyr}}{7.0\,{\rm Gyr}}\Big) 
        \exp\Big(\frac{R_{\rm final}-8\,{\rm kpc}}{6.0\,{\rm kpc}}\Big) $$
        
        Parameters
        ----------
        age : float or array-like
            Age in Gyr.
        Rfinal : float or array-like
            Final radius $R_{\rm{final}}$ in kpc.
        
        Returns
        -------
        float
            Scale height $h_z$ in kpc.
        """
        return 0.25 * m.exp((age - 5.) / 7.) * m.exp((Rfinal - 8.) / 6.)
        # return 0.18 * (age ** 0.63) * (Rform / 8) ** 1.15
        
    @staticmethod
    def sech2_cdf(z, scale):
        r"""
        The cumulative distribution function (CDF) of the hyperbolic sec-square
        probability distribution function (PDF), which determines the density
        of stars as a function of distance from the midplane $z$. For some
        scaling $h_z$, the PDF is
        
        $$ \rm{PDF}(z) = \frac{1}{4 h_z} \cosh^{-2}\Big(\frac{z}{2 h_z}\Big) $$
        
        and the CDF is
        
        $$ \rm{CDF}(z) = \frac{1}{1 + e^{-z / h_z}} $$
        
        Parameters
        ----------
        x : float
            Independent variable.
        scale : float
            Width of the sech-squared distribution, with the same units as x.
        
        Returns
        -------
        float
            The value of the CDF at the given x.
        """
        return 1 / (1 + m.exp(-z / scale))
    
    @staticmethod
    def inverse_sech2_cdf(cdf, scale):
        r"""
        The inverse of the sech$^2$ CDF.
        
        For some scaling $h_z$, the $z$ corresponding to the given value of
        the CDF is
        
        $$ z = -h_z \ln\Big(\frac{1}{\rm{CDF}} - 1\Big) $$
        
        Parameters
        ----------
        cdf : float
            The value of the CDF. Must be within the exclusive interval (0, 1).
        scale : float
            Width of the sech$^2$ distribution. Must be positive.
        
        Returns
        -------
        float
            The value of $z$ corresponding to the CDF value, in the same units
            as ``scale''.
        """
        if cdf <= 0. or cdf >= 1.:
            raise ValueError("The value of the CDF must be between 0 and 1.")
        if scale <= 0.:
            raise ValueError("The scale height must be positive.")
        return -scale * m.log(1/cdf - 1)

    @property
    def write(self):
        r"""
        Type : bool

        Whether or not to write out to the extra star particle data output
        file. For internal use by the vice.multizone object only.
        """
        return self._write

    @write.setter
    def write(self, value):
        if isinstance(value, bool):
            self._write = value
        else:
            raise TypeError("Must be a boolean. Got: %s" % (type(value)))
    
    @property
    def time_power(self):
        """
        Type : float
        
        The power of the migration strength dependence on age.
        """
        return self._time_power
    
    @time_power.setter
    def time_power(self, value):
        if isinstance(value, Number):
            self._time_power = value
        else:
            raise TypeError("Must be a numeric value. Got: %s" % type(value))
    
    @property
    def sigma_rm8(self):
        """
        Type : float
        
        The standard deviation of the Gaussian from which migration distance
        is drawn for a star of age 8 Gyr.
        """
        return self._sigma_rm8
    
    @sigma_rm8.setter
    def sigma_rm8(self, value):
        if value > 0:
            self._sigma_rm8 = value
        else:
            raise ValueError("Radial migration strength must be positive.")


class no_migration:

    r"""
    A class which effects no migration but produces a similar output format to
    the `diskmigration` class.

    Parameters
    ----------
    radbins : array-like
        The bins in galactocentric radius in kpc corresponding to each annulus.
    filename : ``str`` [default : "stars.out"]
        The name of the file to write the extra star particle data to.

    Attributes
    ----------
    write : ``bool`` [default : False]     
        A boolean describing whether or not to write to an output file when
        the object is called. The ``multizone`` object, and by extension the
        ``milkyway`` object, automatically switch this attribute to True at the
        correct time to record extra data.
    
    Calling
    -------
    Returns the star's current zone at the given time, which is a constant.
    zone : int
        Birth zone of the star.
    tform : float
        Formation time of the star in Gyr.
    time : float
        Current simulation time in Gyr.
    """

    def __init__(self, radbins, filename = "stars.out", **kwargs):
        self.radial_bins = radbins
        if isinstance(filename, str):
            self._file = open(filename, 'w')
            self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
        else:
            raise TypeError("Filename must be a string. Got: %s" % (
                type(filename)))

        # Multizone object automatically swaps this to True in setting up
        # its stellar population zone histories
        self.write = False

    def __call__(self, zone, tform, time):
        if tform == time:
            if self.write:
                finalz = 0
                analog_id = -1
                self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
                    analog_id, finalz))
            else: pass
        return zone

    def close_file(self):
        r"""
        Closes the output file - should be called after the multizone model
        simulation runs.
        """
        self._file.close()
    
    
    @property
    def radial_bins(self):
        r"""
        Type : list [elements are positive real numbers]

        The bins in galactocentric radius in kpc describing the disk model.
        Must extend from 0 to at least 20 kpc. Need not be sorted in any way
        when assigned.

        Example Code
        ------------
        >>> from vice.toolkit.hydrodisk import hydrodiskstars
        >>> import numpy as np
        >>> example = hydrodiskstars([0, 5, 10, 15, 20])
        >>> example.radial_bins
        [0, 5, 10, 15, 20]
        >>> example.radial_bins = list(range(31))
        >>> example.radial_bins
        [0,
         1,
         2,
         ...
         17,
         18,
         19,
         20]
        """
        return self._radial_bins

    @radial_bins.setter
    def radial_bins(self, value):
        self._radial_bins = value

    @property
    def write(self):
        r"""
        Type : bool

        Whether or not to write out to the extra star particle data output
        file. For internal use by the vice.multizone object only.
        """
        return self._write

    @write.setter
    def write(self, value):
        if isinstance(value, bool):
            self._write = value
        else:
            raise TypeError("Must be a boolean. Got: %s" % (type(value)))
        