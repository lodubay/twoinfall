r"""
The diskmodel objects employed in the Johnson et al. (2021) study.
"""

try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError
try:
    import vice
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("Could not import VICE.")
if vice.version[:2] < (1, 2):
    raise RuntimeError("""VICE version >= 1.2.0 is required to produce \
Johnson et al. (2021) figures. Current: %s""" % (vice.__version__))
else: pass
from vice.toolkit import hydrodisk, J21_sf_law
from .._globals import END_TIME, MAX_SF_RADIUS, ZONE_WIDTH
from .migration import diskmigration, gaussian_migration, no_migration
from . import models
from . import dtds
from .models.utils import get_bin_number, interpolate
from .models.gradient import gradient
import math as m
import sys

_SECONDS_PER_GYR_ = 3.1536e16
_KPC_PER_KM_ = 3.24e-17


class diskmodel(vice.milkyway):

    r"""
    A milkyway object tuned to the Johnson et al. (2021) models specifically.

    Parameters
    ----------
    zone_width : ``float`` [default : 0.1]
        The width of each annulus in kpc.
    name : ``str`` [default : "diskmodel"]
        The name of the model; the output will be stored in a directory under
        this name with a ".vice" extension.
    spec : ``str`` [default : "static"]
        A keyword denoting the time-dependence of the star formation history.
        Allowed values:

        - "static"
        - "insideout"
        - "lateburst"
        - "outerburst"
        - "twoinfall"
        - "earlyburst"
    
    verbose : ``bool`` [default : True]
        Whether or not the run the models with verbose output.
    migration_mode : ``str`` [default : "diffusion"]
        A keyword denoting the time-dependence of stellar migration.
        Allowed values:

        - "diffusion"
        - "linear"
        - "sudden"
        - "post-process"
        - "gaussian"
        - "none"

    delay : ``float`` [default : 0.04]
        Minimum SN Ia delay time in Gyr.
    RIa : ``str`` [default : "powerlaw"]
        A keyword denoting the time-dependence of the SN Ia rate.
        Allowed values:

        - "powerlaw"
        - "plateau"
        - "exponential"
        - "prompt"
        - "triple"

    RIa_kwargs : ``dict`` [default: {}]
        Keyword arguments to pass to the delay-time distribution initialization
    seed : ``int`` [default: 42]
        Random number generator seed.
    radial_gas_velocity : ``float`` [default: 0]
        Radial gas flow velocity in km/s. Positive denotes an outward gas flow.
    kwargs : varying types
        Other keyword arguments to pass ``vice.milkyway``.

    Attributes and functionality are inherited from ``vice.milkyway``.
    """

    def __init__(self, zone_width = 0.1, name = "diskmodel", spec = "static",
        verbose = True, migration_mode = "diffusion", yields="J21",
        delay = 0.04, RIa = "powerlaw", RIa_kwargs={}, seed=42, 
        radial_gas_velocity = 0., outflows=True, **kwargs):
        super().__init__(zone_width = zone_width, name = name,
            verbose = verbose, **kwargs)
        if self.zone_width <= 0.2 and self.dt <= 0.02 and self.n_stars >= 6:
            Nstars = 3102519
        else:
            Nstars = 2 * int(MAX_SF_RADIUS / zone_width * END_TIME / self.dt *
                self.n_stars)
        analogdata_filename = "%s_analogdata.out" % name
        if migration_mode == "gaussian":
            self.migration.stars = gaussian_migration(self.annuli, seed = seed,
                    zone_width = zone_width, filename = analogdata_filename,
                    post_process = self.simple)
        elif migration_mode == "none":
            self.migration.stars = no_migration(self.annuli, 
                                                filename=analogdata_filename)
        else:
            self.migration.stars = diskmigration(self.annuli,
                    N = Nstars, mode = migration_mode, 
                    filename = analogdata_filename)
        self.evolution = star_formation_history(spec = spec,
            zone_width = zone_width)
        # Set the yields
        if yields == "JW20":
            from vice.yields.presets import JW20
        elif yields == "C22":
            from .yields import C22
        elif yields == "F04":
            from .yields import F04
        elif yields == "W23":
            from .yields import W23
            # Mass-loading factor calibrated for equilibrium metallicity
            self.mass_loading = equilibrium_mass_loading()
        else:
            from .yields import J21
        # Set the SF mode - infall vs star formation rate
        if spec.lower() in ["twoinfall", "earlyburst"]:
            self.mode = "ifr"
            for zone in self.zones: zone.Mg0 = 0
        else:
            self.mode = "sfr"
        # Set the Type Ia delay time distribution
        dtd = delay_time_distribution(dist = RIa, tmin = delay, **RIa_kwargs)
        for i in range(self.n_zones):
            # set the delay time distribution and minimum Type Ia delay time
            self.zones[i].delay = delay
            self.zones[i].RIa = dtd
            # set the star formation efficiency timescale within 15.5 kpc
            mean_radius = (self.annuli[i] + self.annuli[i + 1]) / 2
            if mean_radius <= MAX_SF_RADIUS:
                area = m.pi * (self.annuli[i + 1]**2 - self.annuli[i]**2)
                if spec.lower() == "earlyburst":
                    self.zones[i].tau_star = models.earlyburst_tau_star(area)
                elif spec.lower() == "twoinfall":
                    self.zones[i].tau_star = models.twoinfall_sf_law(area)
                else:
                    # Simplified SF law, single power-law with cutoff
                    self.zones[i].tau_star = J21_sf_law(area, mode = self.mode,
                                                        index1=1.5, index2=1.5,
                                                        Sigma_g2=1e8)
            # Outflows
            if not outflows:
                self.zones[i].eta = 0.
        
        # CONSTANT GAS VELOCITY
        if radial_gas_velocity != 0:
            radial_gas_velocity *= _SECONDS_PER_GYR_
            radial_gas_velocity *= _KPC_PER_KM_ # vrad now in kpc / Gyr
            for i in range(self.n_zones):
                for j in range(self.n_zones):
                    if i - 1 == j:
                        # normalized to 10 Myr time interval
                        numerator = radial_gas_velocity**2 * 0.01**2
                        numerator -= 2 * i * zone_width * radial_gas_velocity * 0.01
                        denominator = zone_width**2 * (2 * i + 1)
                        self.migration.gas[i][j] = numerator / denominator
                    else:
                        self.migration.gas[i][j] = 0

    def run(self, *args, **kwargs):
        out = super().run(*args, **kwargs)
        self.migration.stars.close_file()
        return out

    @classmethod
    def from_config(cls, config, **kwargs):
        r"""
        Obtain a ``diskmodel`` object with the parameters encoded into a
        ``config`` object.

        **Signature**: diskmodel.from_config(config, **kwargs)

        Parameters
        ----------
        config : ``config``
            The ``config`` object with the parameters encoded as attributes.
            See src/simulations/config.py.
        **kwargs : varying types
            Additional keyword arguments to pass to ``diskmodel.__init__``.

        Returns
        -------
        model : ``diskmodel``
            The ``diskmodel`` object with the proper settings.
        """
        model = cls(zone_width = config.zone_width, **kwargs)
        model.dt = config.timestep_size
        model.n_stars = config.star_particle_density
        model.bins = config.bins
        model.elements = config.elements
        return model


class equilibrium_mass_loading:
    """
    An exponential outflow mass-loading parameter.
    
    Tuned to produce the given equilibrium abundance at the Solar radius
    assuming an exponential star formation history
    
    Parameters
    ----------
    alpha_h_eq : float [default: 0.0]
        The equilibrium [alpha/H] abundance at the Solar radius in dex.
    recycling : float [default: 0.4]
        The instantaneous recycling parameter.
    tau_star : float [default: 2.0]
        The star formation efficiency timescale in Gyr.
    tau_sfh : float [default: 10.0]
        The star formation history timescale in Gyr.
    
    Attributes
    ----------
    _eta_sun : float
        The mass-loading factor at the Solar radius.
    _scale_radius : float
        The exponential scale radius for the mass-loading factor in kpc.
    
    Calling
    -------
    Returns the value of the mass-loading factor at the given radius.
    
    Parameters:
        - radius : float
            Galactocentric radius in kpc.
    
    Returns:
        - eta : float
            The mass-loading factor at the given radius.
    """
    def __init__(self, alpha_h_eq=0., recycling=0.4, tau_star=2., tau_sfh=10.,
                 gradient=-0.08):
        # Calculate eta for exponential SFH to reach desired equilibrium [O/H]
        Z_alpha_eq = vice.solar_z["o"] * 10 ** alpha_h_eq
        yield_ratio = vice.yields.ccsne.settings["o"] / Z_alpha_eq
        self._eta_sun = yield_ratio - 1 + recycling + tau_star / tau_sfh
        self._scale_radius = -1 / (gradient * m.log(10))
        
    def __call__(self, radius):
        return self._eta_sun * m.exp((radius - 8) / self._scale_radius)


class star_formation_history:

    r"""
    The star formation history (SFH) of the model galaxy. This object will be
    used as the ``evolution`` attribute of the ``diskmodel``.

    Parameters
    ----------
    spec : ``str`` [default : "static"]
        A keyword denoting the time-dependence of the SFH.
    zone_width : ``float`` [default : 0.1]
        The width of each annulus in kpc.

    Calling
    -------
    - Parameters

        radius : ``float``
            Galactocentric radius in kpc.
        time : ``float``
            Simulation time in Gyr.
    """

    def __init__(self, spec = "static", zone_width = 0.1, dt = 0.01):
        self._radii = []
        self._evol = []
        i = 0
        max_radius = 20 # kpc, defined by ``vice.milkyway`` object.
        while (i + 1) * zone_width < max_radius:
            self._radii.append((i + 0.5) * zone_width)
            self._evol.append({
                "static":             models.static,
                "insideout":          models.insideout,
                "lateburst":          models.lateburst,
                "outerburst":         models.outerburst,
                "twoinfall":          models.twoinfall,
                "earlyburst":         models.earlyburst_ifr,
            }[spec.lower()]((i + 0.5) * zone_width, dr = zone_width, dt = dt))
            i += 1

    def __call__(self, radius, time):
        # The milkyway object will always call this with a radius in the
        # self._radii array, but this ensures a continuous function of radius
        if radius > MAX_SF_RADIUS:
            return 0
        else:
            idx = get_bin_number(self._radii, radius)
            if idx != -1:
                return gradient(radius) * interpolate(self._radii[idx],
                    self._evol[idx](time), self._radii[idx + 1],
                    self._evol[idx + 1](time), radius)
            else:
                return gradient(radius) * interpolate(self._radii[-2],
                    self._evol[-2](time), self._radii[-1], self._evol[-1](time),
                    radius)


class delay_time_distribution:
    """
    The delay time distribution (DTD) of Type Ia supernovae (SNe Ia) in the
    model galaxy. This object will be used as the ``RIa`` attribute of each zone
    in the ``diskmodel``.

    Parameters
    ----------'
    dist : str [default: "powerlaw"]
        A keyword denoting the delay-time distribution of SNe Ia.
    tmin : float [default: 0.04]
        The minimum SN Ia delay time in Gyr.
    tmax : float [default: 13.2]
        The maximum SN Ia delay time in Gyr.
    kwargs : dict
        Keyword arguments passed to the DTD initialization.

    Calling
    -------
    - Parameters

        time : ``float``
            Simulation time in Gyr.
    """

    def __init__(self, dist="powerlaw", tmin=0.04, tmax=END_TIME, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        self._dtd = {
            "powerlaw":         dtds.powerlaw,
            "plateau":          dtds.plateau,
            "exponential":      dtds.exponential,
            "prompt":           dtds.prompt,
            "greggio05_single": dtds.greggio05_single,
            "greggio05_double": dtds.greggio05_double,
            "triple":           dtds.triple,
        }[dist.lower()](tmin=tmin, tmax=tmax, **kwargs)

    def __call__(self, time):
        if time >= self.tmin and time <= self.tmax:
            return self._dtd(time)
        else:
            return 0.
