r"""
This file implements a version of the star formation efficiency model from 
Conroy et al. (2022) to be used in VICE multizone simulations.
"""

from vice.toolkit import J21_sf_law

class earlyburst_sf_law(J21_sf_law):
    r"""
    An implementation of the Conroy et al. (2022) SFE timescale model.
    
    This class maintains the dependence of $\tau_*$ on $M_{\rm{gas}}$ as a
    three-component power-law which is implemented in `vice.toolkit.J21_sf_law`.
    However, it strips out the time dependence (based on the SFE timescale
    of molecular gas as a function of redshift per Tacconi et al. 2018) and
    replaces it with the model by Conroy et al. (2022).
    """
    def __init__(self, area, t1=2.5, t2=3.7, slope=3, tau_star_init=50, 
                 tau_star_final=2.36, mode = "ifr", **kwargs):
        self.t1 = t1
        self.t2 = t2
        self.slope = slope
        self.tau_star_init = tau_star_init
        self.tau_star_final = tau_star_final
        super().__init__(area, mode = mode, **kwargs)
        
    def __call__(self, time, mgas):
        mgas_dependence = super().__call__(time, mgas) / self.molecular(time)
        return mgas_dependence * self.time_dependence(time)

    def time_dependence(self, time):
        r"""
        Implementation of the star formation efficiency timescale prescription
        in Equation 3 of Conroy et al. (2022).

        Parameters
        ----------
        time : float
            Time after the starburst in Gyr.

        Returns
        -------
        float
            Star formation efficiency timescale $\tau_*$ in Gyr.

        """
        if time < self.t1:
            return self.tau_star_init
        elif time >= self.t1 and time <= self.t2:
            return self.tau_star_init / ((1 + self.slope * (time - self.t1))**2)
        else:
            return self.tau_star_final
