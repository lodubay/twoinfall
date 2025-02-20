"""
This file contains the MultizoneStars class and related functions, which handle
star particle data from VICE multizone simulation outputs.
"""

import math as m
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import vice

import paths
from _globals import RANDOM_SEED, ZONE_WIDTH
from utils import box_smooth, sample_rows, weighted_quantile


def main():
    pass


class MultizoneStars:
    """
    Star particle data from VICE multizone outputs.
    
    Note
    ----
    In all but rare circumstances, new instances should be generated using
    the MultizoneStars.from_output() class method.
    
    Parameters
    ----------
    stars : pandas.DataFrame
        Full star particle data including abundance ratios and z-heights.
    name : str, optional
        Name of VICE multizone output. The default is ''.
    fullpath : str or pathlib.Path, optional
        Full path to VICE multizone output, including parent directory and
        '.vice' extension. The default is ''.
    zone_width : float, optional
        Width of simulation zones in kpc. The default is 0.1.
    galr_lim : tuple, optional
        Lower and upper bounds on galactocentric radius in kpc. The default
        is None, which calculates these bounds automatically.
    absz_lim : tuple, optional
        Similar to galr_lim but for absolute z-height. The default is None.
    noisy : bool, optional
        If True, indicates output has been convolved with observational
        uncertainty. The default is False.
    """
    def __init__(self, stars, name='', fullpath='', zone_width=ZONE_WIDTH,
                 galr_lim=None, absz_lim=None, noisy=False):
        self.stars = stars
        self.name = name
        self.fullpath = fullpath
        self.zone_width = zone_width
        # Automatically calculate bounds of galactic region if none are given
        if galr_lim is None:
            galr_lim = (m.floor(10 * self.stars['galr_final'].min()) / 10.,
                        m.ceil(10 * self.stars['galr_final'].max()) / 10.)
        self.galr_lim = galr_lim
        if absz_lim is None:
            absz_lim = (m.floor(10 * self.stars['zfinal'].abs().min()) / 10.,
                        m.ceil(10 * self.stars['zfinal'].abs().max()) / 10.)
        self.absz_lim = absz_lim
        self.noisy = noisy
        
    @classmethod
    def from_output(cls, name, zone_width=ZONE_WIDTH, verbose=False):
        """
        Generate an instance of MultizoneStars from a VICE multizone output.
        
        Parameters
        ----------
        name : str
            Name of VICE output, excluding ``.vice`` extension.
        zone_width : float, optional
            Width of simulation zones in kpc. The default is 0.1.
        verbose : bool, optional
            If True, print some updates to the terminal. The default is False.
        
        Returns
        -------
        MultizoneStars instance
        """
        fullpath = paths.multizone / (name + '.vice')
        # Import star tracer data
        if verbose: 
            print('Importing VICE multizone data from', str(fullpath))
        stars = pd.DataFrame(vice.stars(str(fullpath)).todict())
        # Calculate log age (in years)
        with np.errstate(divide='ignore'): # some stars will have 0 age
            stars['log_age'] = np.log10(stars['age']) + 9.
        # Calculate [Fe/O]
        stars['[fe/o]'] = -stars['[o/fe]']
        # Convert radial zone indices to Galactic radii in kpc
        stars['galr_origin'] = zone_width * stars['zone_origin']
        stars['galr_final'] = zone_width * stars['zone_final']
        stars['dr'] = stars['galr_final'] - stars['galr_origin']
        # Calculate remaining stellar mass for each particle
        stars['mstar'] = stars['mass'] * (
            1 - stars['age'].apply(vice.cumulative_return_fraction))
        # Import star particle analogue (z-height) data
        analogdata = pd.read_csv(str(fullpath).replace('.vice', '_analogdata.out'),
                                 comment='#', sep='\t',
                                 names=['zone_origin', 'time_origin', 
                                        'analog_id', 'zfinal']
        )
        # Limit analogdata to same max time as stars data
        tmax = stars['formation_time'].max()
        analogdata = analogdata[analogdata['time_origin'] <= tmax]
        assert(stars.shape[0] == analogdata.shape[0])
        # Combine
        stars[['analog_id', 'zfinal']] = analogdata[['analog_id', 'zfinal']]
        return cls(stars, name=name, fullpath=fullpath, zone_width=zone_width)
        
    def __call__(self, cols=[]):
        """
        Return the ``stars`` dataframe or a subset of the dataframe.
        
        Parameters
        ----------
        cols : str or list of strings, optional
            If an empty list, return the entire DataFrame. If a string, return
            that column of the DataFrame. If a list of strings, return a subset
            of those columns in the DataFrame. The default is [].
            
        Returns
        -------
        pandas.DataFrame or pandas.Series
            Star particle data or subset of that data.
        """
        if cols == []:
            return self.stars
        else:
            # Error handling
            if isinstance(cols, str):
                if cols not in self.stars.columns:
                    raise ValueError(
                        "Not a column in the stars dataframe:" + cols
                    )
            elif isinstance(cols, list):
                if all([isinstance(c, str) for c in cols]):
                    if not all([c in self.stars.columns for c in cols]):
                        raise ValueError("All elements must be column names.")
                else:
                    raise TypeError(
                        "Each element of ``cols`` must be a string."
                    )
            else:
                raise TypeError("Must be a string or list of strings.")
            return self.stars[cols]
        
    def copy(self):
        """
        Create a copy of the current instance.
        
        Returns
        -------
        MultizoneStars instance
        """
        return MultizoneStars(self.stars.copy(), name=self.name, 
                              fullpath=self.fullpath, 
                              zone_width=self.zone_width,
                              galr_lim=self.galr_lim, absz_lim=self.absz_lim, 
                              noisy=self.noisy)
        
    def filter(self, filterdict, inplace=False):
        """
        Filter stars by the given parameter bounds.
        
        Parameters
        ----------
        filterdict : dict
            Dictionary containing the parameters and bounds with which to
            filter the data. Each key must be a column in the data and
            each value must be a tuple of lower and upper bounds. If either
            element in the tuple is None, the corresponding limit will not be
            applied.
        inplace : bool, optional
            If True, modifies the data of the current instance. If False,
            returns a new instance with the filtered data. The default is
            False.
        """
        stars_copy = self.stars.copy()
        if isinstance(filterdict, dict):
            for key in filterdict.keys():
                # Error handling
                if key not in self.stars.columns:
                    raise ValueError('Keys in "filterdict" must be data',
                                     'column names.')
                elif not isinstance(filterdict[key], tuple):
                    raise TypeError('Each value in "filterdict" must be a',
                                    'tuple of length 2.')
                elif len(filterdict[key]) != 2:
                    raise ValueError('Each value in "filterdict" must be a',
                                     'tuple of length 2.')
                elif not all([isinstance(val, Number) or val is None \
                              for val in filterdict[key]]):
                    raise TypeError('Each element of the tuple must be numeric',
                                    'or NoneType.')
                else:
                    colmin, colmax = filterdict[key]
                    if colmin is not None:
                        stars_copy = stars_copy[stars_copy[key] >= colmin]
                    if colmax is not None:
                        stars_copy = stars_copy[stars_copy[key] < colmax]
            if inplace:
                self.stars = stars_copy
            else:
                return MultizoneStars(stars_copy, name=self.name, 
                                      fullpath=self.fullpath, 
                                      zone_width=self.zone_width, 
                                      galr_lim=self.galr_lim, 
                                      absz_lim=self.absz_lim, 
                                      noisy=self.noisy)
        else:
            raise TypeError('Parameter "filterdict" must be a dict. Got:',
                            type(filterdict))
            
        
    def region(self, galr_lim=(0, 20), absz_lim=(0, 3), min_mass=1.0, 
               origin=False, inplace=False):
        """
        Slice DataFrame of stars within a given Galactic region.

        Parameters
        ----------
        stars : pandas DataFrame
            VICE multizone star data.
        galr_lim : tuple
            Minimum and maximum Galactic radius in kpc. The default is (0, 20).
        absz_lim : tuple
            Minimum and maximum of the absolute value of z-height in kpc. The
            default is (0, 5).
        min_mass : float, optional
            Minimum mass of stellar particle. The default is 1.
        origin : bool, optional
            If True, filter by star's original radius instead of final radius. 
            The default is False.
        inplace : bool, optional
            If True, update the current class instance. If False, returns a 
            new class instance with the limited subset. The default is False.

        Returns
        -------
        MultizoneStars instance or None
        """
        galr_min, galr_max = galr_lim
        absz_min, absz_max = absz_lim
        if origin:
            galr_col = 'galr_origin'
        else:
            galr_col = 'galr_final'
        # Select subset
        subset = self.stars[(self.stars[galr_col]       >= galr_min) &
                            (self.stars[galr_col]       <  galr_max) &
                            (self.stars['zfinal'].abs() >= absz_min) &
                            (self.stars['zfinal'].abs() <  absz_max) &
                            (self.stars['mstar']        >= min_mass)]
        subset.reset_index(inplace=True, drop=True)
        if inplace:
            self.stars = subset
            self.galr_lim = galr_lim
            self.absz_lim = absz_lim
        else:
            return MultizoneStars(subset, name=self.name, 
                                  fullpath=self.fullpath, 
                                  zone_width=self.zone_width, 
                                  galr_lim=galr_lim, absz_lim=absz_lim, 
                                  noisy=self.noisy)
    
    def model_uncertainty(self, apogee_data=None, inplace=False, 
                          seed=RANDOM_SEED, age_col='L23_AGE'):
        """
        Forward-model observational uncertainties from median data errors.
        Star particle data are modified in-place, so only run this once!
        
        Parameters
        ----------
        apogee_data : pandas.DataFrame or NoneType, optional
            Full APOGEE data. If None, will be imported from ``sample.csv``.
            The default is None.
        inplace : bool, optional
            If True, update the current class instance. If False, returns a 
            new class instance with the noisy outputs. The default is False.
        seed : int, optional
            Random seed for sampling. The default is taken from _globals.py.
            
        Returns
        -------
        MultizoneStars instance or None
        """
        noisy_stars = self.stars.copy()
        if apogee_data is None:
            apogee_data = pd.read_csv(paths.data/'APOGEE/sample.csv')
        rng = np.random.default_rng(seed)
        if age_col == 'CN_AGE':
            age_med_err = apogee_data['CN_AGE_ERR'].median()
            age_noise = rng.normal(scale=age_med_err, size=noisy_stars.shape[0])
            noisy_stars['age'] += age_noise
        else:
            # Age uncertainty (Leung et al. 2023)
            log_age_err = apogee_data['L23_LOG_AGE_ERR'].median()
            log_age_noise = rng.normal(scale=log_age_err, 
                                    size=noisy_stars.shape[0])
            noisy_stars['age'] *= 10 ** log_age_noise
        # [Fe/H] uncertainty
        feh_med_err = apogee_data['FE_H_ERR'].median()
        feh_noise = rng.normal(scale=feh_med_err, size=noisy_stars.shape[0])
        noisy_stars['[fe/h]'] += feh_noise
        # [O/Fe] uncertainty 
        ofe_med_err = apogee_data['O_FE_ERR'].median()
        ofe_noise = rng.normal(scale=ofe_med_err, size=noisy_stars.shape[0])
        noisy_stars['[o/fe]'] += ofe_noise
        # [Fe/O] uncertainty 
        feo_med_err = apogee_data['FE_O_ERR'].median()
        feo_noise = rng.normal(scale=feo_med_err, size=noisy_stars.shape[0])
        noisy_stars['[fe/o]'] += feo_noise
        if inplace:
            self.stars = noisy_stars
        else:
            return MultizoneStars(noisy_stars, name=self.name, 
                                  fullpath=self.fullpath, 
                                  zone_width=self.zone_width, 
                                  galr_lim=self.galr_lim, 
                                  absz_lim=self.absz_lim, 
                                  noisy=True)
        
    def sample(self, N, seed=RANDOM_SEED):
        """
        Randomly sample N rows from the full DataFrame, weighted by mass.
        
        Parameters
        ----------
        N : int
            Number of samples to draw without replacement.
        seed : int, optional
            Random seed for sampling. The default is taken from _globals.py.
        
        Returns
        -------
        pandas.DataFrame
            N samples of full DataFrame.
        """
        sample_weights = self.stars['mstar'] / self.stars['mstar'].sum()
        sample = sample_rows(self.stars.copy(), N, weights=sample_weights,
                             seed=seed)
        return sample
    
    def resample_zheight(self, N, apogee_data=None, seed=RANDOM_SEED,
                         inplace=False):
        """
        Randomly sample N rows from the full DataFrame of stellar populations,
        weighted by the vertical distribution of stars in APOGEE.
        
        Parameters
        ----------
        N : int
            Number of samples to draw without replacement.
        apogee_data : pandas.DataFrame or NoneType, optional
            Full APOGEE data. If None, will be imported from ``sample.csv``.
            The default is None.
        seed : int, optional
            Random seed for sampling. The default is taken from _globals.py.
        inplace : bool, optional
            If True, update the current class instance. If False, returns a 
            new class instance with the noisy outputs. The default is False.
            
        Returns
        -------
        MultizoneStars instance or None
        """
        if apogee_data is None:
            apogee_data = pd.read_csv(paths.data/'APOGEE/sample.csv')
        absz_final = self.stars['zfinal'].abs()
        bin_edges = np.linspace(absz_final.min(), absz_final.max(), 100)
        # limit APOGEE data to VICE z-height range
        apogee_data = apogee_data[
            (apogee_data['GALZ'].abs() >= absz_final.min()) &
            (apogee_data['GALZ'].abs() <= absz_final.max())]
        # calculate weights to re-sample to APOGEE vertical distribution
        apogee_dist, _ = np.histogram(apogee_data['GALZ'].abs(), bins=bin_edges, 
                                      density=True)
        vice_dist, _ = np.histogram(absz_final, bins=bin_edges, density=True)
        absz_weights = apogee_dist / vice_dist
        # cut z-height values into above bins
        bin_cuts = pd.cut(absz_final, bins=bin_edges, labels=False, 
                          include_lowest=True)
        sample_weights = absz_weights[bin_cuts] / absz_weights[bin_cuts].sum()
        resample = sample_rows(self.stars.copy(), N, 
                               weights=sample_weights, seed=seed)
        if inplace:
            self.stars = resample
        else:
            return MultizoneStars(resample, name=self.name, 
                                  fullpath=self.fullpath, 
                                  zone_width=self.zone_width, 
                                  galr_lim=self.galr_lim, 
                                  absz_lim=self.absz_lim, 
                                  noisy=self.noisy)
    
    def scatter_plot(self, ax, xcol, ycol, color=None, cmap=None, norm=None, 
                     sampled=True, nsamples=10000, markersize=0.1, **kwargs):
        """
        Create a scatter plot of the given columns.
        
        parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw the scatter plot.
        xcol : str
            Name of column to plot on the x-axis.
        ycol : str
            Name of column to plot on the y-axis.
        color : str, optional
            Scatter plot color. If a column name, will color-code points based
            on the given colormap and normalization. The default is None.
        cmap : matplotlib colormap, optional
        norm : matplotlib normalization, optional
        sampled : bool, optional
            If True, randomly sample nsamples rows from the full DataFrame.
            The default is True.
        nsamples : int, optional
            Number of randomly sampled points to plot if sampled == True. The 
            default is 10000.
        markersize : float, optional
            Scatter plot marker size. The default is 0.1.
        kwargs passed to Axes.scatter()
        """
        if sampled and self.nstars > 0:
            stars = self.sample(nsamples)
        else:
            stars = self.stars
        # If zcol is not a valid column, don't color-code points
        if color in stars.columns:
            color = stars[color]
        # Scatter plot of stellar particles
        ax.scatter(stars[xcol], stars[ycol], c=color, s=markersize,
                   cmap=cmap, norm=norm, rasterized=True, edgecolor='none',
                   **kwargs)


    def binned_intervals(self, col, bin_col, bin_edges, quantiles=[0.16, 0.5, 0.84]):
        """
        Calculate quantiles of a parameter in bins of a secondary parameter.
        
        Parameters
        ----------
        col : str
            Data column corresponding to the first parameter, for which
            intervals will be calculated in each bin.
        bin_col : str
            Data column corresponding to the second parameter (typically an
            abundance, like "FE_H").
        bin_edges : array-like
            List or array of bin edges for the secondary parameter.
        quantiles : list, optional
            List of quantiles to calculate in each bin. The default is
            [0.16, 0.5, 0.84], corresponding to the median and +/- one
            standard deviation.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with each column corresponding to a quantile level
            and each row a bin in the specified secondary parameter, plus
            a final column "Mass" with the fraction of stellar mass in each bin.
        
        Notes
        -----
        Example: to calculate the median, 16th, and 84th percentile of stellar
        age in different bins of [Fe/H], use:
        mzs.binned_intervals('age', '[fe/h]', np.arange(-1.2, 0.7, 0.1),
                             quantiles=[0.16, 0.5, 0.84])
        """
        # Remove entries with no age estimate
        stars = self.stars.dropna(how='any')
        grouped = stars.groupby(pd.cut(stars[bin_col], bin_edges), observed=False)
        param_quantiles = []
        for q in quantiles:
            # Weight stellar populations by mass
            wq = lambda x: weighted_quantile(x, col, 'mstar', quantile=q)
            param_quantiles.append(grouped.apply(wq, include_groups=False))
        param_quantiles.append(grouped['mstar'].sum() / stars['mstar'].sum())
        df = pd.concat(param_quantiles, axis=1)
        df.columns = quantiles + ['mass_fraction']
        return df


    def mdf(self, col, bins=100, range=None, smoothing=0., density=True):
        """
        Generate a metallicity distribution function (MDF) for the given 
        abundance.

        Parameters
        ----------
        col : str
            Column for which to generate a distribution function.
        bins : int or sequence of scalars, optional
            If an int, defines the number of equal-width bins in the given
            range. If a sequence, defines the array of bin edges including
            the right-most edge. The default is 100.
        range : tuple, optional
            Range in the given column to bin. The default is None, which 
            corresponds to the entire range of data. If bins is provided as
            a sequence, range is ignored.
        smoothing : float, optional
            Width of boxcar smoothing to apply to MDF. If 0, no smoothing will
            be applied. The default is 0.
            
        Returns
        -------
        mdf : 1-D array
            Metallicity distribution function.
        bin_edges : 1-D array
            Bin edges of the MDF (length(mdf) + 1).
        """
        mdf, bin_edges = np.histogram(self(col), bins=bins, range=range, 
                                      weights=self('mstar'), density=density)
        if smoothing > 0:
            mdf = box_smooth(mdf, bin_edges, smoothing)
        return mdf, bin_edges
    
    def adf(self, bins=20, **kwargs):
        """
        Generate an age distribution function (ADF).
        
        Parameters
        ----------
        bins : int or sequence of scalars, optional
            If an int, defines the number of equal-width bins in the given
            range. If a sequence, defines the array of bin edges including
            the right-most edge. The default is 100.
        kwargs : dict, optional
            Keyword arguments passed to mean_stellar_mass
        
        Returns
        -------
        adf : 1-D array
            Age distribution function.
        bin_edges : 1-D array
            Bin edges of the ADF (length(adf) + 1).
        """
        bin_edges = np.linspace(0, self.end_time, bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bin_edges[1] - bin_edges[0]
        # Create dummy entries to count at least 0 mass at every age
        temp_df = pd.DataFrame({
            'age': bin_centers, 
            'mstar': np.zeros(bin_centers.shape)
        })
        stars = pd.concat([self.stars, temp_df])
        # stars['age'] = np.round(stars['age'], decimals=2)
        # Sum stellar mass in each bin
        mass_total, _ = np.histogram(stars['age'], bins=bin_edges, 
                                     weights=stars['mstar'])
        # Calculate remaining stellar mass today
        mass_remaining = mass_total * (1 - np.array(
            [vice.cumulative_return_fraction(age) for age in bin_centers]))
        # Average mass of a star of that particular age
        mass_average = np.array(
            [self.mean_stellar_mass(age, **kwargs) for age in bin_centers]
        )
        # Number of stars in each age bin
        nstars = np.around(mass_remaining / mass_average)
        # Fraction of stars in each age bin
        adf = nstars / (bin_width * nstars.sum())
        return adf, bin_edges

    
    def __str__(self):
        return self.stars.__str__()
        
    @property
    def name(self):
        """
        str
            Multizone output name, excluding ``.vice`` extension or parent dir.
        """
        return self._name
    
    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value
        else:
            raise TypeError('Attribute "name" must be a string. Got:', 
                            type(value))
    
    @property
    def fullpath(self):
        """
        pathlib.Path
            Full path to multizone output.
        """
        return self._fullpath
    
    @fullpath.setter
    def fullpath(self, value):
        if isinstance(value, (str, Path)):
            if '.vice' in str(value) and Path(value).is_dir():
                self._fullpath = Path(value)
            else:
                raise ValueError('Value is not a valid VICE output directory.')
        else:
            raise TypeError('Attribute "fullpath" must be a string or Path. Got:',
                            type(value))
    
    @property
    def stars(self):
        """
        pandas.DataFrame
            Complete star particle data.
        """
        return self._stars
    
    @stars.setter
    def stars(self, value):
        if isinstance(value, pd.DataFrame):
            self._stars = value
        else:
            raise TypeError('Attribute "stars" must be a DataFrame. Got:',
                            type(value))
    
    @property
    def zone_width(self):
        """
        float
            Width of each zone in kpc.
        """
        return self._zone_width
    
    @zone_width.setter
    def zone_width(self, value):
        if isinstance(value, float):
            self._zone_width = value
        else:
            raise TypeError('Attribute "zone_width" must be a float. Got:',
                            type(value))
            
    @property
    def galr_lim(self):
        """
        tuple
            Minimum and maximum bounds on the Galactic radius in kpc.
        """
        return self._galr_lim
    
    @galr_lim.setter
    def galr_lim(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                if all([isinstance(x, Number) for x in value]):
                    self._galr_lim = tuple(value)
                else:
                    raise TypeError('Each item in "galr_lim" must be a number.')
            else:
                raise ValueError('Attribute "galr_lim" must have length 2.')
        else:
            raise TypeError('Attribute "galr_lim" must be a tuple or list. Got:',
                            type(value))
            
    @property
    def absz_lim(self):
        """
        tuple
            Minimum and maximum bounds on the absolute z-height in kpc.
        """
        return self._absz_lim
    
    @absz_lim.setter
    def absz_lim(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                if all([isinstance(x, Number) for x in value]):
                    self._absz_lim = tuple(value)
                else:
                    raise TypeError('Each item in "absz_lim" must be a number.')
            else:
                raise ValueError('Attribute "absz_lim" must have length 2.')
        else:
            raise TypeError('Attribute "absz_lim" must be a tuple. Got:',
                            type(value))
    
    @property
    def noisy(self):
        """
        bool
            If True, indicates that forward-modeled observational uncertainty
            has been applied to certain columns of the VICE output.
        """
        return self._noisy
    
    @noisy.setter
    def noisy(self, value):
        if isinstance(value, bool):
            self._noisy = value
        else:
            raise TypeError('Attribute "noisy" must be a Boolean. Got:',
                            type(value))
            
    @property
    def end_time(self):
        """
        float
            Simulation end time in Gyr.
        """
        return self.stars['formation_time']
    
    @property
    def nstars(self):
        """
        int
            Total number of stars in multizone output.
        """
        return self.stars.shape[0]
    
    @staticmethod
    def mean_stellar_mass(age, imf=vice.imf.kroupa, mlr=vice.mlr.larson1974,
                          m_lower=0.08, m_upper=100, dm=0.01):
        """
        Calculate the mean mass of a stellar population of a given age.

        Parameters
        ----------
        age : float
            Stellar age in Gyr
        imf : <function>, optional
            Initial mass function which takes mass in solar masses as an argument.
            The default is vice.imf.kroupa
        mlr : <function>, optional
            Mass-lifetime relation which takes age in Gyr as an argument. The
            default is vice.mlr.larson1974
        m_lower : float, optional
            Lower mass limit on IMF in solar masses. The default is 0.08
        m_upper : float, optional
            Upper mass limit on IMF in solar masses. The default is 100
        dm : float, optional
            IMF integration step in solar masses. The default is 0.01

        Returns
        -------
        float
            Mean mass of stars with lifetime greater than or equal to the given age
            weighted by the IMF
        """
        m_max = min((mlr(age, which='age'), m_upper))
        masses = np.arange(m_lower, m_max + dm, dm)
        dndm = np.array([imf(m) for m in masses])
        weighted_mean = np.average(masses, weights=dndm)
        return weighted_mean


if __name__ == '__main__':
    main()
