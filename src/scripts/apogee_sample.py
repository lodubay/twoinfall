"""
This file contains the class APOGEESample which is used in importing and 
handling the APOGEE sample and related datasets. If run as a script, it will 
re-generate the main sample file at src/data/APOGEE/sample.csv.
"""

from numbers import Number
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
import paths
from utils import fits_to_pandas, box_smooth, galactic_to_galactocentric, quad_add, \
    decode, split_multicol, contour_levels_2D
from stats import skewnormal_mode_sample, jackknife_summary_statistic, kde2D
from _globals import RANDOM_SEED

# Sample galactocentric coordinate bounds
GALR_LIM = (3., 15.)
ABSZ_LIM = (0., 2.)
# Data file names
ALLSTAR_FNAME = 'allStarLite-dr17-synspec_rev1.fits'
LEUNG23_FNAME = 'nn_latent_age_dr17.csv'
# Coefficients for [C/N] age fit polynomial
CN_AGE_COEF = np.array([-1.90931538,  0.75218328,  0.25786775,  0.22440967, 
                        -0.32223068, 9.99041179])
# List of columns to include in the final sample
SAMPLE_COLS = ['APOGEE_ID', 'RA', 'DEC', 'GALR', 'GALPHI', 'GALZ', 'SNREV',
               'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'O_H', 'O_H_ERR', 
               'FE_H', 'FE_H_ERR', 'O_FE', 'O_FE_ERR', 'FE_O', 'FE_O_ERR',
               'C_N', 'C_N_ERR', 'CN_AGE', 'CN_AGE_ERR', 'CN_LOG_AGE',
               'CN_LOG_AGE_ERR',
               'L23_AGE', 'L23_AGE_ERR', 'L23_LOG_AGE', 'L23_LOG_AGE_ERR']

def main():
    sample_df = APOGEESample.generate(verbose=True)

class APOGEESample:
    """
    Contains data from the APOGEE sample and related functions.
    
    Notes
    -----
    In almost all instances, the user should initialize with the
    APOGEESample.load() or APOGEESample.generate() classmethods.
    
    Attributes
    ----------
    data : pandas.DataFrame
        The APOGEE sample data.
    data_dir : pathlib.Path
        Path to the directory containing APOGEE sample and data files.
    galr_lim : tuple of floats
        The minimum and maximum galactocentric radius of the sample in kpc.
    absz_lim : tuple of floats
        The minimum and maximum absolute midplane distance of the sample in kpc.
    nstars : int
        Total number of stars in the sample.
    nstars_ages : int
        Number of stars in the sample with age estimates.
    
    Calling
    -------
    cols : str or list of strings, optional
        If an empty list, returns the entire DataFrame. If a string, returns
        that column of the DataFrame. If a list of strings, returns a subset
        of those columns in the DataFrame. The default is [].
    
    Methods
    -------
    generate : classmethod
        Generate the APOGEE sample from scratch with selection cuts and ages.
    load : classmethod
        Load the APOGEE sample from the data directory.
    kde2D : instancemethod
        Generate a 2-dimensional kernel density estimate (KDE).
    kde2D_path : instancemethod
        File name for saving the 2-D KDE.
    mdf : instancemethod
        Calculate the metallicity distribution function (MDF).
    plot_kde2D_contours : instancemethod
        Plot contours representing a 2D kernel density estimate.
    region : instancemethod
        Select targets within the given Galactocentric region.
    join_latent_ages : staticmethod
        Join APOGEE sample with ages from Leung et al. (2023).
    quality_cuts : staticmethod
        Apply sample quality cuts to APOGEE data.
    """
    def __init__(self, data, data_dir=paths.data/'APOGEE',
                 galr_lim=GALR_LIM, absz_lim=ABSZ_LIM):
        self.data = data
        self.data_dir = data_dir
        self.galr_lim = galr_lim
        self.absz_lim = absz_lim
    
    def __call__(self, cols=[]):
        """
        Return the ``data`` dataframe or a subset of the dataframe.
        
        Parameters
        ----------
        cols : str or list of strings, optional
            If an empty list, returns the entire DataFrame. If a string, returns
            that column of the DataFrame. If a list of strings, returns a subset
            of those columns in the DataFrame. The default is [].
            
        Returns
        -------
        pandas.DataFrame or pandas.Series
            Star particle data or subset of that data.
        """
        if cols == []:
            return self.data
        else:
            # Error handling
            if isinstance(cols, str):
                if cols not in self.data.columns:
                    raise ValueError('Parameter "cols" must be an element ' + \
                                     'of data.columns.')
            elif isinstance(cols, list):
                if all([isinstance(c, str) for c in cols]):
                    if not all([c in self.data.columns for c in cols]):
                        raise ValueError('Each element of "cols" must be ' + \
                                         'an element of data.columns.')
                else:
                    raise TypeError('Each element of "cols" must be a string.')
            else:
                raise TypeError('Parameter "cols" must be a string or list ' +\
                                'of strings.')
            return self.data[cols]

    @classmethod
    def generate(cls, name='sample.csv', data_dir=paths.data/'APOGEE', 
                 verbose=False):
        """
        Generate the APOGEE sample with selection cuts and ages.
        
        Parameters
        ----------
        name : str, optional
            Name of CSV file containing sample data. The default is 'sample.csv'.
        data_dir : str or pathlib.Path, optional
            The parent directory containing APOGEE data files. The default is
            '../data/APOGEE/'.
        verbose : bool, optional
            Whether to print verbose output to terminal. The default is False.
        """
        if verbose:
            print('Importing APOGEE catalogs and generating sample...')
        # Make data directory if needed
        if not Path(data_dir).is_dir():
            data_dir.mkdir(parents=True)
        # Get APOGEE data from the DR17 allStar file
        apogee_catalog_path = data_dir / ALLSTAR_FNAME
        if not apogee_catalog_path.is_file():
            # Download DR17 allStar file from SDSS server
            if verbose: 
                print('Downloading allStar file (this will take a few minutes)...')
            url_write('https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/%s' \
                      % ALLSTAR_FNAME, savedir=data_dir)
        if verbose: print('Importing allStar file...')
        apogee_catalog = fits_to_pandas(apogee_catalog_path, hdu=1)
        # Add ages from row-matched datasets BEFORE any cuts
        # Add ages from Leung et al. (2023)
        leung23_catalog_path = data_dir / LEUNG23_FNAME
        if not leung23_catalog_path.is_file():
            # Download Leung+ 2023 data from GitHub
            if verbose:
                print('Downloading Leung et al. (2023) age data...')
            url_write('https://raw.githubusercontent.com/henrysky/astroNN_ages/main/%s.gz' \
                      % LEUNG23_FNAME, savedir=data_dir)
        if verbose: print('Joining with latent age catalog...')
        leung23_catalog = pd.read_csv(leung23_catalog_path)
        full_catalog = APOGEESample.join_latent_ages(apogee_catalog, leung23_catalog)
        if verbose: print('Implementing quality cuts...')
        sample = APOGEESample.quality_cuts(full_catalog)
        # Calculate galactocentric coordinates based on galactic l, b and Gaia dist
        galr, galphi, galz = galactic_to_galactocentric(
            sample['GLON'], sample['GLAT'], sample['GAIAEDR3_R_MED_PHOTOGEO']/1000
        )
        sample['GALR'] = galr # kpc
        sample['GALPHI'] = galphi # deg
        sample['GALZ'] = galz # kpc
        # Limit by galactocentric radius and z-height
        sample = sample[(sample['GALR'] >= GALR_LIM[0]) & 
                        (sample['GALR'] < GALR_LIM[1]) &
                        (sample['GALZ'].abs() >= ABSZ_LIM[0]) &
                        (sample['GALZ'].abs() < ABSZ_LIM[1])]
        sample.reset_index(inplace=True, drop=True)
        # Add column for [O/H]
        sample['O_H'] = sample['O_FE'] + sample['FE_H']
        sample['O_H_ERR'] = sample['O_FE_ERR'] # [X/Fe] and [X/H] errors are the same
        # Add column for [Fe/O]
        sample['FE_O'] = -sample['O_FE']
        sample['FE_O_ERR'] = sample['O_FE_ERR']
        # Add column for [C/N]
        sample['C_N'] = sample['C_FE'] - sample['N_FE']
        sample['C_N_ERR'] = quad_add(sample['C_FE_ERR'], sample['N_FE_ERR'])
        # Calculate [C/N]-based ages
        sample = APOGEESample.generate_cn_ages(sample)
        # Drop unneeded columns
        data = sample[SAMPLE_COLS].copy()
        # Write sample to csv file
        sample_file_path = data_dir / name
        if verbose:
            print('Saving sample data to %s...' % sample_file_path)
        data.to_csv(sample_file_path, index=False)
        if verbose:
            print('Done.')
        return cls(data, data_dir=data_dir, galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    
    @classmethod
    def load(cls, name='sample.csv', data_dir=paths.data/'APOGEE'):
        """
        Load the APOGEE sample from the data directory.
        
        Parameters
        ----------
        name : str, optional
            Name of CSV file containing sample data. The default is 'sample.csv'.
        data_dir : str or pathlib.Path, optional
            The parent directory containing APOGEE data files. The default is
            '../data/APOGEE/'.

        Returns
        -------
        APOGEESample instance
        """
        sample_file_path = data_dir / name
        try:
            data = pd.read_csv(sample_file_path)
        except FileNotFoundError:
            raise FileNotFoundError('APOGEE sample file not found. Please run \
``python apogee_sample.py`` to generate it first.')
        return cls(data, data_dir=data_dir, galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)

    def kde2D(self, xcol, ycol, bandwidth=0.03, overwrite=False, **kwargs):
        """
        Generate 2-dimensional kernel density estimate (KDE) of APOGEE data, 
        or import previously saved KDE if it already exists.
        
        Parameters
        ----------
        xcol : str
            Name of column with x-axis data
        ycol : str
            Name of column with y-axis data
        bandwidth : float
            Kernel density estimate bandwidth. A larger number will produce
            smoother contour lines. The default is 0.03.
        overwrite : bool
            If True, force re-generate the 2D KDE and save the output.
        **kwargs passed to utils.kde2D()
        
        Returns
        -------
        xx, yy, logz: tuple of numpy.array
            Outputs of utils.kde2D()
        """    
        # Path to save 2D KDE for faster plot times
        path = self.kde2D_path(xcol, ycol)
        if path.exists() and not overwrite:
            xx, yy, logz = read_kde(path)
        else:
            xx, yy, logz = kde2D(self.data[xcol], self.data[ycol], bandwidth, **kwargs)
            save_kde(xx, yy, logz, path)
        return xx, yy, logz

    def kde2D_path(self, xcol, ycol):
        """
        Generate file name for the KDE of the given region.
        
        Parameters
        ---------
        xcol : str
            Name of column with x-axis data
        ycol : str
            Name of column with y-axis data
        """
        kde_dir = '_'.join([''.join(xcol.split('_')).lower(),
                            ''.join(ycol.split('_')).lower()])
        filename = 'r%s-%s_z%s-%s.dat' % (self.galr_lim + self.absz_lim)
        return self.data_dir / 'kde' / kde_dir / filename
    
    def mdf(self, col='FE_H', bins=100, range=None, smoothing=0., density=True):
        """
        Calculate the metallicity distribution function (MDF).
        
        Parameters
        ----------
        col : str, optional
            Column name of desired abundance data. The default is 'FE_H'.
        bins : int or sequence of scalars, optional
            If an int, defines the number of equal-width bins in the given
            range. If a sequence, defines the array of bin edges including
            the right-most edge. The default is 100.
        range : tuple, optional
            Range in the given column to bin. The default is None, which 
            corresponds to the entire range of data. If bins is provided as
            a sequence, range is ignored.
        smoothing : float, optional
            Width of boxcar smoothing in x-axis units. If 0, the distribution will
            not be smoothed. The default is 0.
        
        Returns
        -------
        mdf : numpy.ndarray
            Boxcar-smoothed MDF.
        bin_edges : numpy.ndarray
            [Fe/H] bins including left and right edges, of length len(mdf)+1.
        """
        mdf, bin_edges = np.histogram(self.data[col], bins=bins, range=range, 
                                      density=density)
        if smoothing > 0.:
            mdf = box_smooth(mdf, bin_edges, smoothing)
        return mdf, bin_edges
        
    def plot_kde2D_contours(self, ax, xcol, ycol, enclosed=[0.8, 0.3],
                            c='r', lw=0.5, ls=['--', '-'],
                            plot_kwargs={}, **kwargs):
        """
        Plot 2D density contours from the kernel density estimate for the
        given columns.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the scatter plot.
        xcol : str
            Name of column to plot on the x-axis.
        ycol : str
            Name of column to plot on the y-axis.
        enclosed : list, optional
            List of probabilities enclosed by the contour levels, ordered
            from highest probability (lowest contour level) to lowest (highest).
            The default is [0.8, 0.3].
        c : str or matplotlib color or list of previous, optional
            Color(s) of each contour line. The default is 'r'.
        lw : float or list of floats, optional
            Line widths corresponding to each contour line. The default is 0.5.
        ls : str or list of str, optional
            Line styles of each contour. If a list, length must be equal
            to the length of 'enclosed'. The default is ['--', '-'].
        **kwargs passed to `kde2D()`.
        """
        xx, yy, logz = self.kde2D(xcol, ycol, **kwargs)
        # scale the linear density to the max value
        scaled_density = np.exp(logz) / np.max(np.exp(logz))
        # contour levels at 1 and 2 sigma
        levels = contour_levels_2D(scaled_density, enclosed=enclosed)
        ax.contour(xx, yy, scaled_density, levels, colors=c,
                   linewidths=lw, linestyles=ls, **plot_kwargs)
        
    def filter(self, filterdict, inplace=False):
        """
        Filter data by the given parameter bounds.
        
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
        data_copy = self.data.copy()
        if isinstance(filterdict, dict):
            for key in filterdict.keys():
                # Error handling
                if key not in self.data.columns:
                    raise ValueError('Keys in "filterdict" must be data \
column names.')
                elif not isinstance(filterdict[key], tuple):
                    raise TypeError('Each value in "filterdict" must be a \
tuple of length 2.')
                elif len(filterdict[key]) != 2:
                    raise ValueError('Each value in "filterdict" must be a \
tuple of length 2.')
                elif not all([isinstance(val, Number) or val is None \
                              for val in filterdict[key]]):
                    raise TypeError('Each element of the tuple must be \
numeric or NoneType.')
                else:
                    colmin, colmax = filterdict[key]
                    if colmin is not None:
                        data_copy = data_copy[data_copy[key] >= colmin]
                    if colmax is not None:
                        data_copy = data_copy[data_copy[key] < colmax]
            if inplace:
                self.data = data_copy
            else:
                return APOGEESample(data_copy, data_dir=self.data_dir, 
                                    galr_lim=self.galr_lim, 
                                    absz_lim=self.absz_lim)
        else:
            raise TypeError('Parameter "filterdict" must be a dict. Got:',
                            type(filterdict))
    
    def region(self, galr_lim=(3, 15), absz_lim=(0, 2), inplace=False):
        """
        Select targets within the given Galactocentric region.
        
        Parameters
        ----------
        galr_lim : tuple
            Minimum and maximum Galactic radius in kpc. The default is (0, 20).
        absz_lim : tuple
            Minimum and maximum of the absolute value of z-height in kpc. The
            default is (0, 5).
        inplace : bool, optional
            If True, update the current class instance. If False, return a 
            new class instance with the limited subset. The default is False.
        
        Returns
        -------
        APOGEESample instance or None
        """
        galr_min, galr_max = galr_lim
        absz_min, absz_max = absz_lim
        # Select subset
        subset = self.data[(self.data['GALR'] >= galr_min) &
                           (self.data['GALR'] < galr_max) &
                           (self.data['GALZ'].abs() >= absz_min) &
                           (self.data['GALZ'].abs() < absz_max)].copy()
        subset.reset_index(inplace=True, drop=True)
        if inplace:
            self.data = subset
            self.galr_lim = galr_lim
            self.absz_lim = absz_lim
        else:
            return APOGEESample(subset, data_dir=self.data_dir, 
                                galr_lim=galr_lim, absz_lim=absz_lim)


    def binned_intervals(self, col, bin_col, bin_edges, 
                         quantiles=[0.16, 0.5, 0.84]):
        """
        Calculate stellar age quantiles in bins of a secondary parameter.
        
        Parameters
        ----------
        col : str
            Data column corresponding to the first parameter, for which the
            intervals will be calculated in each bin.
        bin_col : str
            Data column corresponding to the second (binning) parameter.
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
            a final column "Count" with the number of targets in each bin.
        """
        # Remove entries with no age estimate
        data = self.data.dropna(subset=col)
        param_grouped = data.groupby(pd.cut(data[bin_col], bin_edges), observed=False)[col]
        param_quantiles = []
        for q in quantiles:
            param_quantiles.append(param_grouped.quantile(q))
        param_quantiles.append(param_grouped.count())
        df = pd.concat(param_quantiles, axis=1)
        df.columns = quantiles + ['count']
        return df

    def binned_modes(self, col, bin_col, bin_edges, 
                     sample_bins=np.linspace(-3, 2, 1001)):
        """
        Calculate the mode of a parameter in bins of a secondary parameter.
        
        Parameters
        ----------
        col : str
            Data column corresponding to the first parameter, for which the
            intervals will be calculated in each bin.
        bin_col : str
            Data column corresponding to the second (binning) parameter.
        bin_edges : array-like
            List or array of bin edges for the secondary parameter.
        sample_bins : array-like, optional
            Bins on the primary parameter.
        
        Returns
        -------
        """
        # Remove entries with no age estimate
        data = self.data.dropna(subset=col)
        grouped = data.groupby(
            pd.cut(data[bin_col], bin_edges), observed=False
        )[col]
        modes = grouped.apply(
            skewnormal_mode_sample, include_groups=False, bins=sample_bins
        )
        jackknife_mode = lambda x: jackknife_summary_statistic(
            x, skewnormal_mode_sample, n_resamples=25, seed=RANDOM_SEED, bins=sample_bins
        )
        errors = grouped.apply(
            jackknife_mode, 
            # args=(skewnormal_mode_sample), 
            # n_resamples=10, seed=RANDOM_SEED, bins=sample_bins
        )
        counts = grouped.count()
        df = pd.concat([modes, errors, grouped.count()], axis=1)
        df.columns = ['mode', 'error', 'count']
        return df
                    
    
    def count_notna(self, col):
        """
        Count the number of stars in the dataset with real values for the
        given parameter.

        Parameters
        ----------
        col : str
            Data column.
        
        Returns
        -------
        int
            Number of stars in the sample with age estimates.
        """
        return self.data[self.data[col].notna()].shape[0]
        
        
    @property
    def data(self):
        """
        Type: pandas.DataFrame
            The APOGEE sample data.
        """
        return self._data
    
    @data.setter
    def data(self, value):
        if isinstance(value, pd.DataFrame):
            if set(SAMPLE_COLS) == set(value.columns):
                self._data = value
            else:
                raise ValueError('Column names do not match expected.')
        else:
            raise TypeError('APOGEE sample data must be in the form of a ' +
                            'pandas DataFrame.')
            
    @property
    def data_dir(self):
        """
        Type: pathlib.Path
            Path to the directory containing APOGEE data files.
        """
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, value):
        if isinstance(value, (str, Path)):
            if Path(value).is_dir():
                self._data_dir = value
            else:
                raise ValueError('%s is not a directory.' % value)
        else:
            raise TypeError('Attribute data_dir must be a string or path.' +
                            'Got: %s' % type(value))
            
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
    def nstars(self):
        """
        int
            Total number of stars in the sample.
        """
        return self.data.shape[0]
    
    @staticmethod
    def generate_cn_ages(apogee_df):
        """
        Calculate [C/N]-based ages for a subset of the APOGEE sample.
        
        Parameters
        ----------
        apogee_df : pandas.DataFrame
            Full APOGEE dataset (post-cuts okay), must include [C/N] data.
        
        Returns
        -------
        pandas.DataFrame
            [C/N]-based ages and errors indexed on APOGEE IDs.
        
        Reference
        ---------
        Roberts, J. et al (in prep)
        """
        # Hard edge cuts
        goldregion = apogee_df[
            (apogee_df['FE_H'] >= -0.9) & (apogee_df['FE_H'] < 0.45) & 
            (apogee_df['LOGG'] >= 1.5) & (apogee_df['LOGG'] < 3.26) &
            (apogee_df['C_N'] >= -0.75) & (apogee_df['C_N'] < 1.0) &
            (apogee_df['TEFF'] >= 4000) & (apogee_df['TEFF'] < 5200) &
            (apogee_df['C_N_ERR'] < 0.08)
        ].copy()
        # Get evolutionary state
        goldregion = evol_state(goldregion)
        LRGB = goldregion[
            (goldregion['EVOL_STATE'] == 1) & (goldregion['LOGG'] >= 2.5)
        ]
        URGB = goldregion[
            (goldregion['EVOL_STATE'] == 1) & (goldregion['LOGG'] < 2.5)
        ]
        RC = goldregion[goldregion['EVOL_STATE'] == 2]
        # Remove [Fe/H] < -0.4 for URGB and RC stars, then re-merge
        cn_age_region = pd.concat([
            LRGB, URGB[URGB['FE_H'] >= -0.4], RC[RC['FE_H'] >= -0.4]
        ])
        # Inflate abundance uncertainties - reported in APOGEE are too small
        # Cao & Pinsonneault (2025); Pinsonneault et al. (2025)
        cfe_err = 3.0728 * cn_age_region['C_FE_ERR']
        nfe_err = 2.7109 * cn_age_region['N_FE_ERR']
        cn_err = np.sqrt(cfe_err**2 + nfe_err**2 - 0.0946 * cfe_err * nfe_err)
        feh_err = 0.05 * np.ones(cn_age_region.shape[0])
        cn_log_age, cn_log_age_err = recover_age_quad(
            cn_age_region['C_N'].values, 
            cn_age_region['FE_H'].values, 
            CN_AGE_COEF,
            cn_err = cn_err,
            feh_err = feh_err
        )
        # Convert years -> Gyr
        cn_age_region['CN_LOG_AGE'] = cn_log_age - 9.
        cn_age_region['CN_LOG_AGE_ERR'] = 1.4 * cn_log_age_err # inflate by 40% per Jack
        cn_age_region['CN_AGE'] = 10 ** cn_age_region['CN_LOG_AGE']
        cn_age_region['CN_AGE_ERR'] = cn_age_region['CN_AGE'] * np.log(10) * cn_age_region['CN_LOG_AGE_ERR']
        apogee_df = apogee_df.join(cn_age_region[
            ['CN_AGE', 'CN_AGE_ERR', 'CN_LOG_AGE', 'CN_LOG_AGE_ERR']
        ])
        return apogee_df
    
    @staticmethod
    def join_latent_ages(apogee_df, leung23_df):
        """
        Join ages from Leung et al. (2023) to the row-matched APOGEE dataset.
        
        Parameters
        ----------
        apogee_df : pandas.DataFrame
            Full APOGEE dataset without cuts
        leung23_df : pandas.DataFrame
            Dataset from Leung et al. (2023)
        
        Returns
        -------
        joined_df : pandas.DataFrame
            APOGEE dataset with latent ages
        """
        cols = ['LogAge', 'LogAge_Error', 'Age', 'Age_Error']
        latent_ages = leung23_df[cols].copy()
        latent_ages.columns = ['L23_LOG_AGE', 'L23_LOG_AGE_ERR', 
                               'L23_AGE', 'L23_AGE_ERR']
        # Limit to stars with <40% age uncertainty per recommendation
        frac_err = latent_ages['L23_AGE_ERR'] / latent_ages['L23_AGE']
        latent_ages.where(frac_err < 0.4, inplace=True)
        joined = apogee_df.join(latent_ages)
        return joined
    
    @staticmethod
    def quality_cuts(df):
        """
        Make quality cuts on the APOGEE catalog.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Full APOGEE catalog
        
        Returns
        -------
        pandas.DataFrame
        """
        # Limit to main red star sample
        df = df[df['EXTRATARG'] == 0]
        # Weed out bad flags
        fatal_flags = (2**23) # STAR_BAD
        df = df[df['ASPCAPFLAG'] & fatal_flags == 0]
        # Cut low-S/N targets
        df = df[df['SNREV'] > 80]
        # Limit to giants
        df = df[(df['LOGG'] > 1) & (df['LOGG'] < 3.8) & 
                (df['TEFF'] > 3500) & (df['TEFF'] < 5500)]
        # Replace NaN stand-in values with NaN
        df.replace(99.999, np.nan, inplace=True)
        # Limit to stars with measurements of both [Fe/H] and [O/Fe]
        df.dropna(subset=['FE_H', 'O_FE'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def allStar_to_pandas(path):
        """
        Read the allStar fits file and convert to a pandas DataFrame.
        
        Parameters
        ----------
        path : pathlib.Path or string
            Path to allStar fits file.
        
        Returns
        -------
        pandas.DataFrame
        """
        # Read FITS file into astropy table
        table = Table.read(path, format='fits', hdu=1)
        # Filter out multidimensional columns
        cols = [name for name in table.colnames if len(table[name].shape) <= 1]
        # Convert byte-strings to ordinary strings and convert to pandas
        apogee_catalog = decode(table[cols].to_pandas())
        # Get multidimensional column names from HDU3
        array_info = Table.read(path, format='fits', hdu=3)
        param_symbols = [f'{s.decode()}_RAW' for s in array_info['PARAM_SYMBOL'][0]]
        # Get uncalibrated values from FPARAM multidimensional column
        fparams = split_multicol(table['FPARAM'], names=param_symbols)
        apogee_catalog = apogee_catalog.join(fparams)
        return apogee_catalog


def recover_age_quad(cn_arr, feh_arr, params, cn_err=[], feh_err=[]):
    """
    Compute stellar ages via polynomial fit to [C/N] and [Fe/H].
    
    Parameters
    ----------
    cn_arr : array-like
        Array of stellar [C/N] abundances.
    feh_arr : array-like
        Array of stellar [Fe/H] abundances. Must be same length as cn_arr.
    params : array-like
        Polynomial fit coefficients. Must have length 6.
    
    Returns
    -------
    ages: array-like
        Array of log10(stellar ages in years).
    age_errors: array-like
        Array of error in log-age.
    
    Notes
    -----
    Thanks Jack.
    """
    assert len(cn_arr) == len(feh_arr)
    c2,c1,f2,f1,c1f1,b = params
    #check for stars with [C/N] past the parabola peak
    maxcn = (c1+c1f1*feh_arr)/(-2*c2)
    badcn = cn_arr > maxcn
    # Calculate ages
    ages = (c2*cn_arr**2) + (c1*cn_arr) + (f2*feh_arr**2) + (f1*feh_arr) + \
        (c1f1*feh_arr*cn_arr)+b 
    #for stars with c/n past the parabola peak, use the parabola peak instead
    ages[badcn] = (c2*maxcn[badcn]**2) + (c1*maxcn[badcn]) + \
        (f2*feh_arr[badcn]**2) + (f1*feh_arr[badcn]) + \
        (c1f1*feh_arr[badcn]*maxcn[badcn]) + b 
    # Generic age errors of 1 Gyr
    age_errors = 1e9 / (np.log(10) * 10 ** ages)
    # Propagate errors
    if len(cn_err) > 0 and len(feh_err) > 0:
        age_errors = np.sqrt(
            cn_err**2 * (2*c2*cn_arr + c1 + c1f1*feh_arr)**2 +
            feh_err**2 * (2*f2*feh_arr + f1 * c1f1*cn_arr)**2
        )
    return ages, age_errors


def evol_state(dataplot, verbose=False):
    """
    Using Warfield's APOK2 paper to separate RGB from RC stars
    1 is RGB, 2 is RC

    References
    ----------
    Warfield et al. (2024), ApJ 167:208
    """
    #Calculate Reference Temperature
    alp = 4427.1779
    bet = -399.5105
    gam = 553.1705
    Tref = alp + (bet*dataplot['FE_H_SPEC']) + (gam*(dataplot['LOGG_SPEC']-2.5))
   
    #Calculate Equation A4 value
    a = 0.05915
    b = 0.003455
    c = 155.1
    criterion = a - (b*((c*dataplot['FE_H_SPEC']) + dataplot['TEFF_SPEC'] - Tref)) - (dataplot['C_FE_SPEC'] - dataplot['N_FE_SPEC'])
   
    #Apply Criterion
    loggrgb = dataplot['LOGG'] < 2.3 #These stars always RGB
    critrgb = criterion > 0 #A4 > 0 is RGB (Swapped from the paper)
    rgb = np.logical_or(loggrgb,critrgb) #Either case makes RGB
    critrc = criterion <= 0 #A4 < 0 is RC (Swapped from the paper)
    rc = np.logical_and(np.invert(loggrgb),critrc) #Must be A4<0 and not urgb
    if verbose: #double check the counts to make sure nothing got missed or double counted
        print(f"{sum(rc)} RC stars and {sum(rgb)} RGB stars")
        print(f"{dataplot.shape[0]} Total stars: Difference of {dataplot.shape[0] - (sum(rc) + sum(rgb))}")
    #Create output flags
    flagger = np.zeros(dataplot.shape[0])
    flagger[rgb] = flagger[rgb] + 1
    flagger[rc] = flagger[rc] + 2
    dataplot['EVOL_STATE'] = flagger
    return dataplot


def read_kde(path):
    """
    Read a text file generated by save_kde()
    """
    arr2d = np.genfromtxt(path)
    nrows = int(arr2d.shape[0]/3)
    xx = arr2d[:nrows]
    yy = arr2d[nrows:2*nrows]
    logz = arr2d[2*nrows:]
    return xx, yy, logz


def save_kde(xx, yy, logz, path):
    """
    Generate a text file containing the KDE of the given region along with its
    corresponding x and y coordinates.
    """
    if not path.parents[0].is_dir():
        path.parents[0].mkdir(parents=True)
    with open(path, 'w') as f:
        for arr in [xx, yy, logz]:
            f.write('#\n')
            np.savetxt(f, arr)


def url_write(url, savedir=paths.data):
    """
    Retrieve a text file from the provided URL, decompressing if necessary.
    
    Parameters
    ----------
    url : str
        URL of text file to download.
    savedir : str or pathlib.Path, optional
        Directory to save the downloaded file. The default is ../data/APOGEE.
    """
    # These packages are only needed when building the sample from scratch.
    import gzip
    import urllib.request
    fname = url.split('/')[-1]
    with urllib.request.urlopen(url) as response:
        resp = response.read()
    # Decompress file if needed
    if fname[-3:] == '.gz':
        resp = gzip.decompress(resp)
        fname = fname[:-3]
    with open(Path(savedir) / fname, 'wb') as f:
        f.write(resp)
        

if __name__ == '__main__':
    main()
    