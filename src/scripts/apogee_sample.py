"""
This file contains the class APOGEESample which is used in importing and 
handling the APOGEE sample and related datasets. If run as a script, it will 
re-generate the main sample file at src/data/APOGEE/sample.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import paths
from utils import fits_to_pandas, box_smooth, kde2D, galactic_to_galactocentric

# Data file names
ALLSTAR_FNAME = 'allStarLite-dr17-synspec_rev1.fits'
LEUNG23_FNAME = 'nn_latent_age_dr17.csv'
# List of columns to include in the final sample
SAMPLE_COLS = ['APOGEE_ID', 'RA', 'DEC', 'GALR', 'GALPHI', 'GALZ', 'SNREV',
               'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FE_H', 'FE_H_ERR',
               'O_FE', 'O_FE_ERR', 'LATENT_AGE', 'LATENT_AGE_ERR', 
               'LOG_LATENT_AGE', 'LOG_LATENT_AGE_ERR']

class APOGEESample:
    """
    Contains data from the APOGEE sample and related functions.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The APOGEE sample data.
    data_dir : str or pathlib.Path, optional
        The parent directory containing APOGEE data files. The default is
        '../data/APOGEE/'.
    
    Attributes
    ----------
    data : pandas.DataFrame
        The APOGEE sample data.
    data_dir : pathlib.Path
        Path to the directory containing APOGEE sample and data files.
    
    Calling
    -------
    
    Functions
    ---------
    region(galr_lim=(3, 15), absz_lim=(0, 2), inplace=False)
    """
    def __init__(self, data, data_dir=paths.data/'APOGEE'):
        self.data = data
        self.data_dir = data_dir
    
    def __call__(self, cols=[]):
        """
        Return the ``data`` dataframe or a subset of the dataframe.
        
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
            return self.data
        else:
            # Error handling
            if isinstance(cols, str):
                if cols not in self.data.columns:
                    raise ValueError('Parameter "cols" must be an element ' + \
                                     'of stars.columns.')
            elif isinstance(cols, list):
                if all([isinstance(c, str) for c in cols]):
                    if not all([c in self.data.columns for c in cols]):
                        raise ValueError('Each element of "cols" must be ' + \
                                         'an element of stars.columns.')
                else:
                    raise TypeError('Each element of "cols" must be a string.')
            else:
                raise TypeError('Parameter "cols" must be a string or list ' +\
                                'of strings.')
            return self.data[cols]
    
    def mdf(self, col='FE_H', bins=100, range=None, smoothing=0.):
        """
        Calculate the MDF in [Fe/H] of a region of APOGEE data.
        
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
                                      density=True)
        if smoothing > 0.:
            mdf = box_smooth(mdf, bin_edges, smoothing)
        return mdf, bin_edges
    
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
        else:
            return APOGEESample(subset, data_dir=self.data_dir)
        
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
    
    @classmethod
    def load(cls, name='sample.csv', data_dir=paths.data/'APOGEE', 
             verbose=False, overwrite=False):
        """
        Load the APOGEE sample from the data directory, or generate from scratch.
        
        Parameters
        ----------
        name : str, optional
            Name of CSV file containing sample data. The default is 'sample.csv'.
        data_dir : str or pathlib.Path, optional
            The parent directory containing APOGEE data files. The default is
            '../data/APOGEE/'.
        verbose : bool, optional
            Whether to print verbose output to terminal. The default is False.
        overwrite : bool, optional
            If True, re-generates the sample file even if it already exists.
            The default is False.

        Returns
        -------
        APOGEESample instance
        """
        sample_file_path = data_dir / name
        if sample_file_path.exists() and not overwrite:
            if verbose:
                print('Reading APOGEE sample from', sample_file_path)
            data = pd.read_csv(sample_file_path)
        else:
            if verbose:
                print('Sample file at', sample_file_path, 'not found.\n' + \
                      'Importing APOGEE catalog and generating sample...')
            # Make data directory if needed
            if not Path(data_dir).is_dir():
                data_dir.mkdir(parents=True)
            data = cls.generate(verbose=verbose)
            data.to_csv(sample_file_path, index=False)
            if verbose:
                print('Done.')
        return cls(data, data_dir=data_dir)

    @staticmethod
    def generate(data_dir=paths.data/'APOGEE', verbose=False):
        """
        Generate the APOGEE sample with selection cuts and ages.
        
        Parameters
        ----------
        data_dir : str or pathlib.Path, optional
            The parent directory containing APOGEE data files. The default is
            '../data/APOGEE/'.
        verbose : bool, optional
            Whether to print verbose output to terminal. The default is False.
        """
        # Get APOGEE data from the DR17 allStar file
        apogee_catalog_path = data_dir / ALLSTAR_FNAME
        if not apogee_catalog_path.is_file():
            # Download DR17 allStar file from SDSS server
            if verbose: 
                print('Downloading allStar file (this will take a few minutes)...')
            # get_allStar_dr17()
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
            # get_Leung2023_ages()
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
        sample = sample[(sample['GALR'] > 3.) & (sample['GALR'] < 15.) &
                        (sample['GALZ'].abs() < 2)]
        sample.reset_index(inplace=True, drop=True)
        # Drop unneeded columns
        return sample[SAMPLE_COLS].copy()
    
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
        latent_ages.columns = ['LOG_LATENT_AGE', 'LOG_LATENT_AGE_ERR', 
                               'LATENT_AGE', 'LATENT_AGE_ERR']
        # Limit to stars with <40% age uncertainty per recommendation
        frac_err = latent_ages['LATENT_AGE_ERR'] / latent_ages['LATENT_AGE']
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


def url_write(url, savedir=paths.data/'APOGEE'):
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
    