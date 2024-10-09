"""
This file contains the class APOGEESample which is used in importing and 
handling the APOGEE sample and related datasets. If run as a script, it will 
re-generate the main sample file at src/data/APOGEE/sample.csv.
"""

from numbers import Number
from pathlib import Path
import numpy as np
import pandas as pd
import paths
from utils import fits_to_pandas, box_smooth, kde2D, galactic_to_galactocentric

# Sample galactocentric coordinate bounds
GALR_LIM = (3., 15.)
ABSZ_LIM = (0., 2.)
# Data file names
ALLSTAR_FNAME = 'allStarLite-dr17-synspec_rev1.fits'
LEUNG23_FNAME = 'nn_latent_age_dr17.csv'
# List of columns to include in the final sample
SAMPLE_COLS = ['APOGEE_ID', 'RA', 'DEC', 'GALR', 'GALPHI', 'GALZ', 'SNREV',
               'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'O_H', 'O_H_ERR', 
               'FE_H', 'FE_H_ERR', 'O_FE', 'O_FE_ERR', 'FE_O', 'FE_O_ERR',
               'AGE', 'AGE_ERR', 'LOG_AGE', 'LOG_AGE_ERR']

def main():
    APOGEESample.generate(verbose=True)

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
        # Add column for [O/H]
        sample['O_H'] = sample['O_FE'] + sample['FE_H']
        sample['O_H_ERR'] = sample['O_FE_ERR'] # [X/Fe] and [X/H] errors are the same
        # Add column for [Fe/O]
        sample['FE_O'] = -sample['O_FE']
        sample['FE_O_ERR'] = sample['O_FE_ERR']
        # Limit by galactocentric radius and z-height
        sample = sample[(sample['GALR'] >= GALR_LIM[0]) & 
                        (sample['GALR'] < GALR_LIM[1]) &
                        (sample['GALZ'].abs() >= ABSZ_LIM[0]) &
                        (sample['GALZ'].abs() < ABSZ_LIM[1])]
        sample.reset_index(inplace=True, drop=True)
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
        except FileNotFoundError as e:
            print('APOGEE sample file not found. Please run ``python \
apogee_sample.py`` to generate it first.')
            raise e
        return cls(data, data_dir=data_dir, galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)        

    def kde2D(self, xcol, ycol, bandwidth=0.03, overwrite=False):
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
            xx, yy, logz = kde2D(self.data['FE_H'], self.data['O_FE'], bandwidth)
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
    
    def mdf(self, col='FE_H', bins=100, range=None, smoothing=0.):
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
                                      density=True)
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


    def age_intervals(self, col, bin_edges, quantiles=[0.16, 0.5, 0.84], 
                      age_col='AGE'):
        """
        Calculate stellar age quantiles in bins of a secondary parameter.
        
        Parameters
        ----------
        col : str
            Data column corresponding to the second parameter (typically an
            abundance, like "FE_H").
        bin_edges : array-like
            List or array of bin edges for the secondary parameter.
        quantiles : list, optional
            List of quantiles to calculate in each bin. The default is
            [0.16, 0.5, 0.84], corresponding to the median and +/- one
            standard deviation.
        age_col : str, optional
            Name of column containing ages. The default is 'AGE'.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with each column corresponding to a quantile level
            and each row a bin in the specified secondary parameter, plus
            a final column "Count" with the number of targets in each bin.
        """
        # Remove entries with no age estimate
        data = self.data.dropna(subset=age_col)
        age_grouped = data.groupby(pd.cut(data[col], bin_edges), observed=False)[age_col]
        age_quantiles = []
        for q in quantiles:
            age_quantiles.append(age_grouped.quantile(q))
        age_quantiles.append(age_grouped.count())
        df = pd.concat(age_quantiles, axis=1)
        df.columns = quantiles + ['count']
        return df
        
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
                    
    @property
    def nstars_ages(self):
        """
        int
            Number of stars in the sample with age estimates.
        """
        return self.data[self.data['AGE'].notna()].shape[0]
    
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
        latent_ages.columns = ['LOG_AGE', 'LOG_AGE_ERR', 
                               'AGE', 'AGE_ERR']
        # Limit to stars with <40% age uncertainty per recommendation
        frac_err = latent_ages['AGE_ERR'] / latent_ages['AGE']
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


def contour_levels_2D(arr2d, enclosed=[0.8, 0.3]):
    """
    Calculate the contour levels which contain the given enclosed probabilities.
    
    Parameters
    ----------
    arr2d : np.ndarray
        2-dimensional array of densities.
    enclosed : list, optional
        List of enclosed probabilities of the contour levels. The default is
        [0.8, 0.3].
    """
    levels = []
    l = 0.
    i = 0
    while l < 1 and i < len(enclosed):
        frac_enclosed = np.sum(arr2d[arr2d > l]) / np.sum(arr2d)
        if frac_enclosed <= enclosed[i] + 0.01:
            levels.append(l)
            i += 1
        l += 0.01
    return levels


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
    