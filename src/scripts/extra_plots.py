"""
This script creates multiple plots from the given multizone output for diagnostic purposes.
"""

import argparse

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from age_abundance_grid import plot_age_abundance_grid
from feh_distribution import plot_feh_distribution
from ofe_distribution import plot_ofe_distribution
# from ofe_bimodality import plot_bimodality_comparison
from ofe_feh_grid import plot_ofe_feh_grid
import paths

def main(output_name, verbose=False, tracks=False, log_age=False, 
         uncertainties=False, apogee_data=False, style='paper'):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    # Import multizone stars data
    mzs = MultizoneStars.from_output(output_name)
    parent_dir = paths.extra / mzs.name.replace('diskmodel', '')
    # Forward-model APOGEE uncertainties
    if uncertainties:
        mzs.model_uncertainty(inplace=True, apogee_data=apogee_sample.data)
    # Age vs [O/H]
    plot_age_abundance_grid(mzs, '[o/h]', color_by='galr_origin', cmap='winter_r', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Age vs [Fe/H]
    plot_age_abundance_grid(mzs, '[fe/h]', color_by='galr_origin', cmap='winter_r', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Age vs [O/Fe]
    plot_age_abundance_grid(mzs, '[o/fe]', color_by='[fe/h]', cmap='viridis', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Abundance distributions
    plot_feh_distribution(mzs, apogee_sample, style=style)
    plot_ofe_distribution(mzs, apogee_sample, style=style)
    # [O/Fe] vs [Fe/H]
    plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks,
                      apogee_contours=apogee_data, style=style)
    # [O/Fe] vs [Fe/H], color-coded by age
    plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks,
                      apogee_contours=apogee_data, style=style,
                      color_by='age', cmap='Spectral_r', 
                      fname='ofe_feh_age_grid.png')
    print('Done! Plots are located at %s' % str(parent_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='extra_plots.py',
        description='Generate multiple diagnostic plots for the given multizone output.'
    )
    parser.add_argument(
        'output_name', 
        metavar='NAME',
        type=str,
        help='Name of VICE multizone output located within src/data/multizone.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '-t', '--tracks', 
        action='store_true',
        help='Plot ISM tracks in addition to stellar abundances.'
    )
    parser.add_argument(
        '-l', '--log-age', 
        action='store_true',
        help='Plot age on a log scale.'
    )
    parser.add_argument(
        '-u', '--uncertainties', 
        action='store_true',
        help='Forward-model APOGEE uncertainties in VICE output.'
    )
    parser.add_argument(
        '-a', '--apogee-data', 
        action='store_true',
        help='Plot APOGEE data for comparison.'
    )
    parser.add_argument(
        '--style', 
        type=str,
        default='paper',
        choices=['paper', 'poster', 'presentation'],
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
