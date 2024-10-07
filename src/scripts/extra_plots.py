"""
Plot age vs [O/Fe] from all multizone runs and generate a summary table of
RMS median-difference scores.
"""

import paths
from tqdm import tqdm
from multizone_stars import MultizoneStars
from apogee_tools import import_apogee
from age_ofe import plot_age_ofe
from feh_distribution import plot_feh_distribution
from ofe_distribution import plot_ofe_distribution
from ofe_bimodality import plot_bimodality_comparison
from ofe_feh_grid import plot_ofe_feh_grid

SFH_LIST = [
    'insideout', 
    'lateburst', 
    'earlyburst', 
    'twoinfall'
]
DTD_LIST = [
    'powerlaw_slope11', 
    'powerlaw_slope14', 
    'exponential_timescale15',
    'exponential_timescale30', 
    'plateau_width03', 
    'plateau_width10', 
    'prompt',
    'triple'
]
STYLE = 'paper'

def main():
    # Import APOGEE data
    apogee_data = import_apogee()
    # Loop through all VICE multi-zone outputs
    print('Making supplementary plots for every multizone run...')
    with tqdm(total=len(SFH_LIST) * len(DTD_LIST) + 1) as t:
        for evolution in SFH_LIST:
            for RIa in DTD_LIST:
                # Import VICE multi-zone output data
                output_name = '/'.join(['gaussian', evolution, RIa, 'diskmodel'])
                make_all_plots(output_name, apogee_data)
                t.update()
        output_name = 'diffusion/insideout/powerlaw_slope11/diskmodel'
        make_all_plots(output_name, apogee_data)
        t.update()
    print('Done! Plots are located at %s' % str(paths.extra))


def make_all_plots(output_name, apogee_data, uncertainties=True):
    mzs = MultizoneStars.from_output(output_name)
    # Forward-model APOGEE uncertainties
    if uncertainties:
        mzs.model_uncertainty(inplace=True, apogee_data=apogee_data)
    plot_age_ofe(mzs, apogee_data, log=True, style=STYLE)
    plot_feh_distribution(mzs, apogee_data, style=STYLE)
    plot_ofe_distribution(mzs, apogee_data, style=STYLE)
    plot_bimodality_comparison(mzs, apogee_data, resample=True,
                           style=STYLE)
    plot_ofe_feh_grid(mzs, apogee_data, tracks=True, 
                      apogee_contours=True, style=STYLE)


if __name__ == '__main__':
    main()
