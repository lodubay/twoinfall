"""
This script generates a LaTeX table which reports the number of APOGEE stars
in each Galactic region.
"""

import pandas as pd
import paths
from apogee_sample import APOGEESample
from _globals import GALR_BINS, ABSZ_BINS


def main():
    sample = APOGEESample.load()
    data = sample.data
    # Count number of targets in each region bounded by Rgal and |z|
    counts = data.groupby([pd.cut(data['GALR'], GALR_BINS), 
                           pd.cut(data['GALZ'], ABSZ_BINS)], 
                          observed=False)['APOGEE_ID'].count()
    # Unstack the above multi-indexed Series into a DataFrame
    pivot = counts.unstack(level=0).iloc[::-1] # invert |z| bins
    # Prettify row, column labels
    idx = [f'${r}$ kpc' for r in pivot.index]
    pivot.index = pd.Series(idx, name='$|z|\\in$')
    cols = [f'${c}$ kpc' for c in pivot.columns]
    pivot.columns = pd.Series(cols, name='$R_{\\rm gal}\\in$')
    # Convert to LaTeX table
    latex_table = pivot.style.to_latex(column_format='r|cccccc')
    # Add horizontal lines
    latex_table_list = latex_table.split('\n')
    latex_table_list.insert(1, '\\hline\\hline')
    latex_table_list.insert(4, '\\hline')
    latex_table_list.insert(-2, '\\hline')
    latex_table = '\n'.join(latex_table_list)
    # Write to output file
    with open(paths.output / 'apogee_regions_table.tex', 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    main()
