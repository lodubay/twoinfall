"""
This script generates a LaTeX table of the median measurement uncertainties
for the APOGEE DR17 sample.
"""

import pandas as pd
import paths
from apogee_sample import APOGEESample

def main():
    sample = APOGEESample.load()
    params = ['O_H_ERR', 'FE_H_ERR', 'L23_LOG_AGE_ERR', 'CN_AGE_ERR']
    labels = pd.Series([
        '[O/H]', 
        '[Fe/H]', 
        '$\\log_{10}(\\tau_{\\rm NN}/{\\rm Gyr})$', 
        '$\\tau_{\\rm [C/N]}/{\\rm Gyr}$'
    ], name='Parameter')
    medians = sample.data[params].median()
    medians.name = 'Median Uncertainty'
    dispersions = sample.data[params].quantile(0.95) - sample.data[params].quantile(0.05)
    dispersions.name = 'Uncertainty Dispersion ($95\\% - 5\\%$)'
    df = pd.concat([medians, dispersions], axis=1)
    df.set_index(labels, drop=True, inplace=True)
    # Convert to LaTeX
    latex_table = df.to_latex(
        column_format='l|cc', 
        index=True, 
        float_format='%.2g'
    )
    # Replace \toprule, \midrule, \bottomrule with \hline
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    # Custom multi-row column labels
    latex_table = latex_table.replace(
        '& Median Uncertainty & Uncertainty Dispersion ($95\\% - 5\\%$)',
        '& Median & Uncertainty '
    )
    latex_table = latex_table.replace(
        'Parameter &  & ',
        'Parameter & Uncertainty & Dispersion ($95\\% - 5\\%$)'
    )
    # Write to output
    with open(paths.output / 'uncertainties.tex', 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    main()
