"""
This script generates a LaTeX table of IMF-averaged nucleosynthetic yields
adopted in this paper.
"""

import pandas as pd
import vice
import paths
from multizone.src import outflows

def main():
    from multizone.src.yields import yZ1
    yZ1_yields, yZ1_labels = make_column(eta=outflows.yZ1)
    from multizone.src.yields import yZ2
    yZ2_yields, yZ2_labels = make_column(eta=outflows.yZ2)
    from multizone.src.yields import yZ3
    yZ3_yields, yZ3_labels = make_column(eta=outflows.yZ3)
    df = pd.DataFrame({
        '$y/Z_\\odot=1$': ['(empirical)'] + yZ1_yields,
        '$y/Z_\\odot=2$': ['(theoretical)'] + yZ2_yields,
        '$y/Z_\\odot=3$': ['(extreme)'] + yZ3_yields,
    }, index=[''] + yZ1_labels)
    latex_table = df.to_latex(column_format='c|ccc', index=True)
    # Replace float 0s with int
    latex_table = latex_table.replace('0.00e+00', '0')
    # Replace \toprule, \midrule, \bottomrule with \hline
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\n\\midrule', '')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    # Add horizontal rule between column labels and yields
    latex_table = latex_table.replace('$y_{\\rm O}^{\\rm CC}$', '\\hline\n$y_{\\rm O}^{\\rm CC}$')
    # Add horizontal rule between yields and SN Ia rates
    latex_table = latex_table.replace('$N_', '\\hline\n$N_')
    # Write to output
    with open(paths.output / 'yields.tex', 'w') as f:
        f.write(latex_table)

def make_column(elements = ['O', 'Fe'], mfeia=0.7, eta=outflows.yZ1):
    ccsn_yields = []
    ccsn_labels = []
    snia_yields = []
    snia_labels = []
    for el in elements:
        ccsn_yields.append('\\num{%.2e}' % vice.yields.ccsne.settings[el])
        ccsn_labels.append('$y_{\\rm %s}^{\\rm CC}$' % el)
        snia_yields.append('\\num{%.2e}' % vice.yields.sneia.settings[el])
        snia_labels.append('$y_{\\rm %s}^{\\rm Ia}$' % el)
    # Calculate SN Ia rates assuming a mean Fe mass per SN
    snia_labels.append('$N_{\\rm Ia}/M_\\star\\,[{\\rm M}_\\odot^{-1}]$')
    snia_yields.append('\\num{%.2e}' % (vice.yields.sneia.settings['fe'] / mfeia))
    # Yield-specific outflow parameters
    # outflow_params = [f'{eta.solar_value:.1f}']
    # outflow_labels = ['$\\eta_\\odot$']
    # All parameters and labels
    params = ccsn_yields + snia_yields #+ outflow_params
    labels = ccsn_labels + snia_labels #+ outflow_labels
    return params, labels

if __name__ == '__main__':
    main()
