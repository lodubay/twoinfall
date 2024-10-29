"""
This script generates a LaTeX table of IMF-averaged nucleosynthetic yields
adopted in this paper.
"""

import pandas as pd
import vice
from multizone.src.yields import W24
import paths

def main():
    ccsn_yields = []
    snia_yields = []
    elements = ['O', 'Mg', 'Si', 'Fe']
    for el in elements:
        ccsn_yields.append('\\num{%.2e}' % vice.yields.ccsne.settings[el])
        snia_yields.append('\\num{%.2e}' % vice.yields.sneia.settings[el])
    df = pd.DataFrame({
        'Element': elements,
        '$y_{\\rm X}^{\\rm CC}$': ccsn_yields,
        '$y_{\\rm X}^{\\rm Ia}$': snia_yields,
    })
    latex_table = df.to_latex(column_format='c|cc', index=False)
    # Replace float 0s with int
    latex_table = latex_table.replace('0.00e+00', '0')
    # Replace \toprule, \midrule, \bottomrule with \hline
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    # Write to output
    with open(paths.output / 'yields.tex', 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    main()
