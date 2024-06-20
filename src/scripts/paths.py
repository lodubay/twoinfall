'''
Exposes common paths useful for manipulating datasets and generating figures.

'''
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src = root / 'src'

# Absolute path to the `src/data` folder (contains datasets)
data = src / 'data'

# Absolute path to the folder containing multizone simulation outputs
simulation_outputs = data / 'multizone'

# Absolute path to the `src/extra` folder (contains log files and plots which
# won't be included in the manuscript)
extra = src / 'extra'

# Absolute path to the `src/static` folder (contains static images)
static = src / 'static'

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / 'scripts'

# Absolute path to the `src/scripts/styles` folder (contains matplotlib style
# files)
styles = scripts / 'styles'

# Absolute path to the `src/tex` folder (contains the manuscript)
tex = src / 'tex'

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = tex / 'figures'

# Absolute path to the `src/tex/output` folder (contains non-figure output)
output = tex / 'output'
