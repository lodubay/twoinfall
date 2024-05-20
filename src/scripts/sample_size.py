"""
This script prints the size of the APOGEE sample to the file
src/tex/output/sample_size.txt as well as the size of the sample with
ages from Leung et al. (2023) to src/tex/output/age_sample_size.txt
"""

import pandas as pd
from apogee_tools import import_apogee
import paths

data = import_apogee()

# Write file with full sample size
with open(paths.output / 'sample_size.txt', 'w') as f:
    f.write('\\num{%s}' % str(data.shape[0]))

# Select data with non-NA ages
ages = data[data['LATENT_AGE'].notna()]

# Write file with age sample size
with open(paths.output / 'age_sample_size.txt', 'w') as f:
    f.write('\\num{%s}' % str(ages.shape[0]))
