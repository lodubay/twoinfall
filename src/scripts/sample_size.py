"""
This script prints the size of the APOGEE sample to the file
src/tex/output/sample_size.txt as well as the size of the sample with
ages from Leung et al. (2023) to src/tex/output/age_sample_size.txt
"""

from apogee_sample import APOGEESample
import paths

data = APOGEESample.load()

# Write file with full sample size
with open(paths.output / 'sample_size.txt', 'w') as f:
    f.write('\\num{%s}' % str(data.nstars))

# Write file with age sample size
with open(paths.output / 'age_sample_size.txt', 'w') as f:
    f.write('\\num{%s}' % str(data.nstars_ages))
