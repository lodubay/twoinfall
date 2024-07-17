import vice
import numpy as np

output_name = '../data/onezone/test_Zin'
sz = vice.singlezone(
        name=output_name,
        func=lambda t: 3,
        mode='ifr',
        elements=('fe', 'o'),
        Zin={'fe': 1e-4},
        Mg0=1,
        dt=0.01,
    )
sz.run(np.arange(0, 10, 0.01), overwrite=True)
hist = vice.history(output_name)
print(hist)
