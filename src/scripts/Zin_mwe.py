import vice
import numpy as np

sz = vice.singlezone(
        name='test_Zin',
        func=lambda t: 1,
        mode='ifr',
        elements=('fe', 'o'),
        Zin=1e-4,
        Mg0=0,
        dt=0.01,
    )
sz.run(np.arange(0, 10, 0.01), overwrite=True)
hist = vice.history('test_Zin')
print(hist)
