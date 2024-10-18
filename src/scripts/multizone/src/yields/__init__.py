import vice
from ..._globals import YIELDS

if YIELDS == "F04":
    from . import F04
elif YIELDS == "JW20":
    from vice.yields.presets import JW20
elif YIELDS == "J21":
    from . import J21
elif YIELDS == "C22":
    from . import C22
elif YIELDS == "W24":
    from . import W24
else:
    raise ValueError("Unrecognized yield specification in multizone/_globals.py.")
