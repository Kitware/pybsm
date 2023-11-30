# Normally we don't do this but pybsm.py is the entire
# package and atms/ is just data files that it requires.
# So doing this means this package can still be used like
# pybsm.* instead of pybsm.pybsm.*
from .pybsm import *