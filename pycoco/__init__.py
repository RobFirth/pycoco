#!/usr/bin env python

'''
This is the module for the CoCo python tools.

author: Rob Firth, Southampton
date: 06-12-2016
'''

from __future__ import print_function ## Force python3-like printing

import sys

if __name__ is not '__main__':

    __name__ = 'pycoco'

try:
    __file__

except NameError:

    __file__ = sys.argv[0]

# import os
# import warnings
# import unittest
# import subprocess
# # import httplib ## use http.client on python3 - not using at the mo
# # from urlparse import urlparse
# import re
# from collections import OrderedDict
#
# import astropy as ap
# import astropy.units as u
# from astropy.constants import c
# from astropy.time import Time
# from astropy.table import Table, vstack
#
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.ticker import MultipleLocator
# from scipy.interpolate import InterpolatedUnivariateSpline
# from scipy.interpolate import interp1d as interp1d

from .extinction import *
from .colours import *
from .utils import *
from .errors import *
from .kcorr import *
from .functions import *
from .defaults import *
from .classes import *
from .models import *
from .litdata import *

if __name__ == "__main__":
    pass
