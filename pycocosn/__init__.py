#!/usr/bin env python

"""
This is the module for the CoCo python tools.

author: Rob Firth, Southampton
date: 06-12-2016
"""

import sys
import os
import re

if __name__ is not '__main__':

    __author__ = "RobFirth"
    __name__ = 'pycoco'

    packageDir = os.path.dirname(os.path.abspath(__file__))
    versionFile = os.path.join(packageDir, 'version.py')

    with open(versionFile, 'r') as f:
        s = f.read()

    # Look up the string value assigned to __version__ in version.py using regexp
    versionRegExp = re.compile("__version__ = \"(.*?)\"")

    # Assign to __version__
    __version__ = versionRegExp.findall(s)[0]

try:
    __file__

except NameError:

    __file__ = sys.argv[0]

from . import extinction
from . import colours
from . import utils
from . import errors
from . import kcorr
from . import functions
from . import defaults
from . import classes
from . import models
from . import litdata
from . import coco
from . import testing

# from . import test
# test.test("foo")

if __name__ == "__main__":
    pass
