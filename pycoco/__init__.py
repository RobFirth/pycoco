'''
This is the module for the PyCoCo python tools.

author: Rob Firth, Southampton
date: 06-12-2016
'''

from __future__ import print_function ## Force python3-like printing

if __name__ is not '__main__':

    __name__ = 'pycoco'
    __version__ = 0.1

try:
    __file__

except NameError:

    __file__ = sys.argv[0]

import os
import warnings
import unittest
import httplib
import re
from urlparse import urlparse

import astropy as ap
import astropy.units as u
from astropy.time import Time

##----------------------------------------------------------------------------##
##                                   TOOLS                                    ##
##----------------------------------------------------------------------------##

##------------------------------------##
##  TESTING                           ##
##------------------------------------##

class TestClass(unittest.TestCase):
    """
    Class for testing pycoco
    """

    def test_SNClass_get_data_dir_returns_default(self):
        x = SNClass()
        self.assertEqual(x._get_data_directory(), _default_data_dir_path)

    def test_load_all_phot_returns_PathError_for_None(self):
        self.assertRaises(PathError, load_all_phot, None)

    def test_find_phot_finds_SN2005bf_B(self):
        directory_path_to_search = "/Users/berto/Code/verbose-enigma/testdata/"
        phot_path = find_phot(directory_path_to_search, snname = "SN2005bf", verbose = False)[0]
        phot_filename = phot_path.split('/')[-1]
        self.assertEqual(phot_filename, u'SN2005bf_B.dat')

    def test_find_phot_finds_no_SN2011fe_data(self):
        directory_path_to_search = "/Users/berto/Code/verbose-enigma/testdata/"
        self.assertEqual(len(find_phot(directory_path_to_search, snname = "SN2011fe", verbose = False)),
                         0)

    def test_find_phot_throws_path_error_for_None(self):
        self.assertRaises(PathError, find_phot, None)

    def test_find_phot_returns_False_for_zoidberg(self):
        self.assertEqual(find_phot('Zoidberg!'), False)

    def test_check_dir_path_finds_pycoco_dir(self):
        self.assertEqual(check_dir_path(_default_data_dir_path), True)

    def test_check_dir_path_raises_PathError_for_None(self):
        self.assertRaises(PathError, check_dir_path, None)

    def test_check_dir_path_returns_False_for_file(self):
        self.assertEqual(check_dir_path(__file__), False)

    def test_check_file_path_finds_SN2005bf_B(self):
        self.assertEqual(check_file_path(os.path.join(_default_data_dir_path, 'SN2005bf_B.dat')), True)

    def test_check_file_path_raises_PathError_for_None(self):
        self.assertRaises(PathError, check_file_path, None)

    def test_check_file_path_returns_False_for_dir(self):
        self.assertEqual(check_file_path(_default_data_dir_path), False)

##------------------------------------##
##  DUMMY CODE                        ##
##------------------------------------##

class CustomValueError(ValueError):
	"""
	Raise when....
	"""


	def __init__(self, *args, **kwargs):
		ValueError.__init__(self, *args, **kwargs)


class DummyClass():
    '''
    Quick dummy class.

    Contains a test class variable and test class method that prints the
    variable.

    RF
    '''


    def __init__(self):
        self.dummy_string = 'Hello, World!'


    def print_dummy_string(self):
        print(self.test_string)


def dummy_function(*args, **kwargs):
    '''
    Quick dummy function.

    Prints supplied **args and **kwargs
    Issues warnings if nothing passed

    RF
    '''

    warnings.simplefilter('always')
    print(args)
    print(kwargs)


    # warnings.warn("WARNING")

    if not args and not kwargs:
        warnings.warn( "You didn't pass any *args or **kwargs", RuntimeWarning)

    else:
        if args:
            for i, arg in enumerate(args):
                print('an arg passed via *args: ', repr(arg))
        else:
            warnings.warn( "You didn't pass any *args", RuntimeWarning)

        if kwargs:
            for key, value in kwargs.iteritems():
                print('a **kwarg: ', repr(key), ' == ' , repr(value))
        else:
            warnings.warn( "You didn't pass any **kwargs", RuntimeWarning)
    pass


_somevar = 'Foo'


##----------------------------------------------------------------------------##
##  CODE                                                                      ##
##----------------------------------------------------------------------------##

## Important variables

_default_data_dir_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir) + '/testdata/')


##------------------------------------##
##  ERROR DEFS                        ##
##------------------------------------##


class CustomValueError(ValueError):
	"""
	Raise when....
	"""
	def __init__(self, *args, **kwargs):
		ValueError.__init__(self, *args, **kwargs)


class PathError(StandardError):
	"""
	Raise when a path is found to be invalid
	"""
	def __init__(self, *args, **kwargs):
		StandardError.__init__(self, *args, **kwargs)


def StringWarning(path):
    """

    """
    if type(path) is not str and type(path) is not unicode:
        warnings.warn("WARNING: You passed something that was " + str(type(path)) + "This might go wrong.",
                      stacklevel = 2)

    else:
        pass


##------------------------------------##
##                                    ##
##------------------------------------##


class SNClass():
    """docstring for SNClass."""

    def __init__(self, verbose = False):
        """

        """

        ## Initialise the class variables
        self._default_data_dir_path = _default_data_dir_path

        ## Initialise using class methods
        self.set_data_directory(self._get_data_directory())


    def _get_data_directory(self):
        """
        Get the defaul path to the data directory.

        Looks for the data data directory set as environment variable
        $PYCOCO_DATA_DIR. if not found, returns default.

        returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
                 default datalocation: '../testdata/'.
        """

        return os.environ.get('PYCOCO_DATA_DIR', self._default_data_dir_path)


    def set_data_directory(self, data_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if os.path.isdir(os.path.abspath(data_dir_path)):
                self.data_directory = os.path.abspath(data_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(data_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_data_dir_path, UserWarning)
                self.data_directory = self._default_data_dir_path

                if not os.path.isdir(self.data_directory):
                    if verbose: print(os.path.isdir(self.data_directory))
                    raise PathError("The default data directory '" + self.data_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default data directory '" + self._default_data_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_data_dir_path?")
            pass


    def load_phot_from_file():
        """

        """




        pass

    def load_phot_ap_tables():
        """

        """

        pass

    def plot():

        pass

def check_dir_path(path, verbose = False):
    """

    """
    try:
        if os.path.isdir(os.path.abspath(path)):
            if verbose: print("foo")
            return True
        else:
        #     if verbose: print("bar")
            warnings.warn(os.path.abspath(path) +
            " is not a valid directory. Returning 'False'.")
            return False
    except:
        raise PathError("The path '" + str(path) + "'is not a directory or doesn't exist.")
        return False



def check_file_path(path, verbose = True):
    """

    """
    try:
        if os.path.isfile(os.path.abspath(str(path))):
            if verbose: print("foo")
            return True
        else:
            warnings.warn(os.path.abspath(path) +
            " is not a valid file. Returning 'False'.")
            return False
    except:
        raise PathError("The data file '" + str(path) + "' doesn't exist or is a directory.")
        return False



def load_phot(path, names = ('MJD', 'flux', 'flux_err', 'filter'),
              format = 'ascii', verbose = True):
    """
    Loads a single photometry file.


    """

    StringWarning(path)

    phot_table = ap.table.Table.read(path, format = format, names = names)

    phot_table.replace_column("MJD", Time(phot_table["MJD"], format = 'mjd'))

    phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
    phot_table["flux_err"].unit =  phot_table["flux"].unit


    return phot_table

def load_all_phot(path = _default_data_dir_path, format = 'ascii', verbose = True):
    """
    loads photometry into AstroPy Table.

    returns: AstroPy table
    """
    ## Do i need the following? Errors should be handled in find_phot?
    # try:
    #     if os.path.isdir(os.path.abspath(path)):
    #         pass
    #     else:
    #         warnings.warn(os.path.abspath(data_dir_path) +
    #         " is not a valid directory. Returning 'False'.")
    # except:
    #     raise PathError("The data directory '" + path + "' doesn't exist.")
    phot_list = find_phot(path = path)

    if len(phot_list) > 0:
        phot_table = ap.table.Table()

        for phot_file in phot_list:
            print(phot_file)
            print(phot_table.read(phot_file, format = format))

        return phot_table
    else:
        warning.warn("Couldn't find any photometry")


def find_phot(path = _default_data_dir_path, snname = False,
              prefix = 'SN', file_type = '.dat',
              verbose = True):
    """
    Tries to find photometry in the supplied directory.

    Looks in a directory for things that match SN*.dat. Uses regex via `re` -
    probably overkill.

    Parameters
    ----------

    path :

    snname :

    prefix :

    file_type :


    Returns
    -------

    phot_list :

    """
    # regex = re.compile("^SN.*.dat")

    StringWarning(path)
    if not check_dir_path(path):
        return False

    try:
        if snname:
            match_string = "^" + str(snname) + ".*" + '.dat'
        else:
            match_string = "^" + str(prefix) + ".*" + '.dat'
    except:
        raise TypeError

    regex = re.compile(match_string)

    ls = os.listdir(path)

    phot_list = [os.path.abspath(os.path.join(path, match.group(0))) for file_name in ls for match in [regex.search(file_name)] if match]

    if verbose:
        print("Found: ")
        print(ls)
        print("Matched:")
        print(phot_list)
    if len(phot_list) is 0:
        warnings.warn("No matches found.")
    return phot_list


def check_url_status(url):
    """
    Snippet from http://stackoverflow.com/questions/6471275 .

    Checks the status of a website - a status flag of < 400 means the site
    is up.

    """
    p = urlparse(url)
    conn = httplib.HTTPConnection(p.netloc)
    conn.request('HEAD', p.path)
    resp = conn.getresponse()

    return resp.status


def check_url(url):
    """
    Wrapper for check_url_status - considers the status, True if < 400.
    """
    return check_url_status(url) < 400


##----------------------------------------------------------------------------##
##  /CODE                                                                     ##
##----------------------------------------------------------------------------##

if __name__ is '__main__':

    test = False
    test = True

    if test:

        print("Running test suite:\n")

        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        unittest.TextTestRunner(verbosity=2).run(suite)

else:

    pass
