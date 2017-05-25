#!/usr/bin env python
'''
This is the utilities sub-module for the pycoco python tools.

author: Rob Firth, Southampton
date: 28-02-2016
'''

from __future__ import print_function

import sys
import os
import warnings
import pycoco as pcc
from numpy import loadtxt, savetxt, array, array_equal
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u

from .defaults import *
from .errors import *

__all__ = ["setup_plot_defaults",
            "relist",
            "load_coords",
            "check_dir_path",
            "check_file_path",
            "read_list_file",
            "load_formatted_phot"]


def _get_filter_directory():
    """
    Get the default path to the filter directory.

    Looks for the filter directory set as environment variable
    $PYCOCO_FILTER_DIR. if not found, returns default.

    returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
             default datalocation: '../testdata/'.
    """

    return os.environ.get('PYCOCO_FILTER_DIR', _default_filter_dir_path)


def _get_filters():
    """
    Parameters
    ----------

    Returns
    -------
    """
    filter_dir = _get_filter_directory()
    file_list = os.listdir(filter_dir)

    for filter_file in file_list:
        if not os.path.isfile(os.path.join(filter_dir, filter_file)):
            file_list.remove(filter_file)
        elif filter_file[0] == ".":
            file_list.remove(filter_file)
        elif filter_file == "list.txt":
            file_list.remove(filter_file)

    return array(file_list)


def _check_filters():
    """
    Parameters
    ----------

    Returns
    -------
    """
    filter_dir = _get_filter_directory()
    path = os.path.join(filter_dir, "list.txt")

    current_arr = loadtxt(path, dtype = str)

    filter_arr = _get_filters()

    return array_equal(current_arr, filter_arr)


def make_list_dot_txt():
    """
    Parameters
    ----------

    Returns
    -------
    """
    filter_dir = _get_filter_directory()
    outpath = os.path.join(filter_dir, "list.txt")
    new_list = _get_filters()
    savetxt(outpath, new_list, fmt = "%s")
    pass


def relist(force = False):
    """
    Parameters
    ----------

    Returns
    -------
    """
    if force or not _check_filters:
        make_list_dot_txt()
    else:
        print("current list.txt is up to date. re run with force = True to force.")
    pass


def load_coords(filename = "sncoordinates.list"):
    """

    """
    path = os.path.abspath(os.path.join(__file__, os.path.pardir, filename))
    coordtable = Table.read(path, format = 'ascii.commented_header')
    return coordtable


def check_dir_path(path, verbose = False):
    """
    Parameters
    ----------
    Returns
    -------
    """
    try:
        if os.path.isdir(os.path.abspath(path)):
            if verbose: print("foo")
            return True
        elif os.path.isfile(os.path.abspath(path)):
            if verbose: print("is file")
            raise PathError
        else:
        #     if verbose: print("bar")
            warnings.warn(os.path.abspath(path) +
            " is not a valid directory. Returning 'False'.")
            return False
    except:
        raise PathError("The path '" + str(path) + "'is not a directory or doesn't exist.")
        return False


def check_file_path(path, verbose = False):
    """

    """
    try:
        if os.path.isfile(os.path.abspath(str(path))):
            if verbose: print("bar")
            return True

        elif os.path.isdir(os.path.abspath(path)):
            if verbose: print("is dir")
            raise PathError
        else:
            warnings.warn(os.path.abspath(path) +
            " is not a valid file. Returning 'False'.")
            return False
    except:
        raise PathError("The data file '" + str(path) + "' doesn't exist or is a directory.")
        return False


def simulate_out_to_ap_table(mjd_to_sim, flux, dflux, filters_to_sim,
                             names = ('MJD', 'flux', 'flux_err', 'filter')):
    return Table([mjd_to_sim, flux, dflux, filters_to_sim.astype(str)], names = names)


def read_list_file(path, names = ('spec_path', 'snname', 'mjd_obs', 'z'), verbose = True):
    """
    Parameters
    ----------
    Returns
    -------
    """
    check_file_path(path)

    data = Table.read(path, names = names, format = 'ascii')
    return data


def setup_plot_defaults():
    """

    """

    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.subplot.hspace'] = 0.1
    plt.rc('font', family='sans-serif')
    plt.rc('font', serif='Helvetica')
    pass


def load_formatted_phot(path, format = "ascii", names = False,
                        verbose = True):
    """
    Loads a single photometry file.

    Parameters
    ----------
    Returns
    -------
    """

    StringWarning(path)

    if names:
        phot_table = Table.read(path, format = format, names = names)
    else:
        phot_table = Table.read(path, format = format)

    phot_table.meta = {"filename" : path}

    phot_table["MJD"].unit = u.day
    phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
    phot_table["flux_err"].unit =  phot_table["flux"].unit

    return phot_table

if sys.version_info < (3,):
    def b(x):
        return x
else:
    import codecs
    def b(x):
        return codecs.latin_1_encode(x)[0]