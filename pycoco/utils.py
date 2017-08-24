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
from numpy import savetxt, arange, where, array_equiv, exp, sort, asarray, zeros, nanmin, nanmax
import matplotlib.pyplot as plt

from astropy.table import Table, Column
from astropy import units as u

from .defaults import *
from .errors import *

__all__ = ["setup_plot_defaults",
           "relist",
           "load_coords",
           "check_dir_path",
           "check_file_path",
           "read_list_file",
           "load_formatted_phot",
           "strictly_increasing",
           "check_list",
           "check_all_lists",
           "specphot_out_to_ap_table",
           "_get_current_filter_registry",
           "get_mjdmax",
           "get_mjdmax_flux",
           "get_max_info"
           ]


def _get_filter_directory():
    """
    Get the default path to the filter directory.

    Looks for the filter directory set as environment variable
    $PYCOCO_FILTER_DIR. if not found, returns default.

    returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
             default datalocation: '../testdata/'.
    """

    return os.environ.get('PYCOCO_FILTER_DIR', _default_filter_dir_path)


def _get_filters(filter_dir=False):
    """
    Parameters
    ----------

    Returns
    -------
    """
    if not filter_dir:
        filter_dir = _get_filter_directory()

    file_list = os.listdir(filter_dir)

    for filter_file in file_list:
        if not os.path.isfile(os.path.join(filter_dir, filter_file)):
            file_list.remove(filter_file)
        elif filter_file[0] == ".":
            file_list.remove(filter_file)
        elif filter_file == "list.txt":
            file_list.remove(filter_file)

    return asarray(file_list)


def _get_current_filter_registry(verbose = False):
    """
    Parameters
    ----------

    Returns
    -------
    """
    filter_dir = _get_filter_directory()
    path = os.path.join(filter_dir, "list.txt")

    # current_arr = loadtxt(path, dtype = str) ## This causes chaos with encoding
    current_arr = []
    with open(path ,"r") as infile:
        for line in infile:
            if verbose: print(line.strip("\n"))
            current_arr.append(line.strip("\n"))

    current_arr = asarray(current_arr)

    return current_arr


def _check_filters(filter_dir=False, verbose = False):
    """
    Parameters
    ----------

    Returns
    -------
    """
    if not filter_dir:
        filter_dir = _get_filter_directory()

    path = os.path.join(filter_dir, "list.txt")

    # current_arr = sort([str(i) for i in _get_current_filter_registry()])
    # filter_arr = sort([str(i) for i in _get_filters()])
    current_arr = sort(_get_current_filter_registry())
    filter_arr = sort(_get_filters())
    if verbose:
        print(current_arr, filter_arr)
        print(len(current_arr), len(filter_arr))
    # return array_equal(current_arr, filter_arr)
    return array_equiv(current_arr, filter_arr)


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


def relist(force = False, verbose = False):
    """
    Parameters
    ----------

    Returns
    -------
    """
    if verbose: print(force, _check_filters())
    if force or not _check_filters():
        if verbose: print("updating list.txt")
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


def specphot_out_to_ap_table(out, mjdmax, filter_name, names = ('MJD', 'flux', 'flux_err', 'filter'), remove_zero=False):
    """

    :param out:
    :param mjdmax:
    :param filter_name:
    :param names:
    :return:
    """

    mjd = out[0]+mjdmax

    if not isinstance(filter_name, str):
        filters = Column([filter_name.astype(str) for i in out[0]])
    else:
        filters = Column([filter_name for i in out[0]])

    if remove_zero:
        w = where(out[1] != 0.0)
        ap_table = Table([mjd[w], out[1][w], zeros(len(out[1][w])), filters[w]], names = names)

    else:
        ap_table = Table([mjd, out[1], zeros(len(out[1])), filters], names = names)
    return ap_table


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


def strictly_increasing(L):
    """https://stackoverflow.com/a/4983359"""
    return all(x<y for x, y in zip(L, L[1:]))


def check_list(path, names = ('spec_path', 'snname', 'mjd_obs', 'z'),
               specfiletype=".txt", verbose = True):
    """

    :return:
    """

    listtable = read_list_file(path, names=names, verbose=verbose)
    phases = []


    for item in listtable["spec_path"]:
        filename = item.split("/")[-1]
        filename = filename.split("_")[1:][0]
        filename = filename.strip(specfiletype)
        try:
            phase = float(filename)
        except:
            pass
        phases.append(phase)
        if verbose: print(phase)

    return strictly_increasing(phases)


def check_all_lists(lists_dir, verbose=False):
    """
    Checks that the phases in the listfiles within lists_dir are monotonic

    :param lists_dir:
    :param verbose:
    :return:
    """
    checklist = []
    master_list = make_master_list(lists_dir)

    for spec_listfile in master_list:
        if verbose: print(spec_listfile)
        check_status = check_list(os.path.join(_default_list_dir_path, spec_listfile), verbose=verbose)
        checklist.append(check_status)
        if check_status:
            print(spec_listfile, " passed")
        else:
            print(spec_listfile, " failed")

    return checklist


def make_master_list(lists_dir):
    """

    :return:
    """
    if not lists_dir:
        l = os.listdir(os.path.join(_default_coco_dir_path, "lists/"))
    else:
        check_dir_path(lists_dir)
        l = os.listdir(lists_dir)
    badlist = ['.DS_Store',  'master.list', 'lightcurves.list',]
    ## remove hidden files etc.
    goodlist = [i for i in l if i not in badlist]

    return goodlist


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


def get_mjdmax(sn, filter_key):
    """

    :param sn:
    :param filter_key:
    :return:
    """
    f = sn.lcfit.spline[filter_key]
    mjd_spline = arange(nanmin(sn.phot.data[filter_key]["MJD"]),
                           nanmax(sn.phot.data[filter_key]["MJD"]),
                           0.001)
    w = where(f(mjd_spline) == nanmax(f(mjd_spline)))

    mjdmax = mjd_spline[w]

    return mjdmax


def get_mjdmax_flux(sn, filter_key):
    """

    :param sn:
    :param filter_key:
    :return:
    """
    f = sn.lcfit.spline[filter_key]
    mjd_spline = arange(nanmin(sn.phot.data[filter_key]["MJD"]),
                           nanmax(sn.phot.data[filter_key]["MJD"]),
                           0.001)
    return nanmax(f(mjd_spline))

def get_max_info(sn, filter_key):
    """

    :param sn:
    :param filter_key:
    :return:
    """
    f = sn.lcfit.spline[filter_key]
    mjd_spline = arange(nanmin(sn.phot.data[filter_key]["MJD"]),
                           nanmax(sn.phot.data[filter_key]["MJD"]),
                           0.001)
    w = where(f(mjd_spline) == nanmax(f(mjd_spline)))
    mjdmax = mjd_spline[w]

    return mjdmax, nanmax(f(mjd_spline))


if sys.version_info < (3,):
    def b(x):
        return x
else:
    import codecs
    def b(x):
        return codecs.latin_1_encode(x)[0]


def gaussian(x, g0, x0, sigma0):
    """
    1D gaussian
    """

    gauss = g0*(exp((-(x-x0)*(x-x0))/(2.0*sigma0*sigma0)))
    return gauss
