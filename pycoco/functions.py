"""

"""

import os
import warnings
import re
import numpy as np
import subprocess

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# from scipy.optimize import leastsq
from lmfit import minimize, Parameters, fit_report
from scipy.interpolate import InterpolatedUnivariateSpline

from astropy.table import Table, Column, vstack
from astropy.time import Time
from astropy import units as u

from .defaults import *
from .errors import *
from .classes import *
from .utils import *

##
#
##

__all__ = ["load_filter",
           "get_filter_from_filename",
           "load_phot",
        #    "load_formatted_phot",
           "load",
           "load_all_phot",
           "find_filter_phot",
           "find_formatted_phot",
           "find_recon_spec",
        #    "find_specphase_spec",
        #    "setup_plot_defaults",
        #    "read_list_file",
           "load_specfit",
           "compare_spec",
           "filter_within_spec",
           "list_lcfits",
           "list_lcs",
           "read_sndist_file",
           "load_sndist",
           "load_info",
           "plot_mangle",
           "test_LCfit",
           "run_LCfit",
           "test_specfit",
           "run_specfit",
           "specfit_sn",
           "run_LCfit_fileinput",
           "get_all_spec_lists",
           "specfit_all",
           "run_specphase"
           ]

# def importtest():
#     x = BaseSpectrumClass()
#     x = BaseLightCurveClass()
#     x = BaseFilterClass()
#     x = PhotometryClass()
#     x = SpectrumClass()
#     x = LCfitClass()
#     x = specfitClass()
#     x = SNClass()
#     x = FilterClass()
#     x = InfoClass()
#     pass

#  #------------------------------------#  #
#  #  Functions                         #  #
#  #------------------------------------#  #

def load_filter(path, cmap = False, verbose = False):
    """
    Loads a filter response into FilterClass and returns it.

    Parameters
    ----------
    Returns
    -------
    """

    if check_file_path(os.path.abspath(path)):
        filter_object = FilterClass()
        filter_object.read_filter_file(os.path.abspath(path), verbose = verbose)

        if cmap:
            filter_object.calculate_plot_colour(verbose = verbose)

        return filter_object
    else:
        warnings.warn("Couldn't load the filter")
        return None


def get_filter_from_filename(path, snname, file_type):
    """
    Parameters
    ----------
    Returns
    -------
    """
    filename_from_path = path.split("/")[-1]
    filename_no_extension = filename_from_path.replace(file_type, "")
    filter_string = filename_no_extension.replace(snname+"_", "")

    # phot_file.replace(file_type, '').split('_')[-1]

    return filter_string


# def _get_filter_directory():
#     """
#     Get the default path to the filter directory.
#
#     Looks for the filter directory set as environment variable
#     $PYCOCO_FILTER_DIR. if not found, returns default.
#
#     returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
#              default datalocation: '../testdata/'.
#     """
#
#     return os.environ.get('PYCOCO_FILTER_DIR', _default_filter_dir_path)


def load_phot(path, names = ('MJD', 'flux', 'flux_err', 'filter'),
              format = 'ascii', verbose = True):
    """
    Loads a single photometry file.

    Parameters
    ----------
    Returns
    -------
    """

    StringWarning(path)

    # phot_table = ap.table.Table.read(path, format = format, names = names)
    phot_table = Table.read(path, format = format, names = names)

    phot_table.replace_column("MJD", Time(phot_table["MJD"], format = 'mjd'))

    phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
    phot_table["flux_err"].unit =  phot_table["flux"].unit


    return phot_table

#
# def load_formatted_phot(path, format = "ascii", names = False,
#                         verbose = True):
#     """
#     Loads a single photometry file.
#
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     StringWarning(path)
#
#     if names:
#         phot_table = Table.read(path, format = format, names = names)
#     else:
#         phot_table = Table.read(path, format = format)
#
#     phot_table.meta = {"filename" : path}
#
#     phot_table["MJD"].unit = u.day
#     phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
#     phot_table["flux_err"].unit =  phot_table["flux"].unit
#
#     return phot_table


def load(path, format = "ascii", verbose = True):
    pc = PhotometryClass()
    pc.phot = load_formatted_phot(path, format = format, verbose = verbose)
    pc.unpack(verbose = verbose)
    return pc


def load_all_phot(path = _default_data_dir_path, format = "ascii", verbose = True):
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
    phot_list = find_filter_phot(path = path)

    if len(phot_list) > 0:
        # phot_table = Table()
        phot_table = ap.table.Table()

        for phot_file in phot_list:
            print(phot_file)
            print(phot_table.read(phot_file, format = format))

        return phot_table
    else:
        warning.warn("Couldn't find any photometry")


def find_filter_phot(path = _default_data_dir_path, snname = False,
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
        # return False
        raise PathError


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

    if os.path.join(path, snname + file_type) in phot_list:
        phot_list.remove(os.path.join(path,snname + file_type))
        warnings.warn("Found " + os.path.join(path,snname + file_type) + " - you could just read that in.")

    if verbose:
        print("searching for", match_string)
        print("Found: ")
        print(ls)
        print("Matched:")
        print(phot_list)
    if len(phot_list) is 0:
        warnings.warn("No matches found.")
    return phot_list


def find_formatted_phot(path = _default_data_dir_path, snname = False,
              prefix = 'SN', file_type = '.dat',
              verbose = True):
    """
    Tries to find photometry in the supplied directory.

    Looks in a directory for things that match SNname.

    Originally did something else - this is a bit hacky.
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
            match_string = str(snname) + file_type
        else:
            warnings.warn("No SN name given")
            return 0
    except:
        raise TypeError
    #
    # regex = re.compile(match_string)

    ls = os.listdir(path)
    if verbose: print(ls)
    # phot_list = [os.path.abspath(os.path.join(path, match.group(0))) for file_name in ls for match in [regex.search(file_name)] if match]
    phot_list = [os.path.abspath(os.path.join(path, file_name)) for file_name in ls if file_name == match_string]

    if os.path.join(path, snname + file_type) in phot_list:
        # phot_list.remove(os.path.join(path,snname + file_type))
        warnings.warn("Found " + os.path.join(path,snname + file_type) + " - you could just read that in.")

    if verbose:
        print("searching for", match_string)
        print("Found: ")
        print(ls)
        print("Matched:")
        print(phot_list)

    if len(phot_list) is 0:
        warnings.warn("No matches found.")
    return phot_list


def find_recon_spec(snname, dir_path = _default_recon_dir_path, verbose = False):
    """

    Parameters
    ----------

    Returns
    -------
    """
    file_type = ".spec"
    StringWarning(dir_path)
    if not check_dir_path(dir_path):
        return False

    try:
        ls = np.array(os.listdir(dir_path))

        wspec = np.where(np.char.find(ls, file_type, start = -len(file_type)) > -1)
        spec_list = ls[wspec]

        ## The last 18 chars are for the MJD and file_type
        wsn = np.where([i[:-18] == snname for i in spec_list])
        snmatch_list = spec_list[wsn]

        if verbose:
            print("Found: ")
            print(ls)
            print("Spec:")
            print(spec_list)
            print("Matched:")
            print(snmatch_list)
        if len(snmatch_list) is 0:
            warnings.warn("No matches found.")
        return snmatch_list

    except:
        warnings.warn("Something went wrong")
        return False


# def find_specphase_spec(snname, dir_path = _default_specphase_dir_path, file_type = ".spec", verbose = False):
#     """
#
#     Parameters
#     ----------
#
#     Returns
#     -------
#     """
#     if verbose: print(dir_path)
#     StringWarning(dir_path)
#     StringWarning(snname)
#     if type(snname) is not str and type(snname) is not np.string_:
#         raise(PathError)
#
#     if not check_dir_path(dir_path):
#         print("check_dir_path failed")
#         return False
#
#     try:
#         ls = np.array(os.listdir(dir_path))
#
#         # wspec = np.where(np.char.find(ls, file_type, start = -len(file_type)) > -1)
#         # spec_list = ls[wspec]
#         spec_list = [i for i in ls if i[-5:] == ".spec"]
#         ## The last 18 chars are for the MJD and file_type
#         # wsn = np.where([i[:-18] == snname for i in spec_list])
#         # snmatch_list = spec_list[wsn]
#         snmatch_list = [i for i in spec_list if i[:len(snname)] == snname ]
#
#         if verbose:
#             print("Found: ")
#             print(ls)
#             print("Spec:")
#             print(spec_list)
#             print("Matched:")
#             print(snmatch_list)
#         if len(snmatch_list) is 0:
#             warnings.warn("No matches found.")
#         return snmatch_list
#
#     except:
#         warnings.warn("Something went wrong")
#         return False


# def check_url_status(url):
#     """
#     Snippet from http://stackoverflow.com/questions/6471275 .
#
#     Checks the status of a website - a status flag of < 400 means the site
#     is up.
#
#     """
#     p = urlparse(url)
#     conn = httplib.HTTPConnection(p.netloc)
#     conn.request('HEAD', p.path)
#     resp = conn.getresponse()
#
#     return resp.status
#
#
# def check_url(url):
#     """
#     Wrapper for check_url_status - considers the status, True if < 400.
#     """
#     return check_url_status(url) < 400


# def setup_plot_defaults():
#     """
#
#     """
#
#     plt.rcParams['ps.useafm'] = True
#     plt.rcParams['pdf.use14corefonts'] = True
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['figure.subplot.hspace'] = 0.1
#     plt.rc('font', family='sans-serif')
#     plt.rc('font', serif='Helvetica')
#     pass


# def read_list_file(path, names = ('spec_path', 'snname', 'mjd_obs', 'z'), verbose = True):
#     """
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#     check_file_path(path)
#     #
#     # ifile = open(path, 'r')
#     #
#     # for line in ifile:
#     #     if verbose: print(line.strip('\n'))
#     # ifile.close()
#     data = Table.read(path, names = names, format = 'ascii')
#     return data


def load_specfit(path):
    """
    Parameters
    ----------
    Returns
    -------
    """

    specfit = specfitClass()

    return specfit


def compare_spec(orig_spec, specfit,
                 xminorticks = 250, legend = True, verbose = True,
                 normalise = False,
                 *args, **kwargs):
        """
        Parameters
        ----------
        Returns
        -------
        """

        if hasattr(orig_spec, "data") and hasattr(specfit, "data"):
            if normalise:
                orig_spec_flux = orig_spec.flux/np.nanmean(orig_spec.flux)
                mangled_spec_flux = specfit.flux/np.nanmean(specfit.flux)
            else:
                orig_spec_flux = orig_spec.flux
                mangled_spec_flux = specfit.flux

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            if verbose: print(np.nanmean(specfit.flux), np.nanmean(orig_spec.flux))

            plot_label_string = r'$\rm{' + orig_spec.data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'

            ax1.plot(orig_spec.data['wavelength'], orig_spec_flux, lw = 2,
                         label = plot_label_string, color = 'Red',
                         *args, **kwargs)

            plot_label_string = r'$\rm{' + specfit.data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'


            ax1.plot(specfit.data['wavelength'], mangled_spec_flux, lw = 2,
                         label = plot_label_string, color = 'Blue',
                         *args, **kwargs)

            maxplotydata = np.nanmax(np.append(mangled_spec_flux, orig_spec_flux))
            minplotydata = np.nanmin(np.append(mangled_spec_flux, orig_spec_flux))

            if legend:

                plot_legend = ax1.legend(loc = 0, scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            ax1.set_ylim(0, maxplotydata*1.02)

            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{cm}^{-2}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


def plot_mangle(orig_spec, specfit,
                 xminorticks = 250, legend = True, verbose = True,
                 normalise = False,
                 *args, **kwargs):
        """
        Parameters
        ----------
        Returns
        -------
        """

        if hasattr(orig_spec, "data") and hasattr(specfit, "data"):
            if normalise:
                orig_spec_flux = orig_spec.flux/np.nanmean(orig_spec.flux)
                mangled_spec_flux = specfit.flux/np.nanmean(specfit.flux)
            else:
                orig_spec_flux = orig_spec.flux
                mangled_spec_flux = specfit.flux

            mangle = mangled_spec_flux/orig_spec_flux

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            # if verbose: print(np.nanmean(specfit.flux), np.nanmean(orig_spec.flux))
            #
            # plot_label_string = r'$\rm{' + orig_spec.data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'
            #
            # ax1.plot(orig_spec.data['wavelength'], orig_spec_flux, lw = 2,
            #              label = plot_label_string, color = 'Red',
            #              *args, **kwargs)
            #
            # plot_label_string = r'$\rm{' + specfit.data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'
            #
            #
            # ax1.plot(specfit.data['wavelength'], mangled_spec_flux, lw = 2,
            #              label = plot_label_string, color = 'Blue',
            #              *args, **kwargs)
            #
            # maxplotydata = np.nanmax([mangled_spec_flux, orig_spec_flux])
            # minplotydata = np.nanmin([mangled_spec_flux, orig_spec_flux])

            # if legend:
            #
            #     plot_legend = ax1.legend(loc = 0, scatterpoints = 1,
            #                           numpoints = 1, frameon = False, fontsize = 12)

            # ax1.set_ylim(0, maxplotydata*1.02)
            plt.plot(orig_spec.data['wavelength'], mangle)
            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            # yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{cm}^{-2}$'
            yaxis_label_string = r'$\textnormal{Mangling spline}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass

# def load_stat(stats_path = '/Users/berto/Code/CoCo/chains/SN2011dh_Bessell/BessellB-stats.dat'):
#     verbose = False
#     fitparams = None
#     j = None
#     stat = None
#     modenumber = 0
#     modes = OrderedDict()
#
#     del j
#     del stat
#     del fitparams
#
#     # stats_path = '/Users/berto/Code/CoCo/chains/SN2006aj/B-stats.dat'
#
#
#     stat =  OrderedDict()
#     nparams = 8
#
#     key_string = 'Params'
#
#     ifile = open(stats_path, 'r')
#
#     for i, line in enumerate(ifile):
#         if line != '\n':
#             if verbose: print(i, line)
#             if line[:17] == "Total Modes Found":
#                 nmodes = int(line.split()[-1])
#
#             if line[:4] == "Mode":
#                 modenumber = modenumber + 1
#                 fitparams = OrderedDict()
#             if line[:8] == 'Strictly':
#                 try:
#                     stat["SLLE"] = np.float64(line.split()[3])
#                     stat["SLLE_err"] = np.float64(line.split()[5])
#                 except:
#                     if verbose: print("NOPE1")
#                 if verbose: print("BAR")
#             if line[:9] == "Local Log":
#                 try:
#                     stat["LLE"] = np.float64(line.split()[2])
#                     stat["LLE_err"] = np.float64(line.split()[4])
#                 except:
#                     if verbose: print("NOPE2")
#                 if verbose: print("SPAM")
#
#             if line[:3] == "MAP":
#                 key_string = key_string + "-MAP"
#                 fitparams = OrderedDict()
#
#             if line[:7] == "Maximum":
#                 key_string = key_string + "-ML"
#                 fitparams = OrderedDict()
#
#             if line[:7] == 'Dim No.':
#                 j = i + nparams+1
#                 if verbose: print("FOO")
#                 try:
#                     fitparams["Dim No."] = np.array([])
#                     fitparams["Mean"] = np.array([])
#                     if len(line.split()) > 2:
#                         fitparams["Sigma"] = np.array([])
#                 except:
#                     if verbose: print("NOPE3")
#             try:
#                 if verbose: print(i, j)
#                 if i<j and line[:7] != 'Dim No.' :
#
#                     if verbose: print("READ")
#                     if verbose: print(len(line.split()) ,line.split())
#                     fitparams["Dim No."] = np.append(fitparams["Dim No."], int(line.split()[0]))
#                     fitparams["Mean"] = np.append(fitparams["Mean"], np.float64(line.split()[1]))
#                     if len(line.split()) > 2:
#                         if verbose: print("GT 2")
#                         if verbose: print(np.float64(line.split()[2]))
#                         fitparams["Sigma"] = np.append(fitparams["Sigma"], np.float64(line.split()[2]))
#                         if verbose: print("fitparams sigma", fitparams["Sigma"])
#                 if i > j:
#     #                 print("SAVE")
#                     if verbose: print("SAVE")
#                     if verbose: print(key_string)
#                     stat[key_string] = fitparams
#
#                     if key_string[-3:] ==  "MAP":
#                         modes["Mode"+str(modenumber)] = stat
#                     key_string = 'Params'
#
#             except:
#                 if verbose: print("NOPE4")
#                 pass
#     #         if line
#
#     ifile.close()
#     print(nmodes)
#     print(stat.keys())
#     return modes


def filter_within_spec(filter_obj, spec_obj):
    """
    returns true if filter_edges are within spectrum, False otherwise

    Parameters
    ----------

    Returns
    -------
    """
    try:
        if hasattr(filter_obj, "_lower_edge") and hasattr(filter_obj, "_upper_edge") and hasattr(spec_obj, "data"):
            blue_bool = filter_obj._lower_edge > spec_obj.min_wavelength
            red_bool = filter_obj._upper_edge < spec_obj.max_wavelength

            if blue_bool and red_bool:
                return True
            else:
                return False
        else:
            warnings.warn("Filter object has no edges or spectrum object has no data")
            return False
    except:
        raise StandardError


def list_lcfits(verbose = False):
    """

    Parameters
    ----------

    Returns
    -------
    """
    fits_in_recon = [i for i in os.listdir(_default_recon_dir_path) if i[-4:] == ".dat"]
    if verbose: print(fits_in_recon)
    return fits_in_recon


def list_lcs(verbose = False):
    """

    Parameters
    ----------

    Returns
    -------
    """
    lcs_in_data_dir = [i for i in os.listdir(os.path.join(_default_data_dir_path, "lc")) if i[-4:] == ".dat"]
    if verbose: print(lcs_in_data_dir)
    return lcs_in_data_dir


def read_sndist_file(path = _default_sn_dist_path, format = "ascii"):
    """

    Parameters
    ----------

    Returns
    -------
    """
    check_file_path(path)
    table = Table.read(path, format = format)
    return table


def load_sndist(snname, *args, **kwargs):
    """

    Parameters
    ----------

    Returns
    -------
    """
    sndistlist = read_sndist_file(*args, **kwargs)

    try:
        w = np.where(sndistlist["snname"] == snname)
        row = sndistlist[w]
    except:
        warnings.warn("Failed to find distance info for " + snname + ". is it in the list?")
    return row


def load_info(path = _default_info_path, verbose = False):
    """

    """
    if verbose: print(path)
    i = InfoClass()
    i.load(path)

    return i


def combine_spectra(s1, s2, wmin, wmax, scale=False, report=False, showplot=False):
    ## Check the bluer one is first?

    s1_overlap = SpectrumClass()
    s2_overlap = SpectrumClass()

    s1_overlap.load_table(s1.data[np.where(s1.data["wavelength"] > s2.min_wavelength)], path="", trim_wavelength=True,
                          wmin=wmin, wmax=wmax)
    s2_overlap.load_table(s2.data[np.where(s2.data["wavelength"] < s1.max_wavelength)], path="", trim_wavelength=True,
                          wmin=wmin, wmax=wmax)

    s2_spline = InterpolatedUnivariateSpline(s2.wavelength, s2.flux, k=5)
    #     s2_spline = InterpolatedUnivariateSpline(s2_overlap.wavelength, s2_overlap.flux, k=5)

    if scale:
        params = Parameters()
        params.add("scale", value=1)

        out = minimize(data_residual, params, args=(s1_overlap.flux, s2_spline(s1_overlap.wavelength)))
        if report: print(fit_report(out))

        scale_factor = out.params["scale"]

        if showplot:
            fig = plt.figure()
            fig.subplots_adjust(left=0.09, bottom=0.20, top=0.99,
                                right=0.97, hspace=0.1, wspace=0.1)

            ax1 = fig.add_subplot(111)
            ax1.plot(s1_overlap.wavelength, s2_spline(s1_overlap.wavelength))

    else:

        scale_factor = 1

    blue_spec_table = s1.data[np.where(s1.data["wavelength"] < wmin)]

    overlap_mean_flux = Column(
        list(map(np.nanmean, zip(s1_overlap.data["flux"], s2_spline(s1_overlap.wavelength) * scale_factor))),
        name=s1_overlap.data["flux"].name, unit=s1_overlap.data["flux"].unit)
    overlap_spectable = Table((s1_overlap.data["wavelength"], overlap_mean_flux),
                              names=(s2_overlap.data["wavelength"].name, overlap_mean_flux.name))

    red_spec_table = s2.data[np.where(s2.data["wavelength"] > wmax)]
    red_spec_table["flux"] = red_spec_table["flux"] * scale_factor

    combined_spec = SpectrumClass()
    combined_spec.load_table(vstack([blue_spec_table, overlap_spectable, red_spec_table]), path="")

    return combined_spec


def data_residual(params, data1, data2):
    scale = params["scale"]

    res = data1 - scale * data2

    return res


#  #------------------------------------#  #
#  # CoCo Functions                     #  #
#  #------------------------------------#  #


def test_LCfit(snname, coco_dir = _default_coco_dir_path,
               verbose = True):
    """
    Check to see if a fit has been done. Does this by
    looking for reconstructed LC files
    Parameters
    ----------
    Returns
    -------
    """

    # try:
    #     if not coco_dir:
    #         coco_dir = _default_coco_dir_path
    #
    # except:
    #     warnings.warn("Something funky with your input")

    check_dir_path(coco_dir)

    if verbose: print(coco_dir)

    try:
        path_to_test_dat = os.path.join(coco_dir, 'recon', snname + '.dat')
        path_to_test_stat = os.path.join(coco_dir, 'recon', snname + '.stat')

        for path in [path_to_test_stat, path_to_test_dat]:

            if os.path.isfile(os.path.abspath(path)):
                if verbose: print("Looks like you have done a fit, I found ", path )
                boolflag = True
            else:
                warnings.warn(os.path.abspath(path) +
                " not found. Have you done a fit?")
                boolflag = False

    except:

        warnings.warn("Failing gracefully. Can't find the droids you are looking for.")
        boolflag = False

    return boolflag


def run_LCfit(path, coco_dir = _default_coco_dir_path, verbose = True,):
    """
    Parameters
    ----------
    Returns
    -------
    """
    check_file_path(path)
    relist() ## Check filter file is up to date

    if verbose: print("Running CoCo lcfit on " + path)
    cwd = os.getcwd()
    os.chdir(coco_dir)
    subprocess.call([os.path.join(_default_coco_dir_path, "lcfit"), path])
    os.chdir(cwd)
    if verbose: print("Fit complete")
    pass


def run_LCfit_fileinput(listfile_path, coco_dir = _default_coco_dir_path, data_dir = _default_data_dir_path, verbose = True):
    """

    :param listfile:
    :param verbose:
    :return:
    """

    check_file_path(listfile_path)

    if verbose: print("Reading ", listfile_path)

    file_list = []

    with  open(listfile_path) as infile:
        for line in infile:
            if verbose: print(os.path.join(data_dir, os.pardir, line.strip("\n")))
            file_list.append(os.path.join(data_dir, os.pardir, line.strip("\n")))

    file_list = np.asarray(file_list)

    for lc_path in file_list:
        run_LCfit(lc_path, coco_dir=coco_dir, verbose=verbose)

    pass


def run_all_SNe_LCfit(verbose = True):
    """

    :param verbose:
    :return:
    """
    pass


def test_specfit(snname, coco_dir = False,
               verbose = True):
    """
    Check to see if a fit has been done. Does this by
    looking for reconstructed .spec filess
    Parameters
    ----------
    Returns
    -------
    """

    try:
        if not coco_dir:
            coco_dir = _default_coco_dir_path

    except:
        warnings.warn("Something funky with your input")

    check_dir_path(coco_dir)

    if verbose: print(coco_dir)

    try:
        ##
        path_to_test_dat = os.path.join(coco_dir, 'recon', snname + '.dat')
        path_to_test_stat = os.path.join(coco_dir, 'recon', snname + '.stat')
        ## NEED TO THINK OF THE BEST WAY TO DO THIS

        for path in [path_to_test_stat, path_to_test_dat]:

            if os.path.isfile(os.path.abspath(path)):
                if verbose: print("Looks like you have done a fit, I found ", path )
                boolflag = True
            else:
                warnings.warn(os.path.abspath(path) +
                " not found. Have you done a fit?")
                boolflag = False

    except:

        warnings.warn("Failing gracefully. Can't find the droids you are looking for.")
        boolflag = False

    return boolflag


def run_specfit(path, coco_dir=_default_coco_dir_path, verbose = True):
    """
    runs CoCo specfit on the listfile supplied in path

    Parameters
    ----------
    Returns
    -------
    """
    check_file_path(path)
    relist() ## Check filter file is up to date
    cwd = os.getcwd()
    os.chdir(coco_dir)
    if verbose: print("Running CoCo specfit on " + path)
    subprocess.call([os.path.join(_default_coco_dir_path, "./specfit"), path])
    os.chdir(cwd)
    pass


def get_all_spec_lists(dirpath = _default_list_dir_path, verbose=False):
    ignore = [".DS_Store", "master.list", "lightcurves.list"]

    check_dir_path(dirpath)

    if verbose: print(dirpath)

    listfile_list = [i for i in os.listdir(dirpath) if i not in ignore]

    fullpath_list = [os.path.join(dirpath, j) for j in listfile_list]

    return fullpath_list


def specfit_all(verbose=True, dirpath=_default_list_dir_path):

    fullpath_list = get_all_spec_lists(dirpath)

    for i, path in enumerate(fullpath_list):

        if verbose: print(i, path)

        run_specfit(path, verbose=verbose)

        if verbose: print("Done")

    pass


def specfit_sn(snname, verbose = True):
    """
    runs CoCo specfit on the listfile supplied in path.

    Parameters
    ----------

    snname -

    Returns
    -------

    """

    ## Need to look for the recon lc files for snname
    # sn = SNClass(snname)
    lcfit = LCfitClass()

    path = os.path.join(lcfit.recon_directory, snname+".dat")
    lcfit.load_formatted_phot(path)
    lcfit.unpack()
    lcfit._sort_phot()
    lcfit.get_fit_splines()

    ## Need to make new recon lc files for mangling - no overlaps
    # lcfit.
    manglefilters = [i for i in lcfit.filter_names]

    if "BessellR" and "SDSS_r" in manglefilters:
        ## Only use SDSS_r - less hassle
        manglefilters.remove("BessellR")
    if "BessellI" and "SDSS_i" in manglefilters:
        ## Only use SDSS_r - less hassle
        manglefilters.remove("BessellI")

    filename = snname + "_m.dat"
    outpath = lcfit.recon_directory

    lcfit.save(filename, path = outpath, filters = manglefilters, squash = True)

    # lcfit.
    ## Need to change the listfile to one that has snname matches the new lc file
    listpath = os.path.join(_default_coco_dir_path, "lists", snname + ".list")

    origlist = read_list_file(listpath)
    origlist.rename_column('snname', 'snname_nomangle')
    origlist["snname"] = [j+"_m" for j in origlist["snname_nomangle"]]

    newlistpath = os.path.join(_default_coco_dir_path, "lists", snname + "_m.list")
    newlist = origlist["spec_path", "snname", "mjd_obs", "z"]
    # newlist.write(newlistpath, format = "ascii.fast_commented_header", overwrite = True)
    newlist.write(newlistpath, format = "ascii.fast_commented_header")


    ## Need to call run_specfit with path of new list file

    run_specfit(newlistpath)

    pass


def run_specphase(filtername, phase_path, filetype=".dat", coco_dir=_default_coco_dir_path, verbose = True):
    """
    runs CoCo specphase.

    Parameters
    ----------
    Returns
    -------
    """
    filters = _get_current_filter_registry()

    if filtername+filetype not in filters:
        warnings.warn("Filtername not recognised")
        return False

    check_file_path(phase_path)

    relist() ## Check filter file is up to date
    cwd = os.getcwd()
    os.chdir(coco_dir)
    # if verbose: print("Running CoCo specfit on " + path)
    subprocess.call([os.path.join(_default_coco_dir_path, "./specphase"), phase_path, filtername])
    os.chdir(cwd)
    pass