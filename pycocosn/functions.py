"""

"""

import os
import re
import subprocess
import warnings

import numpy as np

from astropy import units as u
from astropy.table import Table, Column, vstack
from astropy.time import Time
# from scipy.optimize import leastsq
from lmfit import minimize, Parameters, fit_report
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import InterpolatedUnivariateSpline

# from .classes import *
# from .colours import *
# from .defaults import *
# from .errors import *
# from .utils import *
from . import classes
from . import colours
from . import defaults
from . import errors
from . import utils

##
#
##

__all__ = ["load_filter",
           "get_filter_from_filename",
           "load_phot",
        #    "utils.load_formatted_phot",
           "load",
           "load_all_phot",
           "find_filter_phot",
           "find_formatted_phot",
           "find_recon_spec",
        #    "find_specphase_spec",
        #    "setup_plot_defaults",
        #    "utils.read_list_file",
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
           "run_specphase",
           "plot_mangledata",
           "calc_linear_terms"
           ]

# def importtest():
#     x = classes.BaseSpectrumClass()
#     x = classes.BaseLightCurveClass()
#     x = classes.BaseFilterClass()
#     x = classes.PhotometryClass()
#     x = SpectrumClass()
#     x = classes.LCfitClass()
#     x = classes.specfitClass()
#     x = classes.SNClass()
#     x = classes.FilterClass()
#     x = classes.InfoClass()
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

    if utils.check_file_path(os.path.abspath(path)):
        filter_object = classes.FilterClass()
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



def load_phot(path, names = ('MJD', 'flux', 'flux_err', 'filter'),
              format = 'ascii', verbose = True):
    """
    Loads a single photometry file.

    Parameters
    ----------
    Returns
    -------
    """

    errors.StringWarning(path)

    # phot_table = ap.table.Table.read(path, format = format, names = names)
    phot_table = Table.read(path, format = format, names = names)

    phot_table.replace_column("MJD", Time(phot_table["MJD"], format = 'mjd'))

    phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
    phot_table["flux_err"].unit =  phot_table["flux"].unit


    return phot_table


def load(path, format = "ascii", verbose = True):
    pc = classes.PhotometryClass()
    pc.phot = utils.load_formatted_phot(path, format = format, verbose = verbose)
    pc.unpack(verbose = verbose)
    return pc


def load_all_phot(path = defaults._default_data_dir_path, format = "ascii", verbose = True):
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
    #     raise errors.PathError("The data directory '" + path + "' doesn't exist.")
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


def find_filter_phot(path = defaults._default_data_dir_path, snname = False,
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
    :param path:
    :param snname:
    :param prefix:
    :param file_type:
    :param verbose:

    """
    # regex = re.compile("^SN.*.dat")

    errors.StringWarning(path)
    if not utils.check_dir_path(path):
        # return False
        raise errors.PathError


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


def find_formatted_phot(path = defaults._default_data_dir_path, snname = False,
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
    :param path:
    :param snname:
    :param prefix:
    :param file_type:
    :param verbose:

    """
    # regex = re.compile("^SN.*.dat")

    errors.StringWarning(path)
    if not utils.check_dir_path(path):
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

# def find_specphase_spec(snname, dir_path = defaults._default_specphase_dir_path, file_type = ".spec", verbose = False):
#     """
#
#     Parameters
#     ----------
#
#     Returns
#     -------
#     """
#     if verbose: print(dir_path)
#     errors.StringWarning(dir_path)
#     errors.StringWarning(snname)
#     if type(snname) is not str and type(snname) is not np.string_:
#         raise(errors.PathError)
#
#     if not utils.check_dir_path(dir_path):
#         print("utils.check_dir_path failed")
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
#     utils.check_file_path(path)
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

    specfit = classes.specfitClass()

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

            utils.setup_plot_defaults()

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

            utils.setup_plot_defaults()

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

    :param verbose:
    :return:
    """
    fits_in_recon = [i for i in os.listdir(defaults._default_recon_dir_path) if i[-4:] == ".dat"]
    if verbose: print(fits_in_recon)
    return fits_in_recon


def list_lcs(verbose = False):
    """

    :param verbose:
    :return:
    """
    lcs_in_data_dir = [i for i in os.listdir(os.path.join(defaults._default_data_dir_path, "lc")) if i[-4:] == ".dat"]
    if verbose: print(lcs_in_data_dir)
    return lcs_in_data_dir


def read_sndist_file(path = defaults._default_sn_dist_path, format = "ascii"):
    """

    :param path:
    :param format:
    :return:
    """
    utils.check_file_path(path)
    table = Table.read(path, format = format)
    return table


def load_sndist(snname, *args, **kwargs):
    """

    :param snname:
    :param args:
    :param kwargs:
    :return:
    """
    sndistlist = read_sndist_file(*args, **kwargs)

    try:
        w = np.where(sndistlist["snname"] == snname)
        row = sndistlist[w]
    except:
        warnings.warn("Failed to find distance info for " + snname + ". is it in the list?")
    return row


def load_info(path = defaults._default_info_path, verbose = False):
    """

    :param path:
    :param verbose:
    :return:
    """
    if verbose: print(path)
    i = classes.InfoClass()
    i.load(path)

    return i


def combine_spectra(s1, s2, wmin, wmax, scale=False, report=False, showplot=False, verbose=True):
    """

    :param s1:
    :param s2:
    :param wmin:
    :param wmax:
    :param scale:
    :param report:
    :param showplot:
    :return:
    """
    ## Check the bluer one is first?

    s1_overlap = classes.SpectrumClass()
    s2_overlap = classes.SpectrumClass()

    s1_overlap.load_table(s1.data[np.where(s1.data["wavelength"] > s2.min_wavelength)], path="", trim_wavelength=True,
                          wmin=wmin, wmax=wmax)
    s2_overlap.load_table(s2.data[np.where(s2.data["wavelength"] < s1.max_wavelength)], path="", trim_wavelength=True,
                          wmin=wmin, wmax=wmax)

    s2_spline = InterpolatedUnivariateSpline(s2.wavelength, s2.flux, k=5)
    #     s2_spline = InterpolatedUnivariateSpline(s2_overlap.wavelength, s2_overlap.flux, k=5)

    if scale:
        print("Scaling")
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

    combined_spec = classes.SpectrumClass()
    combined_spec.load_table(vstack([blue_spec_table, overlap_spectable, red_spec_table]), path="")

    return combined_spec


def data_residual(params, data1, data2):
    """

    :param params:
    :param data1:
    :param data2:
    :return:
    """
    scale = params["scale"]

    res = data1 - scale * data2

    return res

#
# #  #------------------------------------#  #
# #  # CoCo Functions                     #  #
# #  #------------------------------------#  #
#
#
# def test_LCfit(snname, coco_dir = defaults._default_coco_dir_path,
#                verbose = True):
#     """
#     Check to see if a fit has been done. Does this by
#     looking for reconstructed LC files
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     # try:
#     #     if not coco_dir:
#     #         coco_dir = defaults._default_coco_dir_path
#     #
#     # except:
#     #     warnings.warn("Something funky with your input")
#
#     utils.check_dir_path(coco_dir)
#
#     if verbose: print(coco_dir)
#
#     try:
#         path_to_test_dat = os.path.join(coco_dir, 'recon', snname + '.dat')
#         path_to_test_stat = os.path.join(coco_dir, 'recon', snname + '.stat')
#
#         for path in [path_to_test_stat, path_to_test_dat]:
#
#             if os.path.isfile(os.path.abspath(path)):
#                 if verbose: print("Looks like you have done a fit, I found ", path )
#                 boolflag = True
#             else:
#                 warnings.warn(os.path.abspath(path) +
#                 " not found. Have you done a fit?")
#                 boolflag = False
#
#     except:
#
#         warnings.warn("Failing gracefully. Can't find the droids you are looking for.")
#         boolflag = False
#
#     return boolflag
#
#
# def run_LCfit(path, coco_dir = defaults._default_coco_dir_path, model = False,
#               verbose = True,):
#     """
#
#     :param path:
#     :param coco_dir:
#     :param model:
#     :param verbose:
#     :return:
#     """
#
#     utils.check_file_path(path)
#     utils.relist() ## Check filter file is up to date
#
#     if model:
#         models = np.unique([i.split(".")[0] for i in os.listdir(os.path.join(defaults._default_coco_dir_path, "src/models"))])
#
#         if model not in models:
#             return False
#         callargs = [os.path.join(defaults._default_coco_dir_path, "lcfit"), path, "-m", model]
#         print("running with", model)
#     else:
#         print("No Model supplied - running with default")
#         callargs = [os.path.join(defaults._default_coco_dir_path, "lcfit"), path]
#     if verbose: print("Running CoCo lcfit on " + path)
#     if verbose: print("callargs are ", callargs)
#
#     cwd = os.getcwd()
#     os.chdir(coco_dir)
#     subprocess.call(callargs)
#     os.chdir(cwd)
#     if verbose: print("Fit complete")
#     pass
#
#
# def run_LCfit_fileinput(listfile_path, coco_dir = defaults._default_coco_dir_path, data_dir = defaults._default_data_dir_path,
#                         verbose = True):
#     """
#
#     :param listfile_path:
#     :param coco_dir:
#     :param data_dir:
#     :param verbose:
#     :return:
#     """
#
#     utils.check_file_path(listfile_path)
#
#     if verbose: print("Reading ", listfile_path)
#
#     file_list = []
#
#     with  open(listfile_path) as infile:
#         for line in infile:
#             if verbose: print(os.path.join(data_dir, os.pardir, line.strip("\n")))
#             file_list.append(os.path.join(data_dir, os.pardir, line.strip("\n")))
#
#     file_list = np.asarray(file_list)
#
#     for lc_path in file_list:
#         run_LCfit(lc_path, coco_dir=coco_dir, verbose=verbose)
#
#     pass
#
#
# def run_all_SNe_LCfit(verbose = True):
#     """
#
#     :param verbose:
#     :return:
#     """
#     pass
#
#
# def test_specfit(snname, coco_dir = False,
#                verbose = True):
#     """
#     Check to see if a fit has been done. Does this by
#     looking for reconstructed .spec filess
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     try:
#         if not coco_dir:
#             coco_dir = defaults._default_coco_dir_path
#
#     except:
#         warnings.warn("Something funky with your input")
#
#     utils.check_dir_path(coco_dir)
#
#     if verbose: print(coco_dir)
#
#     try:
#         ##
#         path_to_test_dat = os.path.join(coco_dir, 'recon', snname + '.dat')
#         path_to_test_stat = os.path.join(coco_dir, 'recon', snname + '.stat')
#         ## NEED TO THINK OF THE BEST WAY TO DO THIS
#
#         for path in [path_to_test_stat, path_to_test_dat]:
#
#             if os.path.isfile(os.path.abspath(path)):
#                 if verbose: print("Looks like you have done a fit, I found ", path )
#                 boolflag = True
#             else:
#                 warnings.warn(os.path.abspath(path) +
#                 " not found. Have you done a fit?")
#                 boolflag = False
#
#     except:
#
#         warnings.warn("Failing gracefully. Can't find the droids you are looking for.")
#         boolflag = False
#
#     return boolflag
#
#
# def run_specfit(SNObject, wantedfilters=False, anchor_distance=1000, save=True, plot = False, coco_dir=defaults._default_coco_dir_path, verbose = True):
#     """
#     replacement for `run_cocospecfit`. Mangles the spectra in the listfiles. Built for comfort, not speed.
#
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     if not wantedfilters:
#         wantedfilters = SNObject.phot.filter_names.data
#
#     outfile_log = []
#     if hasattr(SNObject, "spec") and hasattr(SNObject, "lcfit"):
#         for name, mS in SN.spec.items():
#             #     print(name, mS)
#
#             S = copy.deepcopy(mS)
#             fit_dict = mangle(SNObject, mS, mS.mjd_obs, wantedfilters, anchor_distance=anchor_distance)
#             if plot:
#                 pcc.plot_mangledata(S, fit_dict["data_table"], mS=fit_dict["SpectrumObject"], spl=fit_dict["final_spl"],
#                                     show_linear_extrap=True, normalise=True)
#
#             outfile = snname + "_" + str(fit_dict["SpectrumObject"].mjd_obs).ljust(12, "0") + ".spec"
#
#             while outfile in outfile_log:
#                 j = 1
#                 outfile = snname + "_" + str(fit_dict["SpectrumObject"].mjd_obs + j * 0.00001).ljust(12, "0") + ".spec"
#                 j += 1
#             print(outfile)
#             outfile_log.append(outfile)
#
#             if save:
#                 kcorr.save_mangle(fit_dict["SpectrumObject"], outfile, fit_dict["SpectrumObject"].infile)
#     else:
#         print("SNObject needs lcfit and spectra")
#     # utils.check_file_path(path)
#     # utils.relist() ## Check filter file is up to date
#     # cwd = os.getcwd()
#     # os.chdir(coco_dir)
#     # if verbose: print("Running CoCo specfit on " + path)
#     # subprocess.call([os.path.join(defaults._default_coco_dir_path, "./specfit"), path])
#     # os.chdir(cwd)
#     pass
#
# def run_cocospecfit(path, coco_dir=defaults._default_coco_dir_path, verbose = True):
#     """
#     runs CoCo specfit on the listfile supplied in path
#
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#     utils.check_file_path(path)
#     utils.relist() ## Check filter file is up to date
#     cwd = os.getcwd()
#     os.chdir(coco_dir)
#     if verbose: print("Running CoCo specfit on " + path)
#     subprocess.call([os.path.join(defaults._default_coco_dir_path, "./specfit"), path])
#     os.chdir(cwd)
#     pass
#
# def get_all_spec_lists(dirpath = defaults._default_list_dir_path, verbose=False):
#     ignore = [".DS_Store", "master.list", "lightcurves.list"]
#
#     utils.check_dir_path(dirpath)
#
#     if verbose: print(dirpath)
#
#     listfile_list = [i for i in os.listdir(dirpath) if i not in ignore]
#
#     fullpath_list = [os.path.join(dirpath, j) for j in listfile_list]
#
#     return fullpath_list
#
#
# def specfit_all(verbose=True, dirpath=defaults._default_list_dir_path):
#
#     fullpath_list = get_all_spec_lists(dirpath)
#
#     for i, path in enumerate(fullpath_list):
#
#         if verbose: print(i, path)
#
#         run_specfit(path, verbose=verbose)
#
#         if verbose: print("Done")
#
#     pass
#
#
# def specfit_sn(snname, verbose = True):
#     """
#     runs CoCo specfit on the listfile supplied in path.
#
#     Parameters
#     ----------
#
#     snname -
#
#     Returns
#     -------
#
#     """
#
#     ## Need to look for the recon lc files for snname
#     # sn = classes.SNClass(snname)
#     lcfit = classes.LCfitClass()
#
#     path = os.path.join(lcfit.recon_directory, snname+".dat")
#     lcfit.load_formatted_phot(path)
#     lcfit.unpack()
#     lcfit._sort_phot()
#     lcfit.get_fit_splines()
#
#     ## Need to make new recon lc files for mangling - no overlaps
#     # lcfit.
#     manglefilters = [i for i in lcfit.filter_names]
#
#     if "BessellR" and "SDSS_r" in manglefilters:
#         ## Only use SDSS_r - less hassle
#         manglefilters.remove("BessellR")
#     if "BessellI" and "SDSS_i" in manglefilters:
#         ## Only use SDSS_r - less hassle
#         manglefilters.remove("BessellI")
#
#     filename = snname + "_m.dat"
#     outpath = lcfit.recon_directory
#
#     lcfit.save(filename, path = outpath, filters = manglefilters, squash = True)
#
#     # lcfit.
#     ## Need to change the listfile to one that has snname matches the new lc file
#     listpath = os.path.join(defaults._default_coco_dir_path, "lists", snname + ".list")
#
#     origlist = utils.read_list_file(listpath)
#     origlist.rename_column('snname', 'snname_nomangle')
#     origlist["snname"] = [j+"_m" for j in origlist["snname_nomangle"]]
#
#     newlistpath = os.path.join(defaults._default_coco_dir_path, "lists", snname + "_m.list")
#     newlist = origlist["spec_path", "snname", "mjd_obs", "z"]
#     # newlist.write(newlistpath, format = "ascii.fast_commented_header", overwrite = True)
#     newlist.write(newlistpath, format = "ascii.fast_commented_header")
#
#
#     ## Need to call run_specfit with path of new list file
#
#     run_specfit(newlistpath)
#
#     pass
#
#
# def run_specphase(filtername, phase_path, filetype=".dat", coco_dir=defaults._default_coco_dir_path, verbose = True):
#     """
#     runs CoCo specphase.
#
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#     filters = utils._get_current_filter_registry()
#
#     if filtername+filetype not in filters:
#         warnings.warn("Filtername not recognised")
#         return False
#
#     utils.check_file_path(phase_path)
#
#     utils.relist() ## Check filter file is up to date
#     cwd = os.getcwd()
#     os.chdir(coco_dir)
#     # if verbose: print("Running CoCo specfit on " + path)
#     subprocess.call([os.path.join(defaults._default_coco_dir_path, "./specphase"), phase_path, filtername])
#     os.chdir(cwd)
#     pass
#
#
# def check_specphase(snname, spectra_dir="spectra/", coco_dir=defaults._default_coco_dir_path, absolute_path=False):
#     """
#
#     :param snname:
#     :param spectra_dir:
#     :param coco_dir:
#     :param absolute_path:
#     :return:
#     """
#     if not absolute_path:
#         spectra_dir = os.path.join(coco_dir, spectra_dir)
#
#     utils.check_dir_path(spectra_dir)
#
#     dir_list = [spec for spec in os.listdir(spectra_dir) if spec[:len(snname)] == snname]
#     if verbose:
#         print(len(dir_list), "files found matching", snname)
#         print(dir_list)
#     return dir_list
#
#
#  #------------------------------------#  #
#  # Mangling                           #  #
#  #------------------------------------#  #

def plot_mangledata(S, data_table, mS=False, xminorticks=250, yminorticks=0.1, show_lims=True, show_linear_extrap=False,
                    spl=False, spl_clamped=False, spl_wav=False, return_fig=False, ylim=False, frameon=True, units=True,
                    legend=True, zero=False, knot = True, savepng=False, savepdf=False, outpath="mangle", show=True,
                    plot_anchors=True, plot_anchor_fitflux=True, normalise=False, verbose=False):
    """

    :param normalise:
    :param verbose:
    :param legend:
    :param zero:
    :param knot:
    :param savepng:
    :param savepdf:
    :param outpath:
    :param show:
    :param plot_anchors:
    :param plot_anchor_fitflux:
    :param S:
    :param data_table:
    :param mS:
    :param xminorticks:
    :param yminorticks:
    :param show_lims:
    :param show_linear_extrap:
    :param spl:
    :param spl_clamped:
    :param spl_wav:
    :param return_fig:
    :param ylim:
    :param frameon:
    :param units:
    :return:
    """

    utils.setup_plot_defaults()
    xaxis_label_string = r'$\textnormal{Wavelength, Angstrom (\AA)}$'
    yaxis_label_string = r'$\textnormal{Fractional Throughput}$'

    if units:
        spec_yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'
    else:
        spec_yaxis_label_string = r'$\textnormal{Flux, Scaled}$'

    yminorLocator = MultipleLocator(yminorticks)
    xminorLocator = MultipleLocator(xminorticks)

    # fig = plt.figure(figsize=[12, 6])
    fig = plt.figure(figsize=[8, 4])
    fig.subplots_adjust(left=0.1, bottom=0.15, top=0.95,
                        right=0.92, hspace=0, wspace=0)

    ax = fig.add_subplot(111)
    ax1 = ax.twinx()

    #     ax.plot(S.data['wavelength'], S.data['flux'], zorder = 0, label=r"$\textnormal{Spectrum}$")

    if normalise:
        norm_factor = mS.norm_factor
    else:
        norm_factor = 1

    ax.plot(S.wavelength, S.flux*norm_factor, zorder=0, label=r"$\textnormal{Spectrum}$")

    if plot_anchors:
        if plot_anchor_fitflux:
            ax.scatter(data_table["lambda_eff"], data_table["fitflux"], color=data_table["knot_colours"], label=None,
                       marker="*", s=120)
        else:
            ax.scatter(data_table["lambda_eff"][data_table["mask"]], data_table["fitflux"][data_table["mask"]], color=data_table["knot_colours"][data_table["mask"]], label=None,
                   marker="*", s=120)

        ax.scatter(data_table["lambda_eff"], data_table["spec_filterflux"], edgecolors=data_table["knot_colours"],
                   label=None)
    else:
        ax.scatter(data_table["lambda_eff"][data_table["mask"]], data_table["fitflux"][data_table["mask"]], color=data_table["knot_colours"][data_table["mask"]], label=None,
                   marker="*", s=120)
        ax.scatter(data_table["lambda_eff"][data_table["mask"]], data_table["spec_filterflux"][data_table["mask"]], edgecolors=data_table["knot_colours"][data_table["mask"]],
                   label=None)
    if mS:
        ax.plot(mS.wavelength, mS.flux, zorder=0, label=r"$\textnormal{Mangled Spectrum}$")
        ax.scatter(data_table["lambda_eff"], data_table["mangledspec_filterflux"],
                   edgecolors=data_table["knot_colours"],
                   label=None)

    if not spl_wav:
        spl_wav = S.wavelength
    if spl:
        # ax.plot(spl_wav, spl(spl_wav), color="Black", label=r"$\textnormal{Spline}$")
        ax1.plot(spl_wav, spl(spl_wav), color="Black", label=r"$\textnormal{Spline}$")
        if knot:
            ax1.scatter(data_table["lambda_eff"], spl(data_table["lambda_eff"]), color = "Black", label=None, marker="D", s=20)

    if spl_clamped:
        # ax.plot(spl_wav, spl_clamped(spl_wav), color="Black", label=r"$\textnormal(Clamped Spline)$")
        ax.plot(spl_wav, spl_clamped(spl_wav), color="Black", label=r"$\textnormal(Clamped Spline)$")

        if knot:
            ax1.scatter(data_table["lambda_eff"], spl(data_table["lambda_eff"]), color="Black", label=None, marker="D",
                    s=100)

    if show_linear_extrap:
        if "weights" in data_table.colnames:
            mc_l, mc_u = calc_linear_terms(data_table[data_table["mask"]], key="weights", verbose=verbose)

            ax1.plot(S.data['wavelength'].data, mc_u[0] * S.data['wavelength'].data + mc_u[1], color=colours.hex["batman"],
                    ls=":", label=None)
            ax1.plot(S.data['wavelength'].data, mc_l[0] * S.data['wavelength'].data + mc_l[1], color=colours.hex["batman"],
                    ls=":", label=None)

    for i, f in enumerate(data_table["filter_object"]):
        if isinstance(f, classes.FilterClass):
            filter_label_string = r'$\textnormal{' + f.filter_name.replace("_", " ") + '}$'
            #             filter_label_string = r'$\textnormal{' + f.filter_name.decode().replace("_", " ") + '}$'


            if hasattr(f, "_plot_colour"):
                ax1.plot(f.wavelength, f.throughput, color=f._plot_colour,
                         lw=2, label=filter_label_string, alpha=0.5)
            else:
                ax1.plot(f.wavelength, f.throughput, lw=2, label=filter_label_string, alpha=0.5)

            if show_lims:
                try:
                    ax1.plot([f._upper_edge, f._upper_edge], [0, 99.],
                             lw=1.5, alpha=0.5, ls=':',
                             color=f._plot_colour, zorder=0, )
                    ax1.plot([f._lower_edge, f._lower_edge], [0, 99.],
                             lw=1.5, alpha=0.5, ls=':',
                             color=f._plot_colour, zorder=0, )
                except:
                    print("Failed")

    default_xlims = ax1.get_xlim()
    default_axylims = ax.get_ylim()

    if zero:
        ax1.plot(default_xlims, [0, 0], color=colours.hex["black"], ls=":")

    ax1.set_xlim(default_xlims)
    ax1.set_xlim(S.min_wavelength * 0.95, S.max_wavelength * 1.05)
    default_ylims = ax1.get_ylim()

    # ax.set_ylim([0, default_axylims[1]])
    # ax.set_ylim([0, 7.01e-16])
    if mS:
        ax_uplim = np.nanmax(np.append(mS.flux.data,np.append(S.flux.data*norm_factor, np.nanmax([data_table["fitflux"],data_table["spec_filterflux"]*norm_factor,data_table["mangledspec_filterflux"]]))))
    else:
        ax_uplim = np.nanmax(np.append(S.flux.data*norm_factor, np.nanmax([data_table["fitflux"],data_table["spec_filterflux"]*norm_factor,data_table["mangledspec_filterflux"]])))

    if spl:
        # ax1_uplim = np.nanmax(data_table["weights"]) * 1.25
        ax1_yuplim = np.nanmax(spl(spl_wav))*1.5
        ax1.set_ylim(0, ax1_yuplim)
    else:
        ax1.set_ylim([0, default_ylims[1]])
    if ylim:
        ax1.set_ylim(ylim)

    ax.set_ylim([-0.05*ax_uplim, 1.05*ax_uplim])
    # ax.set_ylim([0.0, 7.01e-16])
    if verbose: print(default_axylims[1])

    ax1.set_xlabel(xaxis_label_string)
    ax.set_xlabel(xaxis_label_string)
    ax1.set_ylabel(yaxis_label_string)

    ax1.yaxis.set_minor_locator(yminorLocator)
    ax1.xaxis.set_minor_locator(xminorLocator)

    ## https://stackoverflow.com/a/10129461
    lines, labels = ax.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()

    if legend:
        # plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
        #     ax.legend(lines + lines1 + lines2,labels + labels1 + labels2, loc=0, scatterpoints=1,

        ax.legend(lines + lines1, labels + labels1, loc=0, scatterpoints=1,
                  numpoints=1, frameon=frameon, fontsize=12)

    ax.set_ylabel(spec_yaxis_label_string)
    plt.draw()
    if savepdf and outpath:
        fig.savefig(outpath + ".pdf", format='pdf', dpi=500)
    if savepng and outpath:
        fig.savefig(outpath + ".png", format='png', dpi=500)

    if show:
        plt.show()
    else:
        plt.close()
    if return_fig:
        return fig
    pass


def calc_linear_terms(data_table, key = "fitflux", verbose=False):
    """

    :param key:
    :param data_table:
    :param verbose:
    :return:
    """
   ## x1-x2
    dx = data_table["lambda_eff"][0] - data_table["lambda_eff"][1]
    if verbose: print(dx)
    ## y1 - y2
    dy = data_table[key][0] - data_table[key][1]
    if verbose: print(dy)
    ##
    m_lower = dy / dx
    c_lower = data_table[key][0] - m_lower * data_table["lambda_eff"][0]
    if verbose: print(m_lower, c_lower)

   ## x1-x2
    dx = data_table["lambda_eff"][-2] - data_table["lambda_eff"][-1]
    if verbose: print(dx)
    ## y1 - y2
    dy = data_table[key][-2] - data_table[key][-1]
    if verbose: print(dy)
    ##
    m_upper = dy / dx
    c_upper = data_table[key][-2] - m_upper * data_table["lambda_eff"][-2]
    if verbose: print(m_upper, c_upper)

    return [m_lower, c_lower], [m_upper, c_upper]
