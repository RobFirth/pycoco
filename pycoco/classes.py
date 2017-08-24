"""

"""

from __future__ import print_function ## Force python3-like printing

import os
import warnings
import re

# import unittest
# import httplib ## use http.client on python3 - not using at the mo
# from urlparse import urlparse
# import re
from collections import OrderedDict

import astropy as ap
import astropy.units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
from astropy.table import Table, vstack, Row


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d as interp1d
from scipy.integrate import simps

from .extinction import *
from .colours import *
from .utils import *
from .errors import *
from .defaults import *
from .kcorr import *
from .models import *
# from .functions import *
# from .coco_calls import *

# warnings.resetwarnings()
# warnings.simplefilter("error") ## Turn warnings into errors - good for debugging

__all__ = ["BaseSpectrumClass",
           "BaseLightCurveClass",
           "BaseFilterClass",
           "BaseLCModelClass",
           "PhotometryClass",
           "SpectrumClass",
           "LCfitClass",
           "specfitClass",
           "SNClass",
           "FilterClass",
           "InfoClass",
           "find_specphase_spec"]

#  #----------------------------------------------------------------------------#  #
#  #                                   TOOLS                                    #  #
#  #----------------------------------------------------------------------------#  #

#  #------------------------------------#  #
#  #  DUMMY CODE                        #  #
#  #------------------------------------#  #

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


def dummy_function(verbose = True, *args, **kwargs):
    '''
    Quick dummy function.

    Prints supplied **args and **kwargs
    Issues warnings if nothing passed

    RF
    '''
    if verbose: print(__name__)
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
            for key, value in kwargs.items():
                print('a **kwarg: ', repr(key), ' == ' , repr(value))
        else:
            warnings.warn( "You didn't pass any **kwargs", RuntimeWarning)
    pass


_somevar = 'Foo'


#  #----------------------------------------------------------------------------#  #
#  #  CODE                                                                      #  #
#  #----------------------------------------------------------------------------#  #
#  #----------------------------------------------------------------------------#  #
#  #  Classes                                                                   #  #
#  #----------------------------------------------------------------------------#  #
#  #------------------------------------#  #
#  #  Base Classes                      #  #
#  #------------------------------------#  #

class BaseSpectrumClass():
    """
    Base class for handling Spectra.
    """

    def __init__(self):
        """

        """

        ## Initialise the class variables
        self._default_list_dir_path = os.path.join(_default_coco_dir_path, "lists/")
        #
        # ## Initialise using class methods
        self.set_list_directory(self._get_list_directory())

        pass


    def _get_list_directory(self):
        """
        Get the default path to the spec lists directory.

        Looks for the list file directory set as environment variable
        $COCO_ROOT_DIR. if not found, returns default.

        returns: Absolute path in environment variable $COCO_ROOT_DIR, or
                 default location: '~/Code/CoCo/', with 'lists/' appended.
        """

        return os.path.join(os.environ.get('COCO_ROOT_DIR', os.path.join(self._default_list_dir_path, os.pardir)), "lists/")


    def set_list_directory(self, list_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if verbose: print(list_dir_path, self._default_list_dir_path)
            if os.path.isdir(os.path.abspath(list_dir_path)):
                self.list_directory = os.path.abspath(list_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(list_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_list_dir_path, UserWarning)
                self.list_directory = self._default_list_dir_path

                if not os.path.isdir(self.list_directory):
                    if verbose: print(os.path.isdir(self.list_directory))
                    raise PathError("The default list directory '" + self.list_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default list directory '" + self._default_list_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_list_dir_path?")
            pass


    def load(self, filename, directory=False, fmt="ascii",
             wmin=3500 * u.angstrom, wmax=11000 * u.angstrom,
             names=("wavelength", "flux"), wavelength_u=u.angstrom,
             flux_u=u.cgs.erg / u.si.cm ** 2 / u.si.s / u.angstrom,
             convert_flux_u=u.cgs.erg / u.si.cm ** 2 / u.si.s / u.angstrom,
             verbose=False):
        """
        Parameters
        ----------
        Returns
        -------
        """

        StringWarning(filename)

        if not directory:
            ## Differentiate between the two child classes
            if hasattr(self, 'data_directory'):
                path = os.path.join(self.data_directory, filename)
                if verbose: print("You didn't supply a directory, so using self.data_directory")

            if hasattr(self, 'recon_directory'):
                path = os.path.join(self.recon_directory, filename)
                if verbose: print("You didn't supply a directory, so using self.recon_directory")
        else:
            StringWarning(directory)
            check_dir_path(directory)

            path = os.path.join(directory, filename)
            if verbose: print(path)

        if os.path.isfile(path):

            ## Some might have three columns, deal with laters - this is untidy
            try:
                if hasattr(self, "recon_directory"):
                    names = names + ("flux_err",)
                spec_table = Table.read(path, format=fmt, names=names)

            except:
                if "flux_err" not in names:
                    names = names + ("flux_err",)
                spec_table = Table.read(path, format=fmt, names=names)

            if verbose: print("Reading " + path)

            spec_table.meta["filepath"] = path
            spec_table.meta["filename"] = path.split("/")[-1]

            spec_table['wavelength'].unit = wavelength_u

            if wavelength_u != u.Angstrom:
                spec_table['wavelength'] = spec_table['wavelength'].to(u.Angstrom)

            spec_table['flux'].unit = flux_u
            if "flux_err" in spec_table.colnames:
                spec_table["flux_err"].unit = flux_u

                #  Automatically convert units?
            if flux_u != convert_flux_u:
                spec_table["flux"] = spec_table["flux"].to(convert_flux_u)
                if "flux_err" in spec_table.colnames:
                    spec_table["flux_err"] = spec_table["flux_err"].to(convert_flux_u)

                flux_u = convert_flux_u

            # enforce wmin and wmax
            spec_table = spec_table[np.bitwise_and(spec_table['wavelength'] > wmin, spec_table['wavelength'] < wmax)]
            self.min_wavelength = np.nanmin(spec_table["wavelength"])
            self.max_wavelength = np.nanmax(spec_table["wavelength"])

            #  assign to class
            self.data = spec_table
            self.wavelength = spec_table["wavelength"]
            self.flux = spec_table["flux"]

            #  If you got this far...
            self.success = True
        else:
            warnings.warn(path + " is not a valid file path")
            if verbose: print(path + ' not found')


    def load_table(self, spec_table, path, trim_wavelength=False, wmin=3500 * u.angstrom,
                   wmax=11000 * u.angstrom):
        """Use with care - basically assumes you have all of your ducks in a row"""

        if trim_wavelength:
            spec_table = spec_table[np.bitwise_and(spec_table['wavelength'] > wmin, spec_table['wavelength'] < wmax)]

        spec_table.meta["filepath"] = path
        spec_table.meta["filename"] = path.split("/")[-1]

        self.min_wavelength = np.nanmin(spec_table["wavelength"])
        self.max_wavelength = np.nanmax(spec_table["wavelength"])
        self.data = spec_table
        self.wavelength = spec_table["wavelength"]
        self.flux = spec_table["flux"]
        pass

    def plot(self, xminorticks = 250, legend = True,
             verbose = False, compare_red = True,
             *args, **kwargs):
        """
        Plots spec.

        xminorticks : spacing for minor tick marks on x axis

        legend : (True) Show legend

        verbose : print verbose output

        compare_red :

        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.95,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)


            if verbose: print(self.data.__dict__)
            plot_label_string = r'$\rm{' + self.data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'


            ax1.plot(self.data['wavelength'], self.flux, lw = 2,
                         label = plot_label_string, color = 'C0',
                         *args, **kwargs)

            maxplotydata = np.nanmax(self.flux)
            minplotydata = np.nanmin(self.flux)

            if hasattr(self, 'flux_dered') and compare_red:
                ax1.plot(self.data['wavelength'], self.data['flux_dered'], lw = 2,
                             label = plot_label_string, color = 'Blue',
                             *args, **kwargs)
                maxplotydata = np.nanmax(np.append(maxplotydata, np.nanmax(self.data['flux_dered'])))
                minplotydata = np.nanmin(np.append(minplotydata, np.nanmin(self.data['flux_dered'])))
            if legend:

                plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            ax1.set_ylim(minplotydata*0.98, maxplotydata*1.02)

            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            # yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{cm}^{-2}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def set_MJD_obs(self, mjd):
        """
        Log MJD of the observation.

        Parameters
        ----------

        Returns
        -------
        """
        self.mjd_obs = mjd

        pass


    def set_EBV(self, EBV):
        """
        Parameters
        ----------

        Returns
        -------
        """
        self.EBV = EBV


    def deredden(self, z, EBV_host, EBV_MW = False, verbose = True):
        """
        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):
            if verbose: print("Foo")
            if hasattr(self, "EBV") and not EBV_MW:
                EBV_MW = self.EBV

            self.flux_dered = deredden(self.wavelength, self.flux, z, EBV_MW = EBV_MW, EBV_host=EBV_host)
            self.data["flux_dered"] = self.flux_dered

        else:
            warnings.warn("No extinction value set")
        pass


    def use_flux_dered(self):
        """
        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):
            self.flux_red = self.flux
            self.flux = self.data['flux_dered']
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def _spec_format_for_save(self):
        """
        Parameters
        ----------
        Returns
        -------
        """

        save_table = Table()

        save_table['wavelength'] = self.wavelength
        save_table['flux'] = self.flux

        save_table['wavelength'].format = "5.5f"
        save_table['flux'].format = "5.5e"

        return save_table


    def save(self, filename, path = False,
             squash = False, verbose = True, *args, **kwargs):
        """
        Output the spectrum loaded into the Class via self.load into a format
        and location recognised by CoCo.

        Parameters
        ----------
        Returns
        -------
        """

        if hasattr(self, "data"):
            if verbose: print("has data")
            if not path:
                if verbose: print("No directory specified, assuming " + self._default_data_dir_path)
                path = self._default_data_dir_path
            else:
                StringWarning(path)

            outpath = os.path.join(path, filename)

            check_dir_path(path)

            if os.path.isfile(outpath):
                warnings.warn("Found existing file matching " + os.path.join(path, filename) + ". Run with squash = True to overwrite")
                if squash:
                    print("Overwriting " + outpath)
                    self._spec_format_for_save().write(outpath, format = "ascii.fast_commented_header")


            else:
                    print("Writing " + outpath)
                    self._spec_format_for_save().write(outpath, format = "ascii.fast_commented_header")

        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def _add_to_overlapping_filters(self, filter_name):
        if hasattr(self, "_overlapping_filter_list"):
            self._overlapping_filter_list = np.append(self._overlapping_filter_list, filter_name)
        else:
            self._overlapping_filter_list = np.array(filter_name)
        pass


class BaseLightCurveClass():
    """
    Base class for handling Lightcurves.
    """
    def __init__(self, verbose = False):
        """

        """
        ## Initialise the class variables

        ## Initialise using class methods

        pass


    def _get_filter_directory(self):
        """
        Get the default path to the filter directory.

        Looks for the filter data directory set as environment variable
        $PYCOCO_FILTER_DIR. if not found, returns default.

        returns: Absolute path in environment variable $PYCOCO_FILTER_DIR, or
                 default datalocation, i.e.: '/Users/berto/Code/CoCo/data/filters/'.
        """
        return os.path.abspath(os.environ.get('PYCOCO_FILTER_DIR', self._default_filter_dir_path))


    def set_filter_directory(self, filter_dir_path='', verbose=False):
        """
        Set a new filter directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if os.path.isdir(os.path.abspath(filter_dir_path)):
                self.filter_directory = os.path.abspath(filter_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(filter_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_filter_dir_path, UserWarning)
                self.data_directory = self._default_data_dir_path

                if not os.path.isdir(self.filter_directory):
                    if verbose: print(os.path.isdir(self.filter_directory))
                    raise PathError("The default data directory '" + self.filter_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default filter directory '" + self._default_filter_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_filter_dir_path?")
            pass


    def _sort_phot(self):
        """
        resorts the photometry according to effective wavelength of the filter.

        Parameters
        ----------

        Returns
        -------

        """
        if hasattr(self, "data") and hasattr(self, "data_filters"):
            ## This looks fugly.
            newkeys = np.array([i for i in self.data_filters.keys()])[np.argsort([self.data_filters[i].lambda_effective.value for i in self.data_filters])]

            sorted_data = OrderedDict()
            sorted_data_filters = OrderedDict()

            for newkey in newkeys:
                sorted_data[newkey] = self.data[newkey]
                sorted_data_filters[newkey] = self.data_filters[newkey]

            self.data = sorted_data
            self.data_filters = sorted_data_filters

        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def unpack(self, filter_file_type=".dat", verbose=False):
        """
        If loading from preformatted file, then unpack the table into self.data
        OrderedDict and load FilterClass objects into self.data_filters OrderedDict

        Parameters
        ----------

        Returns
        -------

        """

        if hasattr(self, "phot"):
            filter_names = np.unique(self.phot["filter"])

            self.phot.add_index('filter', unique = True)


            for filter_name in filter_names:

                phot_table = self.phot.loc["filter", filter_name]
                filter_filename = filter_name + filter_file_type
                if verbose: print(filter_filename)
                if verbose: print(phot_table)
                if verbose: print(type(filter_name), type(filter_file_type))

                # phot_table.meta = {"filter_filename": filter_filename}
                phot_table.meta["filter_filename"] = filter_filename
                if not isinstance(phot_table, Row):
                # if len(np.unique(self.phot.loc["filter", filter_name]["MJD"])) > 1:
                    indices = phot_table.argsort("MJD")
                    # for column_name in phot_table.colnames:
                    #     phot_table[column_name] = phot_table[column_name][indices]
                    sorted_phot_table = Table([phot_table[column_name][indices] for column_name in phot_table.colnames])
                else:
                    sorted_phot_table = phot_table

                filter_key = np.unique(phot_table["filter"])[0]

                if len(np.unique(phot_table["filter"])) > 1 or filter_key != filter_name:
                    raise FilterMismatchError("There is a more than one filterdata in here! or there is a mismatch with filename")
                path_to_filter = os.path.join(self.filter_directory, phot_table.meta['filter_filename'])

                # def load_filter(path, cmap = False, verbose = False):
                #
                if check_file_path(os.path.abspath(path_to_filter)):
                    filter_object = FilterClass()
                    filter_object.read_filter_file(os.path.abspath(path_to_filter), verbose = verbose)
                    filter_object.calculate_AB_zp()
                else:
                    warnings.warn("Couldn't load the filter")

                self.data_filters[filter_key] = filter_object

                self.data[filter_name] = sorted_phot_table

            self.filter_names = filter_names

        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")

        pass


    def load_table(self, phot_table, verbose=False):
        """
        Loads a single photometry table.

        Parameters
        ----------
        Returns
        -------
        """
        # StringWarning(path)
        try:
            self.phot = phot_table
            self.unpack(verbose=verbose)

            ## Sort the OrderedDict
            self._sort_phot()
        except:
            raise Exception


    def load_phot_dict(self, data_dict):
        """

        """
        self.data = data_dict
        pass


    def _combine_phot(self, verbose = True):
        """

        """

        if hasattr(self, "data"):
            if verbose: print(self.data.keys())

            for i, phot_filter in enumerate(self.data.keys()):

                if verbose: print(i, phot_filter)

                if i == 0:

                    full_phot = self.data[phot_filter]

                else:

                    full_phot = vstack([full_phot, self.data[phot_filter]])

                    pass

            self.data['full'] = full_phot

        else:
            warnings.warn("Cant find self.data")

        pass


    # def _phot_format_for_save(self, filters = False, verbose = False):
    #     """
    #     This is hacky - clear it up!
    #
    #     Parameters
    #     ----------
    #     Returns
    #     -------
    #     """
    #
    #     if not filters:
    #         ## if none specified, use all filters
    #         filters = self.data.keys()
    #
    #     w = np.array([])
    #     for i, f in enumerate(filters):
    #         w = np.append(w, np.where(self.phot["filter"] == f))
    #     if verbose: print(w)
    #
    #     save_table = self.phot["MJD", "flux", "flux_err", "filter"][w.astype(int)]
    #     save_table['MJD'].format = "5.5f"
    #     save_table['flux'].format = "5.5e"
    #     save_table['flux_err'].format = "5.5e"
    #     # save_table[save_table.argsort("MJD")]
    #     return save_table


    def _phot_format_for_save(self, names = ('MJD', 'flux', 'flux_err', 'filter'), filters = False, verbose = False, sort=False):
            """
            This is hacky - clear it up!

            Parameters
            ----------
            Returns
            -------
            """


            if sort:
                save_table = self.phot
                save_table = save_table[save_table.argsort()]
            else:
                save_table = self.phot

            return save_table


    def save(self, filename, filters = False, path = False,
             names = ('MJD', 'flux', 'flux_err', 'filter'),
             squash = False, verbose = True, *args, **kwargs):
        """
        Output the photometry loaded into the SNClass via self.load_phot* into a format
        and location recognised by CoCo.

        Parameters
        ----------
        Returns
        -------
        """

        if hasattr(self, "data"):
            if verbose: print("has data")
            if not path:
                if verbose: print("No directory specified, assuming " + self._default_data_dir_path)
                path = self._default_data_dir_path
            else:
                StringWarning(path)

            check_dir_path(path)

            outpath = os.path.join(path, filename)

            if verbose: print(outpath)
            if not filters:
                ## if none specified, use all filters
                filters = self.data.keys()

            if os.path.isfile(outpath):
                warnings.warn("Found existing file matching " + outpath + ". Run with squash = True to overwrite")
                if squash:
                    print("Overwriting " + outpath)
                    self._phot_format_for_save(filters = filters).write(outpath, format = "ascii.fast_commented_header", overwrite = True, names=names)
            else:
                    print("Writing " + outpath)
                    self._phot_format_for_save(filters = filters).write(outpath, format = "ascii.fast_commented_header", names=names)

        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


class BaseLCModelClass():
    """

    """

    def __init__(self, model_name):
        if model_name in _defined_models:

            if model_name == "bazin09":
                self.function = bazin09_listarg
                self.nparams = 4
                self._paramnames = ["a", "t_0", "t_rise", "t_fall"]

            elif model_name == "karpenka12":
                self.function = karpenka12_listarg
                self.nparams = 6
                self._paramnames = ["a", "b", "t_0", "t_1", "t_rise", "t_fall"]

            elif model_name == "firth17":
                self.function = firth17_listarg
                self.nparams = 8
                self._paramnames = ["a", "b", "t_0", "t_1", "t_2", "t_x", "t_rise", "t_fall"]

        else:
            warnings.warn("Model Not Recognised.")
        # self.fit_params = OrderedDict
        pass

    def load_bestfitparams(self, param_array):
        """
        pass array where the keys are the param names.
        """

        self.params = OrderedDict()

        for i, element in enumerate(param_array):
            self.params[self._paramnames[i]] = element
        pass

    def evaluate(self, t):

        self.fit = self.function(t, [self.params[p] for p in self.params])
        pass


class BaseFilterClass():
    """

    """

    def __init__(self, verbose = True):
        """

        :param verbose:
        """
        self._wavelength_units = u.Angstrom
        self._wavelength_units._format['latex'] = r'\rm{\AA}'
        self._frequency_units = u.Hertz
        # self.calculate_frequency()
        # self.calculate_effective_frequency()
        pass


    def calculate_filter_area(self):
        """

        :return:
        """
        if hasattr(self, "throughput"):
            self._effective_area = simps(self.throughput, self.wavelength)


    def calculate_AB_zp(self, ABpath = os.path.join(_default_kcorr_data_path, "AB_pseudospectrum.dat"), wmin = 1500 * u.angstrom, wmax=25000 * u.angstrom):
        """
        """


        AB = SpectrumClass()
        AB.load(ABpath, wmin=wmin, wmax=wmax)

        if not hasattr(self, "lambda_effective"):
            self.calculate_effective_wavelength()

        self.resample_response(new_wavelength=AB.wavelength)

        transmitted_spec = self.throughput * AB.flux
        integrated_flux = simps(transmitted_spec, AB.wavelength)

        if not hasattr(self, "_effective_area"):
            self.calculate_filter_area()

        area_corr_integrated_flux = integrated_flux / self._effective_area

        self.zp_AB = -2.5 * log10(area_corr_integrated_flux)
        pass


    def calculate_effective_wavelength(self):
        """
        Well, what are you expecting something called `calculate_effective_wavelength`
         to do?
        """

        spline_rev = interp1d((np.cumsum(self.wavelength*self.throughput)/np.sum(self.wavelength*self.throughput)), self.wavelength)
        lambda_eff = spline_rev(0.5)

        self.lambda_effective = lambda_eff * self._wavelength_units
        pass


    def calculate_frequency(self):
        nu = c/self.wavelength_u
        self.frequency_u = nu.to(self._frequency_units)
        self.frequency = self.frequency_u.value


    def calculate_effective_frequency(self):
        """

        """

        if hasattr(self, "frequency"):
            spline_rev = interp1d((np.cumsum(self.frequency*self.throughput)/np.sum(self.frequency*self.throughput)), self.frequency)
            nu_eff = spline_rev(0.5)

            self.nu_effective = nu_eff * self._frequency_units
        pass


    def plot(self, xminorticks = 250, yminorticks = 0.1,
             show_lims = False, small = False, cumulative = False,
             *args, **kwargs):
        """
        Plots filter throughput, so you can double check it.

        Parameters
        ----------

        Returns
        -------
        """

        ## Check if there is something in the class to plot
        if hasattr(self, "wavelength") and hasattr(self, "throughput"):

            setup_plot_defaults()
            if hasattr(self._wavelength_units, "format"):
                if "latex" in self._wavelength_units.format:
                    xaxis_label_string = r'$\textnormal{Wavelength, ' + self._wavelength_units.name + ' (}' + self._wavelength_units._format['latex'] +')$'
            else:
                xaxis_label_string = r'$\textnormal{Wavelength, ' + self._wavelength_units.name + '}$'

            plot_label_string = r'$\textnormal{' + self.filter_name.replace('_', '\\_') + '}$'

            yminorLocator = MultipleLocator(yminorticks)
            xminorLocator = MultipleLocator(xminorticks)

            if not small:
                fig = plt.figure(figsize=[8, 4])
            else:
                fig = plt.figure(figsize=[4, 2])
                plt.rcParams['font.size'] = 10

            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            if cumulative:
                throughput = np.cumsum(self.throughput)/np.sum(self.throughput)
                yaxis_label_string = r'$\textnormal{Cumulative Throughput}$'

            else:
                throughput = self.throughput
                yaxis_label_string = r'$\textnormal{Fractional Throughput}$'


            if hasattr(self, "_plot_colour"):
                ax1.plot(self.wavelength, throughput, color = self._plot_colour,
                         lw = 2, label = plot_label_string)
            else:
                ax1.plot(self.wavelength, throughput, lw = 2, label = plot_label_string)

            if show_lims:
                try:
                    ax1.plot([self._upper_edge, self._upper_edge], [0,1] ,
                             lw = 1.5, alpha = 0.5, ls = ':',
                             color = hex['batman'], zorder = 0, )
                    ax1.plot([self._lower_edge, self._lower_edge], [0,1] ,
                             lw = 1.5, alpha = 0.5, ls = ':',
                             color = hex['batman'], zorder = 0, )
                except:
                    print("Failed")

            ax1.spines['top'].set_visible(True)

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.yaxis.set_minor_locator(yminorLocator)
            ax1.xaxis.set_minor_locator(xminorLocator)

            ax1.legend(loc = 0)

            plt.show()
            pass
        else:
            warnings.warn("Doesn't look like you have loaded a filter into the object")


    def resample_response(self, new_wavelength = False, k = 1,
                          *args, **kwargs):
        """
        Bit dodgy - spline has weird results for poorly sampled filters.
        Now the order is by default 1, seems to be less likely to introduce artifacts

        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "wavelength") and hasattr(self, "throughput"):
            self._wavelength_orig = self.wavelength
            self._throughput_orig = self.throughput

            self.wavelength = np.concatenate(([0,1], self._wavelength_orig, [24999,25000]))
            self.throughput = np.concatenate(([0,0], self._throughput_orig, [0,0]))

            interp_func = InterpolatedUnivariateSpline(self.wavelength, self.throughput, k = k,
                                                       *args, **kwargs)
            self.throughput = interp_func(new_wavelength)
            self.wavelength = new_wavelength
            # self.wavelength.name = "wavelength"

            self.throughput[np.where(self.throughput < 0.0)] = 0.0
        else:
            warning.warn("Doesn't look like you have loaded a filter into the object")


    def load(self, path, directory = False, fmt = "ascii.commented_header",
             names = ("wavelength", "throughput"), wavelength_u = u.angstrom,
             verbose = False, name = False):
        """
        Assumes Response function is fractional rather than %.

        Parameters
        ----------

        Returns
        -------
        """


        if check_file_path(os.path.abspath(path), verbose = verbose):

            self.data = Table.read(path, format = fmt, names = names)

            self.wavelength = self.data["wavelength"]*wavelength_u
            self.wavelength = self.wavelength.to(u.angstrom)
            self.data["wavelength"] = self.wavelength
            self.throughput = self.data["throughput"]

            self.wavelength_u = self.wavelength * wavelength_u
            self._wavelength_units = wavelength_u

            self._filter_file_path = path

            if name:
                self.filter_name = name

            filename = path.split('/')[-1]
            filename_no_extension = filename.split('.')[0]

        else:
            warnings.warn("Foo")


    def load_table(self, table, name,  directory = False, wavelength_u = u.angstrom,
             verbose = False):
        """
        Parameters
        ----------

        Returns
        -------
        """

        """
        Assumes Response function is fractional rather than %.
        """

        self.filter_name = name

        self.data = table

        if not hasattr(table["wavelength"], "unit"):
            self.wavelength = self.data["wavelength"]*wavelength_u
        else:
            self.wavelength = self.data["wavelength"]

        self.wavelength = self.wavelength.to(u.angstrom)
        self.data["wavelength"] = self.wavelength
        self.throughput = self.data["throughput"]

        self.wavelength_u = self.wavelength.to(wavelength_u)
        self._wavelength_units = wavelength_u


    def save(self, filename, path = False,
         squash = False, verbose = True, *args, **kwargs):
        """
        Output the filter loaded into the Class into a format
        and location recognised by CoCo.

        based on BaseSpectrumClass.save

        Parameters
        ----------
        Returns
        -------
        """

        if hasattr(self, "wavelength") and hasattr(self, "throughput"): ## enables resampling and wavelength conversion to be easily saved
            if verbose: print("has data")
            if not path:
                if verbose: print("No directory specified, assuming " + _default_filter_dir_path)
                path = _default_filter_dir_path
            else:
                StringWarning(path)

            outpath = os.path.join(path, filename)

            check_dir_path(path)

            if os.path.isfile(outpath):
                warnings.warn("Found existing file matching " + path + ". Run with squash = True to overwrite")
                if squash:
                    print("Overwriting " + outpath)
                    outtable = Table([self.wavelength, self.throughput], names = ["wavelength", "throughput"])
                    outtable.write(outpath, format = "ascii.fast_commented_header", overwrite = True)
                    self._format_for_save = outtable


            else:
                    print("Writing " + outpath)
                    outtable = Table([self.wavelength, self.throughput], names = ["wavelength", "throughput"])

                    outtable.write(outpath, format = "ascii.fast_commented_header")
                    self._format_for_save = outtable

        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass




#  #------------------------------------#  #
#  #  Inheriting Classes                #  #
#  #------------------------------------#  #


class PhotometryClass(BaseLightCurveClass):
    """
    Inherits from BaseLightCurveClass

    Probably also overkill - but should be easier to store metadata etc. Hopefully
    flexible enough to just be a wrapper for AP tables of phot.

    Photometry stored in PhotometryClass.data should have a FilterClass method
    describing the observations stored in PhotometryClass.data_filters.

    ## NOTE should I use properties instead of get/set? http://www.python-course.eu/python3_properties.php
    looks like only python3?
    """

    def __init__(self, verbose = False):
        """

        """

        ## Initialise the class variables
        self._default_data_dir_path = os.path.join(_default_data_dir_path, 'lc/')
        self._default_filter_dir_path = _default_filter_dir_path
        self.data = OrderedDict()
        self.data_filters = OrderedDict()

        ## Initialise using class methods
        self.set_data_directory(self._default_data_dir_path)
        self.set_filter_directory(self._get_filter_directory())


    def _get_data_directory(self):
        """
        Get the default path to the data directory.

        Looks for the data data directory set as environment variable
        $PYCOCO_DATA_DIR. if not found, returns default.

        returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
                 default datalocation: '../testdata/', with '/lc/' appended.
        """

        return self.data_directory


    def set_data_directory(self, data_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if verbose: print(data_dir_path, self._default_data_dir_path)
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


    def load(self, path, names = ('MJD', 'flux', 'flux_err', 'filter'),
                  format = 'ascii.commented_header', verbose = True):
        """
        Loads a single photometry file.

        Parameters
        ----------
        Returns
        -------
        """
        StringWarning(path)
        try:
            phot_table = self._load_formatted_phot(path, names = names, format = format, verbose = verbose)
            self.phot = phot_table
            self.unpack()

            ## Sort the OrderedDict
            self._sort_phot()
        except:
            raise Exception


    def _load_formatted_phot(self, path, format = "ascii", names = False,
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

        phot_table.meta["filename"] = path

        phot_table["MJD"].unit = u.day
        phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
        phot_table["flux_err"].unit =  phot_table["flux"].unit

        return phot_table


    def load_phot_from_file(self, path, names = ('MJD', 'flux', 'flux_err', 'filter'),
                  format = 'ascii', verbose = True):
        """
        For single filter data
        """
        StringWarning(path)
        try:
            # phot_table = load_phot(path, names = names, format = format, verbose = verbose)
                # phot_table = ap.table.Table.read(path, format = format, names = names)
            phot_table = Table.read(path, format = format, names = names)

            phot_table.replace_column("MJD", Time(phot_table["MJD"], format = 'mjd'))

            phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
            phot_table["flux_err"].unit =  phot_table["flux"].unit

            self.data[np.unique(phot_table["filter"])[0]] = phot_table

            ## Sort the OrderedDict
            self._sort_phot()
        except:
            raise Exception

        pass


    def load_phot_ap_tables(self):
        """

        """

        pass


    def load_phot_from_files(self, path = False, snname = False, prefix = 'SN',
             file_type = '.dat', names = ('MJD', 'flux', 'flux_err', 'filter'),
             format = 'ascii', filter_file_type = '.dat', verbose = False):
        """
        Finds and loads in data (from file) into phot objects.

        Parameters
        ----------

        Returns
        -------

        """

        if snname:
            if not path:
                path = self._default_data_dir_path
            ## Find matching photometry
            phot_list = find_filter_phot(path = path, snname = snname, prefix = prefix,
                              file_type = file_type, verbose = verbose)

            full_phot_table = Table()

            ## Loop over files (shouldn't be that many really)
            if len(phot_list) > 0:

                for phot_file in phot_list:

                    if verbose: print(phot_file)
                    phot_table = Table.read(phot_file, names = names, format = format)

                    ## NOTE astropy vstack does not support mixin columns http://docs.astropy.org/en/stable/table/mixin_columns.html
                    # This means I might have problems joining the tables together if I don't add together as I go along.

                    full_phot_table = vstack([full_phot_table, phot_table])

                    filter_string = get_filter_from_filename(phot_file, snname, file_type)
                    phot_table.meta = {"filename" : phot_file,
                                       "filter" : filter_string,
                                       "filter_filename": filter_string + filter_file_type}

                    ## Sort out units
                    phot_table.sort("MJD")
                    phot_table["t"] = Time(phot_table["MJD"], format = 'mjd')

                    phot_table["MJD"].unit = u.day
                    phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
                    phot_table["flux_err"].unit =  phot_table["flux"].unit

                    ## Put in dictionary - use filter from the file
                    filter_key = np.unique(phot_table["filter"])[0]
                    if verbose: print(len(np.unique(phot_table["filter"])) , phot_table.meta["filter"], filter_key)

                    if len(np.unique(phot_table["filter"])) > 1 or filter_key != phot_table.meta["filter"]:
                        raise FilterMismatchError("There is a mismatch between the filter filename and that in the "
                                                   + "photometry file")

                    self.data[filter_key] = phot_table

                    path_to_filter = os.path.join(self.filter_directory, phot_table.meta['filter_filename'])
                    self.data_filters[filter_key] = load_filter(path_to_filter)


                ## NOTE doing it this way because vstack doesn't like mixin columns (see above comment)
                full_phot_table.sort("MJD")
                # full_phot_table["t"] = Time(full_phot_table["MJD"], format = 'mjd')
                full_phot_table["MJD"].unit = u.day

                full_phot_table["flux"].unit = u.cgs.erg / u.si.angstrom / u.si.cm ** 2 / u.si.s
                full_phot_table["flux_err"].unit =  full_phot_table["flux"].unit

                self.phot = full_phot_table

                ## Sort the OrderedDict
                self._sort_phot()
            else:
                warning.warn("Couldn't find any photometry")
        else:
            warnings.warn("Provide a SN name")

        pass


    def _combine_phot(self, verbose = True):
        """

        """

        if hasattr(self, "data"):
            if verbose: print(self.data.keys())

            for i, phot_filter in enumerate(self.data.keys()):

                if verbose: print(i, phot_filter)

                if i == 0:

                    full_phot = self.data[phot_filter]

                else:

                    full_phot = vstack([full_phot, self.data[phot_filter]])

                    pass

            self.data['full'] = full_phot

        else:
            warnings.warn("Cant find self.data")

        pass


    def plot(self, filters=False, legend=True, xminorticks=5, enforce_zero = True,
             verbose=False, xlim=False, yaxis_max_factor=1.02, *args, **kwargs):
        """
        Plots phot.

        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):

            if not filters:
                filters = self.data_filters
            if type(filters) == str:
                filters = [filters]
            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.93,
                                right = 0.96, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            for i, filter_key in enumerate(filters):
                if verbose: print(i, self.data[filter_key].__dict__)
                plot_label_string = r'$\rm{' + self.data_filters[filter_key].filter_name.replace('_', '\\_') + '}$'
                if filter_key in hex.keys():
                    self.data_filters[filter_key]._plot_colour = hex[filter_key]
                else:
                    warnings.warn("Cannot find filter in the pycoco colours registry")
                    self.data_filters[filter_key]._plot_colour = "C0"

                ax1.errorbar(self.data[filter_key]['MJD'], self.data[filter_key]['flux'],
                             yerr = self.data[filter_key]['flux_err'],
                             capsize = 0, fmt = 'o', color = self.data_filters[filter_key]._plot_colour,
                             label = plot_label_string, ecolor = hex['batman'], mec = hex["batman"],
                             *args, **kwargs)

            if legend:

                # plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                #                       numpoints = 1, frameon = False, fontsize = 12)
                plot_legend = ax1.legend(loc = 1, scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)
            ## Use ap table groups instead? - can't; no support for mixin columns.
            if enforce_zero:
                ax1.set_ylim(0., yaxis_max_factor * np.nanmax(self.phot['flux']))
            else:
                ax1.set_ylim(np.nanmin(self.phot['flux']), yaxis_max_factor * np.nanmax(self.phot['flux']))

            if xlim:
                ax1.set_xlim(xlim)

            ## Label the axes
            xaxis_label_string = r'$\textnormal{Time, MJD (days)}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'
            # yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{cm}^{-2}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.spines['top'].set_visible(True)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def plot_filters(self, xminorticks = 250, yminorticks = 0.1,
                     legend = True, use_cmap = False, verbose = False):
        """
        Plots filters.

        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self, "data_filters"):

            setup_plot_defaults()
            xaxis_label_string = r'$\textnormal{Wavelength, (\AA)}$'
            yaxis_label_string = r'$\textnormal{Fractional Throughput}$'
            yminorLocator = MultipleLocator(yminorticks)
            xminorLocator = MultipleLocator(xminorticks)

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)
            ax1 = fig.add_subplot(111)

            ## Plot the throughput for each filter
            for i, filter_key in enumerate(self.data_filters):
                if verbose: print(i, self.data_filters[filter_key].__dict__)
                plot_label_string = r'$\rm{' + self.data_filters[filter_key].filter_name.replace('_', '\\_') + '}$'
                if hasattr(self.data_filters[filter_key], "_plot_colour") and use_cmap:
                    ax1.plot((self.data_filters[filter_key].wavelength_u).to(u.angstrom),
                             self.data_filters[filter_key].throughput,
                             color = self.data_filters[filter_key]._plot_colour,
                             lw = 2, label = plot_label_string)
                else:
                    ax1.plot((self.data_filters[filter_key].wavelength_u).to(u.angstrom),
                             self.data_filters[filter_key].throughput,
                             lw = 2, label = plot_label_string)
            # if hasattr(self, "_plot_colour"):
            #     ax1.plot(self.wavelength, self.throughput, color = self._plot_colour,
            #              lw = 2, label = plot_label_string)
            # else:
            #     ax1.plot(self.wavelength, self.throughput, lw = 2, label = plot_label_string)

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.yaxis.set_minor_locator(yminorLocator)
            ax1.xaxis.set_minor_locator(xminorLocator)

            if legend:
                plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any filters here (empty self.filter_data)")
        pass


class SpectrumClass(BaseSpectrumClass):
    """
    Class for handling Spectra.
    Inherits from BaseSpectrumClass.
    """

    def __init__(self):
        """

        """

        ## Initialise the class variables
        self._default_data_dir_path = os.path.join(_default_data_dir_path, "spec/")
        # self._default_list_dir_path = self._default_data_dir_path

        ## Initialise using class methods
        self.set_data_directory(self._default_data_dir_path)

        pass


    def _get_data_directory(self, path=False):
        """
        Get the default path to the data directory.

        Looks for the data data directory set as environment variable
        $PYCOCO_DATA_DIR. if not found, returns default.

        returns: Absolute path in environment variable $PYCOCO_DATA_DIR, or
                 default datalocation: '../testdata/', with '/spec/' appended.
        """

        return self.data_directory


    def set_data_directory(self, data_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if verbose: print(data_dir_path, self._default_data_dir_path)
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


class LCfitClass(BaseLightCurveClass):
    """
    Small class to hold the output from CoCo LCfit.
    Inherits from BaseLightCurveClass
    """

    def __init__(self):

        ## Initialise the class variables
        self._default_recon_dir_path = os.path.join(_default_coco_dir_path, "recon/")
        self._default_filter_dir_path = _default_filter_dir_path

        ## Initialise using class methods
        self.set_recon_directory(self._get_recon_directory())
        self.set_filter_directory(self._get_filter_directory())

        ## Initialise some other stuff
        self.data = OrderedDict()
        self.data_filters = OrderedDict()

        pass


    def _get_recon_directory(self):
        """
        Get the default path to the data directory.

        Looks for the CoCo home directory set as environment variable
        $COCO_ROOT_DIR. if not found, returns default.

        returns: Absolute path in environment variable $COCO_ROOT_DIR, or
                 default CoCo location: '~/Code/CoCo/', with 'recon/' appended.
        """

        return os.path.join(self._default_recon_dir_path, os.path.pardir, "recon/")


    def set_recon_directory(self, recon_dir_path = '', verbose = False):
        """
        Set a new recon directory path.

        Enables the recon directory to be changed by the user.

        """
        try:
            if verbose: print(recon_dir_path, self._default_recon_dir_path)
            if os.path.isdir(os.path.abspath(recon_dir_path)):
                self.recon_directory = os.path.abspath(recon_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(recon_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_recon_dir_path, UserWarning)
                self.recon_directory = self._default_recon_dir_path

                if not os.path.isdir(self.recon_directory):
                    if verbose: print(os.path.isdir(self.recon_directory))
                    raise PathError("The default recon directory '" + self.recon_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default recon directory '" + self._default_recon_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_recon_dir_path?")
            pass


    def load_formatted_phot(self, path, names = ('MJD', 'flux', 'flux_err', 'filter'),
                  format = 'ascii', verbose = True):
        """

        """
        StringWarning(path)

        try:
            phot_table = load_formatted_phot(path, format = format, names = names,
                                             verbose = verbose)
            self.phot = phot_table

            self.phot['flux_upper'] = phot_table['flux'] + phot_table['flux_err']
            self.phot['flux_lower'] = phot_table['flux'] - phot_table['flux_err']

        except:
            raise Exception

        pass


    def plot(self, legend = True, xminorticks = 5,
             verbose = False, *args, **kwargs):
        """
        Plots phot.

        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            for i, filter_key in enumerate(self.data_filters):
                if verbose: print(i, self.data[filter_key].__dict__)
                plot_label_string = r'$\rm{' + self.data_filters[filter_key].filter_name.replace('_', '\\_') + '}$'

                # ax1.errorbar(self.data[filter_key]['MJD'], self.data[filter_key]['flux'],
                #              yerr = self.data[filter_key]['flux_err'],
                #              capsize = 0, fmt = 'o',
                #              label = plot_label_string,
                #              *args, **kwargs)

                # ## Best Fit
                # ax1.plot(self.data[filter_key]['MJD'], self.data[filter_key]['flux'],
                #          lw = 2, label = plot_label_string,
                #           *args, **kwargs)

                ## With error
                ax1.fill_between(self.data[filter_key]['MJD'], self.data[filter_key]['flux_upper'], self.data[filter_key]['flux_lower'],
                                 label = plot_label_string, color = self.data_filters[filter_key]._plot_colour,
                                 alpha = 0.8,
                                 *args, **kwargs)
            if legend:

                plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            ## Use ap table groups instead? - can't; no support for mixin columns.
            ax1.set_ylim(np.nanmin(self.phot['flux']), np.nanmax(self.phot['flux']))

            ## Label the axes
            xaxis_label_string = r'$\textnormal{Time, MJD (days)}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def get_fit_splines(self, verbose = False):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if hasattr(self, "data"):
            self.spline = OrderedDict()

            for i, filter_key in enumerate(self.data):
                try:
                    if verbose: print(filter_key)
                    self.spline[filter_key] = InterpolatedUnivariateSpline(self.data[filter_key]["MJD"], self.data[filter_key]["flux"])
                    self.spline[filter_key+"_err"] = InterpolatedUnivariateSpline(self.data[filter_key]["MJD"], self.data[filter_key]["flux_err"])
                except:
                    print("NOPE")
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def colour_from_model(self, filter_key1, filter_key2):

        return phot_1 - phot_2


class specfitClass(BaseSpectrumClass):
    """
    Small class to hold the output from CoCo spec.
    Inherits from BaseSpectrumClass.

    """

    def __init__(self):
        """

        """

        ## Initialise the class variables
        self._default_recon_dir_path = os.path.join(_default_coco_dir_path, "recon/")
        # self._default_list_dir_path = self._default_data_dir_path

        ## Initialise using class methods
        self.set_recon_directory(self._get_recon_directory())

        pass


    def _get_recon_directory(self):
        """
        Get the default path to the recon directory.

        Looks for the CoCo directory set as environment variable
        $COCO_ROOT_DIR. if not found, returns default.

        returns: Absolute path in environment variable $COCO_ROOT_DIR, or
                 default datalocation: '../testdata/', with '/spec/' appended.
        """

        return os.path.join(os.environ.get('COCO_ROOT_DIR', os.path.join(self._default_recon_dir_path, os.pardir)), "recon/")


    def set_recon_directory(self, recon_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if verbose: print(recon_dir_path, self._default_recon_dir_path)
            if os.path.isdir(os.path.abspath(recon_dir_path)):
                self.recon_directory = os.path.abspath(recon_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(recon_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_recon_dir_path, UserWarning)
                self.recon_directory = self._default_recon_dir_path

                if not os.path.isdir(self.recon_directory):
                    if verbose: print(os.path.isdir(self.recon_directory))
                    raise PathError("The default data directory '" + self.recon_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default data directory '" + self._default_recon_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_recon_dir_path?")
            pass


    def set_orig_specpath(self, orig_specpath = False, verbose = False):
        """
        Parameters
        ----------
        Returns
        -------
        """

        if not orig_specpath:
            self.orig_specpath = self.data.meta["comments"][0].split("/")[-1]

        else:
            self.orig_specpath = orig_specpath

        pass


    def plot_comparison(self, SpectrumClassInstance,
                         xminorticks=250, legend=True,
                         verbose=True, twoaxes=True,
                         *args, **kwargs):
        """
        Plots spec.

        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, "data"):

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.94, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            if verbose: print(self.data.__dict__)
            plot_label_string = r'$\rm{' + self.data.meta["filename"].replace('_', '\_') + '}$'
            plot_label_string_compare = r'$\rm{' + SpectrumClassInstance.data.meta["filename"].replace('_', '\_') + '}$'


            ax1.plot(self.data['wavelength'], self.flux, lw = 2,
                         label = plot_label_string, color = 'Red',
                         *args, **kwargs)
            if twoaxes:
                ax2 = ax1.twinx()
                ax2.plot(SpectrumClassInstance.data['wavelength'], SpectrumClassInstance.data['flux'],
                         label = plot_label_string_compare, color = 'Blue',
                         *args, **kwargs)

            else:
                ax1.plot(SpectrumClassInstance.data['wavelength'], SpectrumClassInstance.data['flux'],
                         label = plot_label_string_compare, color = 'Blue',
                         *args, **kwargs)

            maxplotydata = np.nanmax(np.append(self.flux, SpectrumClassInstance.data['flux']))
            minplotydata = np.nanmin(np.append(self.flux, SpectrumClassInstance.data['flux']))

            if legend:
                ## https://stackoverflow.com/a/10129461
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                # plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                ax1.legend(lines + lines2,labels + labels2, loc=0, scatterpoints=1,
                                        numpoints = 1, frameon = False, fontsize = 12)

            ax1.set_ylim(minplotydata*0.98, maxplotydata*1.02)

            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


class FilterClass(BaseFilterClass):
    """Docstring for FilterClass inherits from BaseFilterClass"""

    def __init__(self):
        self._wavelength_units = u.Angstrom
        self._wavelength_units._format['latex'] = r'\rm{\AA}'
        self._frequency_units = u.Hertz
        pass


    def read_filter_file(self, path, fmt = "ascii",
                         names = ("wavelength", "throughput"),
                         wavelength_u = u.angstrom, verbose = False):
        """
        Assumes Response function is fractional rather than %.
        """
        if check_file_path(os.path.abspath(path), verbose = verbose):
            self.data = Table.read(path, format = fmt, names = names)
            self.wavelength = self.data["wavelength"] * wavelength_u
            if verbose: print("1", np.nanmax(self.wavelength))
            self.wavelength = self.wavelength.to(u.angstrom)
            self.throughput = self.data["throughput"]
            if verbose: print("2", np.nanmax(self.wavelength))

            self.wavelength_u = self.wavelength.to(wavelength_u)
            self._filter_file_path = path
            if verbose: print("3", np.nanmax(self.wavelength))

            filename = path.split('/')[-1]
            filename_no_extension = filename.split('.')[0]
            self.filter_name = filename_no_extension
            if verbose: print("4", np.nanmax(self.wavelength))

            self.set_plot_colour(verbose = verbose)
            if verbose: print("5", np.nanmax(self.wavelength))
            self.calculate_effective_wavelength()
            if verbose: print("6", np.nanmax(self.wavelength))
            self.calculate_edges()
            if verbose: print("7", np.nanmax(self.wavelength))
            self.get_zeropoint()
            if verbose: print("8", np.nanmax(self.wavelength))

        else:
            warnings.warn("Foo")


    def calculate_edges_zero(self, verbose = False):
        """
        calculates the first and last wavelength that has non-zero and steps one
         away

        Parameters
        ----------

        Returns
        -------
        """

        ## calculates the first and last wavelength that has non-zero
        # w = np.where(self.throughput > 0)[0]
        # if verbose: print(w)
        # self._upper_edge = self.wavelength[w[-1]]
        # self._lower_edge = self.wavelength[w[0]]

        w = np.where(self.throughput > 0)[0]
        if verbose: print(w)
        if w[0] - 1 < 0:
            w_low = 0
        else:
            w_low =  w[0] - 1

        if w[-1] + 1 == len(self.throughput):
            w_high = w[-1]
        else:
            w_high = w[-1] + 1

        self._upper_edge = self.wavelength[w_high]
        self._lower_edge = self.wavelength[w_low]


    def calculate_edges(self, pc = 3., verbose = True):
        """
        calculates edges by defining the region that contains (100 - pc)% of the
        flux.

        Parameters
        ----------

        Returns
        -------
        """
        self._cumulative_throughput = np.cumsum(self.throughput)/np.sum(self.throughput)
        self._cumulative_throughput_spline = interp1d(self._cumulative_throughput, self.wavelength)

        self._upper_edge = self._cumulative_throughput_spline(1.0 - 0.5*(0.01*pc))
        self._lower_edge = self._cumulative_throughput_spline(0.0 + 0.5*(0.01*pc))

        pass


    def calculate_plot_colour(self, colourmap_name = "plasma", verbose = False):
        """


        Parameters
        ----------

        Returns
        -------
        """

        if not hasattr(self, "_colourmap"):
            self._colourmap = plt.get_cmap(_colourmap_name)

        if hasattr(self, 'lambda_effective'):

            relative_lambda = self.lambda_effective - _colour_lower_lambda_limit
            relative_lambda = relative_lambda / _colour_lower_lambda_limit

            if verbose: print("relative_lambda = ", relative_lambda)

            self._plot_colour = self._colourmap(relative_lambda)

        else:
            warnings.warn("No self.lambda_effective set.")


    def set_plot_colour(self, colour = False, verbose = False):
        """


        Parameters
        ----------

        Returns
        -------
        """
        if colour:
            self._plot_colour = colour

        else:

            try:
                if verbose:
                    if self.filter_name in hex.keys:
                        print(hex[self.filter_name])
                self._plot_colour = hex[self.filter_name]
            except:
                if verbose: print("Nope")
                self.calculate_plot_colour(verbose = verbose)

        pass


    def get_zeropoint(self):
        """

        :return:
        """

        if hasattr(self, "filter_name"):
            # self.zp_AB = self.calculate_AB_zp()
            self.calculate_AB_zp()
            # self.zp_vega = self.calc_vega_zp(filter_name)
        else:
            warnings.warn("No filter name - have you loaded in a bandpass?")


#  #------------------------------------#  #
#  # Model Classes                      #  #
#  #------------------------------------#  #
#
# class (BaseLCModelClass)


#  #------------------------------------#  #
#  # Standalone Classes                 #  #
#  #------------------------------------#  #


class SNClass():
    """docstring for SNClass."""

    def __init__(self, snname):
        """
        Parameters
        ----------

        Returns
        -------
        """
        ## Initialise
        self.spec = OrderedDict()
        self.mangledspec = OrderedDict()
        # self.spec = SpectrumClass()
        self.phot = PhotometryClass()
        info = InfoClass()
        info.load()
        self.info = info.get_sn_info(snname)

        self.coco_directory = self._get_coco_directory()
        self.recon_directory = self._get_recon_directory()

        self.name = snname
        pass


    def _get_coco_directory(self):
        """
        Get the default path to the data directory.

        Looks for the CoCo home directory set as environment variable
        $COCO_ROOT_DIR. if not found, returns default.

        returns: Absolute path in environment variable $COCO_ROOT_DIR, or
                 default CoCo location: '~/Code/CoCo/', with appended.
        """

        return os.path.abspath(os.environ.get('COCO_ROOT_DIR', os.path.abspath(_default_coco_dir_path)))


    def _get_recon_directory(self):
        """
        Get the default path to the recon directory.

        Looks for the CoCo directory set as environment variable
        $COCO_ROOT_DIR. if not found, returns default.

        returns: Absolute path in environment variable $COCO_ROOT_DIR, or
                 default datalocation: '../testdata/', with '/spec/' appended.
        """

        return os.path.join(os.path.abspath(os.environ.get('COCO_ROOT_DIR', os.path.join(_default_recon_dir_path, os.pardir))), "recon/")


    def set_recon_directory(self, recon_dir_path = '', verbose = False):
        """
        Set a new data directory path.

        Enables the data directory to be changed by the user.

        """
        try:
            if verbose: print(recon_dir_path, self._default_recon_dir_path)
            if os.path.isdir(os.path.abspath(recon_dir_path)):
                self.recon_directory = os.path.abspath(recon_dir_path)
                pass
            else:
                warnings.warn(os.path.abspath(recon_dir_path) +
                " is not a valid directory. Restoring default path: " +
                self._default_recon_dir_path, UserWarning)
                self.recon_directory = self._default_recon_dir_path

                if not os.path.isdir(self.recon_directory):
                    if verbose: print(os.path.isdir(self.recon_directory))
                    raise PathError("The default data directory '" + self.recon_directory
                     + "' doesn't exist. Or isn't a directory. Or can't be located.")
                else:
                    pass
        except:
            if verbose: print("foo")
            raise PathError("The default data directory '" + self._default_recon_dir_path
             + "' doesn't exist. Or isn't a directory. Or can't be located. Have"
             + " you messed with _default_recon_dir_path?")
            pass


    def load_phot(self, phot_table = False, snname = False, path = False, file_type = '.dat',
                  verbose = False):
        """

        Parameters
        ----------

        Returns
        -------
        """

        if not snname:
            snname = self.name
        if phot_table:
            self.phot.load_table(phot_table=phot_table, verbose=verbose)
        else:
            if not path:
                path = os.path.join(self.phot._default_data_dir_path, snname + file_type)
            if verbose: print(path)
            self.phot.load(path, verbose = verbose)

        pass


    def load_list(self, path, specfiletype = ".txt", verbose = False):
        """
        Parameters
        ----------
        Returns
        -------
        """
        listdata = read_list_file(path, verbose=verbose)
        listdata.sort('mjd_obs')

        phases = []

        for item in listdata["spec_path"]:
            filename = item.split("/")[-1]
            filename = filename.split("_")[1:][0]
            filename = filename.strip(specfiletype)
            try:
                phase = float(filename)
            except:
                pass

            phases.append(phase)
            if verbose: print(phase)
        listdata["phase"] = phases

        self.list  = listdata


    def load_spec(self, snname = False, spec_dir_path = False, verbose = False):
        """
        Parameters
        ----------

        Returns
        -------
        """


        # if not snname:
        #     snname = self.name
        #
        # if not spec_dir_path:
        #     spec_dir_path = os.path.abspath(os.path.join(self._default_spec_data_dir_path, snname))
        #
        # if verbose: print("Loading spectra from: ", spec_dir_path)

        # spec_dir_path =


        if hasattr(self, 'coco_directory') and hasattr(self, 'list'):
            for i, path in enumerate(self.list['spec_path']):
                spec_fullpath = os.path.join(self.coco_directory, path)
                spec_filename = path.split('/')[-1]
                spec_dir_path = spec_fullpath.replace(spec_filename, '')
                if verbose: print(spec_fullpath, spec_dir_path, spec_filename)

                self.spec[spec_filename] = SpectrumClass()
                self.spec[spec_filename].load(spec_filename, directory = spec_dir_path,
                                              verbose = verbose)
                self.spec[spec_filename].set_MJD_obs(self.list['mjd_obs'][i])
                # self.spec[spec_filename].data.add_index('wavelength')

        else:
            warnings.warn("no coco or no listfile")
        pass


    def load_mangledspec(self, snname = False, spec_dir_path = False, verbose = False):
        """
        Parameters
        ----------

        Returns
        -------
        """

        if not snname:
            snname = self.name

        # self._mangledspeclist = find_recon_spec(snname)
        self._mangledspeclist = find_specphase_spec(self.name)
        self.mangledspec = OrderedDict()
        if verbose: print("loading mangledspec")
        if hasattr(self, 'recon_directory') and hasattr(self, '_mangledspeclist') and hasattr(self, "mangledspec"):
            for i, spec_filename in enumerate(self._mangledspeclist):

                if verbose: print(i, spec_filename)
                # self.mangledspec[spec_filename] = SpectrumClass()
                self.mangledspec[spec_filename] = specfitClass()
                self.mangledspec[spec_filename].load(spec_filename, directory = self.recon_directory,
                                              verbose = verbose)

                orig_specpath = self.mangledspec[spec_filename].data.meta['comments']
                orig_specname = orig_specpath
                print(orig_specpath)
                w = np.where(self.list["spec_path"] == orig_specpath)
                if verbose: print(w[0])

                if len(w[0]) > 0:
                    self.mangledspec[spec_filename].set_MJD_obs(self.list['mjd_obs'][w].data[0])
                    self.mangledspec[spec_filename].data.add_index('wavelength')
        #
        else:
            warnings.warn("no coco or no listfile")
        pass


    def load_sndist(self, path = _default_sn_dist_path, format = "ascii"):
        """
        based on read_sndist_file and load_sndist
        """

        if hasattr(self, "name"):
            # sndist = load_sndist(self.name)
            # self.z = sndist["z"].data[0]
            # self.distmod = sndist["mu"].data[0]

            check_file_path(path)
            sndistlist = Table.read(path, format = format)

            try:
                w = np.where(sndistlist["snname"] == snname)
                sndist = sndistlist[w]

                self.z = sndist["z"].data[0]
                self.distmod = sndist["mu"].data[0]
            except:
                warnings.warn("Failed to find distance info for " + snname + ". is it in the list?")
        else:
            if verbose: print("self.name not defined.")

        pass


    def plot_lc(self, filters = False, legend = True, xminorticks = 10, mark_spectra = True,
                simplespecphot = False, fade = False, xlims = False, insidelegend = True,
                fit = True, enforce_zero = True, multiplot = True, yaxis_lim_multiplier = 1.1,
                lock_axis = False, xextent = False, filter_uncertainty = 10,
                savepng = False, savepdf = False, outpath = False, showsnname = False,
                verbose = False, *args, **kwargs):
        """
        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self.phot, "data"):
            if not fade:
                alpha = 1.0
            else:
                alpha = 0.2

            if not filters:
                filters = self.phot.data_filters
            if type(filters) == str:
                filters = [filters]

            setup_plot_defaults()
            if not multiplot:
                fig = plt.figure(figsize=[8, 4])
            else:
                fig = plt.figure(figsize=[8, len(filters)*1.5])

            if showsnname:
                fig.suptitle(r"$\textrm{"+self.name+"}$")
                if verbose: print(self.name)

            fig.subplots_adjust(left = 0.1, bottom = 0.13, top = 0.93,
                                right = 0.91, hspace=0, wspace = 0)
            ## Label the axes
            xaxis_label_string = r'$\textnormal{Time, MJD (days)}$'
            yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{\AA}^{-1}\textnormal{cm}^{-2}$'
            # yaxis_label_string = r'$\textnormal{Flux, erg s}^{-1}\textnormal{cm}^{-2}$'


            if not multiplot:
                ax1 = fig.add_subplot(111)
                axes_list = [ax1]
            else:
                axes_list = [plt.subplot2grid((len(filters), 1), (j, 0)) for j, k in enumerate(filters)]

            for i, filter_key in enumerate(filters):
                if multiplot:
                    ax1 = axes_list[i]

                if filter_key in self.phot.data:
                    if verbose: print(i, self.phot.data[filter_key].__dict__)
                    plot_label_string = r'$\rm{' + self.phot.data_filters[filter_key].filter_name.replace('_', '\\_') + '}$'

                    if filter_key in hex.keys():
                        self.phot.data_filters[filter_key]._plot_colour = hex[filter_key]
                    else:
                        warnings.warn("Cannot find filter in the pycoco colours registry")
                        self.phot.data_filters[filter_key]._plot_colour = "C0"
                    ax1.errorbar(self.phot.data[filter_key]['MJD'], self.phot.data[filter_key]['flux'],
                                 yerr = self.phot.data[filter_key]['flux_err'],
                                 capsize = 0, fmt = 'o', color = self.phot.data_filters[filter_key]._plot_colour,
                                 label = plot_label_string, ecolor = hex['batman'], mec = hex["batman"],
                                 alpha = alpha,
                                 *args, **kwargs)

                    if fit and hasattr(self, 'lcfit'):
                        ax1.fill_between(self.lcfit.data[filter_key]['MJD'], self.lcfit.data[filter_key]['flux_upper'], self.lcfit.data[filter_key]['flux_lower'],
                                         color = self.phot.data_filters[filter_key]._plot_colour,
                                         alpha = 0.8, zorder = 0,
                                         *args, **kwargs)

                    if simplespecphot and hasattr (self, "simplespecphot"):
                        ax1.errorbar(self.simplespecphot.data[filter_key]['MJD'], self.simplespecphot.data[filter_key]['flux'],
                                     yerr = self.simplespecphot.data[filter_key]['flux_err'],
                                     capsize = 0, fmt = 'o', color = hex["batman"],
                                     ecolor = hex['batman'], mec = hex["batman"], label = r"$\textnormal{SpecPhot}$",
                                     *args, **kwargs)

                    if legend:
                        if multiplot or insidelegend:
                            plot_legend = ax1.legend(loc = 'upper right', scatterpoints = 1, markerfirst = False,
                                              numpoints = 1, frameon = False, bbox_to_anchor=(1., 1.),
                                              fontsize = 12.)

                    if i == len(axes_list)-1:
                        ax1.set_xlabel(xaxis_label_string)

                    else:
                        if multiplot:
                            ax1.set_xticklabels('')


                    xminorLocator = MultipleLocator(xminorticks)
                    ax1.spines['top'].set_visible(True)
                    ax1.xaxis.set_minor_locator(xminorLocator)

                    if mark_spectra:

                        for spec_key in self.spec:
                            if verbose: print(np.nanmin(self.spec[spec_key].wavelength) - filter_uncertainty, self.phot.data_filters[filter_key]._lower_edge)
                            if verbose: print(np.nanmax(self.spec[spec_key].wavelength) + filter_uncertainty, self.phot.data_filters[filter_key]._upper_edge)

                            if verbose: print(self.spec[spec_key].data.meta["filename"] )
                            too_blue =  self.phot.data_filters[filter_key]._lower_edge < np.nanmin(self.spec[spec_key].wavelength) - filter_uncertainty
                            too_red = self.phot.data_filters[filter_key]._upper_edge > np.nanmax(self.spec[spec_key].wavelength) + filter_uncertainty
                            # if self.spec[spec_key]. self.phot.data_filters[filter_key]._upper_edge and self.phot.data_filters[filter_key]._lower_edge
                            if verbose: print(too_blue, too_red)
                            if not too_red and not too_blue:
                                ax1.plot([self.spec[spec_key].mjd_obs, self.spec[spec_key].mjd_obs],
                                         [0.0, np.nanmax(self.phot.phot['flux'])*1.5],
                                         ls = ':', color = hex['batman'], zorder = 0)

                    if enforce_zero:
                        ## Use ap table groups instead? - can't; no support for mixin columns.
                        if multiplot and not lock_axis:
                            ax1.set_ylim(np.nanmin(np.append(self.phot.data[filter_key]['flux'], 0.0)), np.nanmax(self.phot.data[filter_key]['flux'])*yaxis_lim_multiplier)
                        else:
                            ax1.set_ylim(np.nanmin(np.append(self.phot.phot['flux'], 0.0)), np.nanmax(self.phot.phot['flux'])*yaxis_lim_multiplier)
                    else:
                        if multiplot and not lock_axis:
                            ax1.set_ylim(np.nanmin(self.phot.data[filter_key]['flux']), np.nanmax(self.phot.data[filter_key]['flux'])*yaxis_lim_multiplier)
                        else:
                            ax1.set_ylim(np.nanmin(self.phot.phot['flux']), np.nanmax(self.phot.phot['flux'])*yaxis_lim_multiplier)

                    if multiplot:
                        if not xextent:
                            ax1.set_xlim(np.nanmin(self.phot.phot["MJD"])-10, np.nanmax(self.phot.phot["MJD"]))
                        if xextent:
                            ax1.set_xlim(np.nanmin(self.phot.phot["MJD"])-10,np.nanmin(self.phot.phot["MJD"]) + xextent)
                    else:
                        pass

                    if xlims:
                        ax1.set_xlim(xlims)
                    if verbose:
                        print("xrange = ", ax1.get_xlim())
                        print("yrange = ", ax1.get_ylim())

                else:
                    if verbose: print("Filter '" + filter_key + "' not found")
                    warnings.warn("Filter '" + filter_key + "' not found")



            if not multiplot:

                ax1.set_ylabel(yaxis_label_string)

                if legend and not insidelegend:

                    plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                          numpoints = 1, frameon = False, fontsize = 12)
            else:
                fig.text(0.0, 0.5, yaxis_label_string, va = 'center', ha = 'left', rotation = 'vertical')


            if savepdf and outpath:
                fig.savefig(outpath + ".pdf", format = 'pdf', dpi=500)
            if savepng and outpath:
                fig.savefig(outpath + ".png", format = 'png', dpi=500)
            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def plot_spec(self, xminorticks = 250, legend = True,
                  wmin = 3500,
                  savepng = False, savepdf = False, outpath = False,
                  verbose = False, add_mjd = True,
                  *args, **kwargs):
        """
        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self, "spec"):

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 10])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            cmap_indices = np.linspace(0,1, len(self.spec))

            j = 0
            for i, spec_key in enumerate(self.spec):
                # if verbose: print(self.spec[spec_key].data.__dict__)

                plot_label_string = r'$\rm{' + self.spec[spec_key].data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'


                v_eff = 5436.87 ##Angstrom
                w = np.logical_and(self.spec[spec_key].data['wavelength'] > (v_eff-100.),self.spec[spec_key].data['wavelength'] < v_eff+100.)

                if verbose: print(i, len(w[np.where(w == True)]), spec_key, len(self.spec[spec_key].data['wavelength']), len(self.spec[spec_key].data['flux']), len(self.spec[spec_key].flux))
                if len(w[np.where(w == True)]) > 0:
                    if verbose: print(len(w), 'Foo')
                    flux_norm = self.spec[spec_key].flux / np.nanmean(self.spec[spec_key].flux[w])

                    ax1.plot(self.spec[spec_key].data['wavelength'], flux_norm - 0.5*j, lw = 2,
                                 label = plot_label_string, color = spec_colourmap(cmap_indices[i]),
                                 *args, **kwargs)

                    maxspecxdata = np.nanmax(self.spec[spec_key].data['wavelength'])
                    minspecxdata = np.nanmin(self.spec[spec_key].data['wavelength'])

                    w = np.where(self.spec[spec_key].data['wavelength'] >=  maxspecxdata - 200)
                    yatmaxspecxdata = np.nanmean((flux_norm - 0.5*j)[w])
                    w = np.where(self.spec[spec_key].data['wavelength'] <=  minspecxdata + 200)
                    yatminspecxdata = np.nanmean((flux_norm - 0.5*j)[w])
                    if verbose: print(yatminspecxdata)
                    # if i == 0:
                    if j == 0:
                        maxplotydata = np.nanmax(flux_norm - 0.5*j)
                        # minplotydata = np.nanmin(flux_norm - 0.5*j)
                        minplotydata = 0. - 0.5*j ## Assumes always positive flux


                        maxplotxdata = maxspecxdata
                        minplotxdata = np.nanmin(self.spec[spec_key].data['wavelength'])
                    else:
                        maxplotydata = np.nanmax(np.append(maxplotydata, np.append(yatminspecxdata, yatminspecxdata)))
                        # minplotydata = np.nanmin(np.append(minplotydata, flux_norm - 0.5*j))
                        minplotydata = 0. - 0.5*j ## Assumes always positive flux
                        maxplotxdata = np.nanmax(np.append(maxplotxdata, np.nanmax(self.spec[spec_key].data['wavelength'])))
                        minplotxdata = np.nanmin(np.append(minplotxdata, np.nanmin(self.spec[spec_key].data['wavelength'])))
                    if add_mjd:
                        # ax1.plot([maxspecxdata, 11000],[1 - 0.5*j, 1 - 0.5*j], ls = '--', color = hex['batman'])
                        # ax1.plot([maxspecxdata, 11000],[yatmaxspecxdata, yatmaxspecxdata], ls = '--', color = hex['batman'])
                        ax1.plot([2000, minspecxdata],[1 - 0.5*j, yatminspecxdata], ls = '--', color = hex['batman'])
                        # txt = ax1.text(1500, yatminspecxdata, r'$' + str(self.spec[spec_key].mjd_obs) + '$',
                        #                horizontalalignment = 'right', verticalalignment = 'center')
                        txt = ax1.text(2000, 1 - 0.5*j, r'$' + str(self.spec[spec_key].mjd_obs) + '$',
                                       horizontalalignment = 'right', verticalalignment = 'center')
                        # ax1.text(1000, 1 - 0.5*j, r'$' + str(self.spec[spec_key].mjd_obs) + '$', horizontalalignment = 'right')
                    j = j + 1
                else:
                    if verbose: print("Not enough data to normalise")
            if legend:

                plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            ax1.set_ylim(minplotydata - 0.5, maxplotydata + 0.5)
            ax1.set_xlim(1250, maxplotxdata*1.02)

            if verbose: print(minplotydata, maxplotydata)
            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            yaxis_label_string = r'$\textnormal{Flux, Arbitrary}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.set_yticklabels('')

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            if savepdf and outpath:
                fig.savefig(outpath + ".pdf", format = 'pdf', dpi=500)
            if savepng and outpath:
                fig.savefig(outpath + ".png", format = 'png', dpi=500)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def plot_mangledspec(self, xminorticks = 250, legend = True,
                  wmin = 3500,
                  savepng = False, savepdf = False, outpath = False,
                  verbose = False, add_mjd = True,
                  *args, **kwargs):
        """
        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self, "mangledspec"):

            setup_plot_defaults()

            fig = plt.figure(figsize=[8, 10])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            cmap_indices = np.linspace(0,1, len(self.spec))

            j = 0
            for i, spec_key in enumerate(self.mangledspec):
                # if verbose: print(self.spec[spec_key].data.__dict__)

                plot_label_string = r'$\rm{' + self.mangledspec[spec_key].data.meta["filename"].split('/')[-1].replace('_', '\_') + '}$'


                v_eff = 5436.87 ##Angstrom
                w = np.logical_and(self.mangledspec[spec_key].data['wavelength'] > (v_eff-100.),self.mangledspec[spec_key].data['wavelength'] < v_eff+100.)

                if verbose: print(i, len(w[np.where(w == True)]), spec_key, len(self.mangledspec[spec_key].data['wavelength']), len(self.mangledspec[spec_key].data['flux']), len(self.mangledspec[spec_key].flux))
                if len(w[np.where(w == True)]) > 0:

                    if verbose: print(len(w), 'Foo')

                    flux_norm = self.mangledspec[spec_key].flux / np.nanmean(self.mangledspec[spec_key].flux[w])

                    ax1.plot(self.mangledspec[spec_key].data['wavelength'], flux_norm - 0.5*j, lw = 2,
                                 label = plot_label_string, color = spec_colourmap(cmap_indices[i]),
                                 *args, **kwargs)

                    maxspecxdata = np.nanmax(self.mangledspec[spec_key].data['wavelength'])
                    minspecxdata = np.nanmin(self.mangledspec[spec_key].data['wavelength'])

                    w = np.where(self.mangledspec[spec_key].data['wavelength'] >=  maxspecxdata - 200)
                    yatmaxspecxdata = np.nanmean((flux_norm - 0.5*j)[w])
                    w = np.where(self.mangledspec[spec_key].data['wavelength'] <=  minspecxdata + 200)
                    yatminspecxdata = np.nanmean((flux_norm - 0.5*j)[w])
                    if verbose: print(yatminspecxdata)
                    # if i == 0:
                    if j == 0:
                        maxplotydata = np.nanmax(flux_norm - 0.5*j)
                        # minplotydata = np.nanmin(flux_norm - 0.5*j)
                        minplotydata = 0. - 0.5*j ## Assumes always positive flux


                        maxplotxdata = maxspecxdata
                        minplotxdata = np.nanmin(self.mangledspec[spec_key].data['wavelength'])
                    else:
                        maxplotydata = np.nanmax(np.append(maxplotydata, np.append(yatminspecxdata, yatminspecxdata)))
                        # minplotydata = np.nanmin(np.append(minplotydata, flux_norm - 0.5*j))
                        minplotydata = 0. - 0.5*j ## Assumes always positive flux
                        maxplotxdata = np.nanmax(np.append(maxplotxdata, np.nanmax(self.mangledspec[spec_key].data['wavelength'])))
                        minplotxdata = np.nanmin(np.append(minplotxdata, np.nanmin(self.mangledspec[spec_key].data['wavelength'])))
                    if add_mjd:
                        # ax1.plot([maxspecxdata, 11000],[1 - 0.5*j, 1 - 0.5*j], ls = '--', color = hex['batman'])
                        # ax1.plot([maxspecxdata, 11000],[yatmaxspecxdata, yatmaxspecxdata], ls = '--', color = hex['batman'])
                        ax1.plot([2000, minspecxdata],[1 - 0.5*j, yatminspecxdata], ls = '--', color = hex['batman'])
                        # txt = ax1.text(1500, yatminspecxdata, r'$' + str(self.mangledspec[spec_key].mjd_obs) + '$',
                        #                horizontalalignment = 'right', verticalalignment = 'center')
                        txt = ax1.text(2000, 1 - 0.5*j, r'$' + str(self.mangledspec[spec_key].mjd_obs) + '$',
                                       horizontalalignment = 'right', verticalalignment = 'center')
                        # ax1.text(1000, 1 - 0.5*j, r'$' + str(self.mangledspec[spec_key].mjd_obs) + '$', horizontalalignment = 'right')
                    j = j + 1
                else:
                    if verbose: print("Not enough data to normalise")
            if legend:

                plot_legend = ax1.legend(loc = [1.,0.0], scatterpoints = 1,
                                      numpoints = 1, frameon = False, fontsize = 12)

            ax1.set_ylim(minplotydata - 0.5, maxplotydata + 0.5)
            ax1.set_xlim(1250, maxplotxdata*1.02)

            if verbose: print(minplotydata, maxplotydata)
            ## Label the axes
            xaxis_label_string = r'$\textnormal{Wavelength (\AA)}$'
            yaxis_label_string = r'$\textnormal{Flux, Arbitrary}$'

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.set_yticklabels('')

            xminorLocator = MultipleLocator(xminorticks)
            ax1.xaxis.set_minor_locator(xminorLocator)

            if savepdf and outpath:
                fig.savefig(outpath + ".pdf", format = 'pdf', dpi=500)
            if savepng and outpath:
                fig.savefig(outpath + ".png", format = 'png', dpi=500)

            plt.show()
        else:
            warnings.warn("Doesn't seem to be any data here (empty self.data)")
        pass


    def plot_filters(self, filters = False, xminorticks = 250, yminorticks = 0.1,
                     show_lims = False,
                     *args, **kwargs):
        """
        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self.phot, 'data_filters'):
            if not filters:
                filters = self.phot.data_filters

            setup_plot_defaults()
            xaxis_label_string = r'$\textnormal{Wavelength, Angstrom }(\AA)$'
            yaxis_label_string = r'$\textnormal{Fractional Throughput}$'

            yminorLocator = MultipleLocator(yminorticks)
            xminorLocator = MultipleLocator(xminorticks)

            fig = plt.figure(figsize=[8, 4])
            fig.subplots_adjust(left = 0.09, bottom = 0.13, top = 0.99,
                                right = 0.99, hspace=0, wspace = 0)

            ax1 = fig.add_subplot(111)

            for i, filter_key in enumerate(filters):
                ## Check if there is something in the class to plot
                if hasattr(self.phot.data_filters[filter_key], "wavelength") and hasattr(self.phot.data_filters[filter_key], "throughput"):

                    plot_label_string = r'$\textnormal{' + self.phot.data_filters[filter_key].filter_name + '}$'


                    if hasattr(self.phot.data_filters[filter_key], "_plot_colour"):
                        ax1.plot(self.phot.data_filters[filter_key].wavelength, self.phot.data_filters[filter_key].throughput, color = self.phot.data_filters[filter_key]._plot_colour,
                                 lw = 2, label = plot_label_string)
                    else:
                        ax1.plot(self.phot.data_filters[filter_key].wavelength, self.phot.data_filters[filter_key].throughput, lw = 2, label = plot_label_string)

                    if show_lims:
                        try:
                            ax1.plot([self.phot.data_filters[filter_key]._upper_edge, self.phot.data_filters[filter_key]._upper_edge], [0,1] ,
                                     lw = 1.5, alpha = 0.5, ls = ':',
                                     color = self.phot.data_filters[filter_key]._plot_colour, zorder = 0, )
                            ax1.plot([self.phot.data_filters[filter_key]._lower_edge, self.phot.data_filters[filter_key]._lower_edge], [0,1] ,
                                     lw = 1.5, alpha = 0.5, ls = ':',
                                     color = self.phot.data_filters[filter_key]._plot_colour, zorder = 0, )
                        except:
                            print("Failed")
                else:
                    warning.warn("Doesn't look like you have loaded a filter into the object")

            default_xlims = ax1.get_xlim()
            ax1.plot(default_xlims, [0,0], color = hex["black"], ls = ":")
            ax1.set_xlim(default_xlims)

            ax1.set_xlabel(xaxis_label_string)
            ax1.set_ylabel(yaxis_label_string)

            ax1.yaxis.set_minor_locator(yminorLocator)
            ax1.xaxis.set_minor_locator(xminorLocator)

            ax1.legend(loc = 0)

            plt.show()
        pass


    def get_lcfit(self, path):
        """
        Parameters
        ----------

        Returns
        -------
        """
        StringWarning(path)
        self.lcfit = LCfitClass()
        self.lcfit.load_formatted_phot(path)
        self.lcfit.unpack()
        self.lcfit._sort_phot()
        self.lcfit.get_fit_splines()
        pass


    def get_specfit(self, verbose = False):
        """
        Parameters
        ----------

        Returns
        -------
        """

        self.specfit = OrderedDict()

        if hasattr(self, "name"):
            specfit_list = find_recon_spec(self.recon_directory, self.name, verbose = verbose)
            # if verbose: print(specfit_list)

            for i, specfit_file in enumerate(specfit_list):
                if verbose: print(i, specfit_file)
                self.specfit[specfit_file] = specfitClass()
                self.specfit[specfit_file].load(filename = specfit_file,
                            directory = self.recon_directory, verbose = verbose)
                self.specfit[specfit_file].set_orig_specpath()

        else:
            warnings.warn("This SNClass object has no name")
            if verbose: print("This SNClass object has no name")

        pass


    def get_simplespecphot(self, verbose = False):
        """
        When the SNClass has both lcfits and spec, sample the lcfits at the
        obsdate of the relevant (i.e. overlapping) spectra. Initially to
        recreate Fig 2 of Paper.

        Parameters
        ----------

        Returns
        -------
        """

        if hasattr(self, 'lcfit') and hasattr(self, 'spec'):
            # if verbose: print("Foo")

            # try:
            #     # self.simplespecphot = LCfitClass()
            #     self.simplespecphot = PhotometryClass()
            #
            #     lenstring = np.nanmax([len(i) for i in self.lcfit.data_filters.keys()]) ## object dtype is slow
            #     self.simplespecphot.phot = Table(names = ('MJD', 'flux', 'flux_err', 'filter'),
            #                                      dtype = [float, float, float, '|S'+str(lenstring)])
            #
            #     for i, spectrum in enumerate(self.spec):
            #
            #         for filter_name in self.spec[spectrum]._overlapping_filter_list:
            #             if verbose: print(i, spectrum, filter_name)
            #
            #             mjd = self.spec[spectrum].mjd_obs
            #             flux = self.lcfit.spline[filter_name](mjd)
            #             flux_err = self.lcfit.spline[filter_name + "_err"](mjd)
            #             newrow = {'MJD': mjd, 'flux': flux, 'flux_err': flux_err, 'filter':filter_name}
            #             self.simplespecphot.phot.add_row([mjd, flux, flux_err, filter_name])
            #
            #     self.simplespecphot.unpack()
            # except:
            #     warnings.warn("simplespecphot failed")


            # self.simplespecphot = LCfitClass()
            self.simplespecphot = PhotometryClass()

            lenstring = np.nanmax([len(i) for i in self.lcfit.data_filters.keys()])  ## object dtype is slow
            # self.simplespecphot.phot = Table(names=('MJD', 'flux', 'flux_err', 'filter'),
            #                                  dtype=[float, float, float, '|S' + str(lenstring)])

            mjd_list = []
            flux_list = []
            flux_err_list = []
            filter_list = []

            for i, spectrum in enumerate(self.spec):

                for filter_name in self.spec[spectrum]._overlapping_filter_list:
                    if verbose: print(i, spectrum, filter_name, type(filter_name))

                    mjd = self.spec[spectrum].mjd_obs
                    flux = self.lcfit.spline[filter_name](mjd)
                    flux_err = self.lcfit.spline[filter_name + "_err"](mjd)
                    # newrow = {'MJD': mjd, 'flux': flux, 'flux_err': flux_err, 'filter': filter_name}
                    # if i == 0:
                    #     self.simplespecphot.phot = Table(newrow)
                    # else:
                    #     self.simplespecphot.phot.add_row([mjd, flux, flux_err, filter_name])

                    mjd_list.append(mjd)
                    flux_list.append(flux)
                    flux_err_list.append(flux_err)
                    filter_list.append(filter_name)

                self.simplespecphot.phot = Table((mjd_list, flux_list, flux_err_list, filter_list), names=('MJD', 'flux', 'flux_err', 'filter'))

            self.simplespecphot.unpack(verbose=verbose)

        pass


    def check_overlaps(self, verbose = False):
        """
        Checks the filters that the spectrum overlaps with.
        originally used filter_within_spec

        Parameters
        ----------

        Returns
        -------
        """
        if hasattr(self.phot, "data") and hasattr(self, 'spec'):
            for i, spectrum in enumerate(self.spec):
                if verbose:print(i, spectrum)
                for j, filtername in enumerate(self.phot.data_filters):
                    if verbose:print(j, filtername)

                    if hasattr(self.phot.data_filters[filtername], "_lower_edge") and \
                      hasattr(self.phot.data_filters[filtername], "_upper_edge") and \
                      hasattr(self.spec[spectrum], "data"):
                       blue_bool = self.phot.data_filters[filtername]._lower_edge > self.spec[spectrum].min_wavelength
                       red_bool = self.phot.data_filters[filtername]._upper_edge < self.spec[spectrum].max_wavelength

                       if blue_bool and red_bool:
                            within = True
                       else:
                            within = False

                    if verbose:print(within)
                    if within:
                        self.spec[spectrum]._add_to_overlapping_filters(filtername)
        else:
            warnings.warn("SNClass.check_overlaps - something went wrong... no data?")
        pass


class InfoClass():
    """

    """

    def __init__(self):
        pass

    def load(self, path = False):
        if not path:
            path = _default_info_path

        self._data = Table.read(path, format = "ascii.commented_header")

        self.table = self._data

        self.table.meta["success"] = True
        self.snname = self.table["snname"]
        self.z_obs = self.table["z_obs"]
        self.distmod = self.table["mu"]
        self.distance = Distance(distmod = self.table["mu"])
        self.table["z_distmod"] = [i.z for i in self.distance]

        self.RA = self.table["RA"]
        self.Dec = self.table["Dec"]

        self.table["SkyCoords"] = SkyCoord(self.table["RA"], self.table["Dec"], unit=(u.hourangle, u.deg))
        self.coords = self.table["SkyCoords"]

        self.type = self.table["Type"]

    def get_sn_info(self, snname):
        try:
            w = np.where(self.snname == snname)
        except:
            print("foo")

        return self.table[w]



#  #----------------------------------------------------------------------------#  #
#  #  /CODE                                                                     #  #
#  #----------------------------------------------------------------------------#  #

## FUNCTIONS THAT ITS A PAIN TO SHIFT

def find_specphase_spec(snname, dir_path = _default_specphase_dir_path, file_type = ".spec", verbose = False):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if verbose: print(dir_path)
    StringWarning(dir_path)
    StringWarning(snname)
    if type(snname) is not str and type(snname) is not np.string_:
        raise(PathError)

    if not check_dir_path(dir_path):
        print("check_dir_path failed")
        return False

    try:
        ls = np.array(os.listdir(dir_path))

        # wspec = np.where(np.char.find(ls, file_type, start = -len(file_type)) > -1)
        # spec_list = ls[wspec]
        spec_list = [i for i in ls if i[-5:] == ".spec"]
        ## The last 18 chars are for the MJD and file_type
        # wsn = np.where([i[:-18] == snname for i in spec_list])
        # snmatch_list = spec_list[wsn]
        snmatch_list = [i for i in spec_list if i[:len(snname)] == snname ]

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

    phot_list = [os.path.join(path, match.group(0)) for file_name in ls for match in [regex.search(file_name)] if match]

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
