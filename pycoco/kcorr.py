"""

author: Rob Firth, Southampton
date: 01-2017

'offset' Based on Blanton et al. 2007: 2007AJ....133..734B
The UBRVI filters are those of Bessell (1990). The ugriz filters are those
determined by M. Doi, D. Eisenstein, and J. Gunn and are available on the
SDSS DR4 Web site ( http://www.sdss.org/dr4/ ). The JHKs filters are those from
Cohen et al. (2003).
"""

from __future__ import print_function

import os
import sys
import copy

from numpy import log10, linspace, ones, array_equal, zeros, append, array
from scipy.integrate import simps
from astropy import units as u
from astropy.table import Table
from astropy.constants import c as c
from lmfit import minimize, Parameters, fit_report
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from .classes import *
from .functions import *
from .defaults import *
from .utils import check_file_path, setup_plot_defaults, check_dir_path

__all__ = ["offset",
            # "convert_AB_to_Vega",
            # "convert_Vega_to_AB",
            "calc_AB_zp",
            "calc_vega_zp",
            "load_dark_sky_spectrum",
            "calc_spectrum_filter_flux",
            "load_atmosphere",
            "save_mangle",
            "applymangle",
            "calculate_fluxes",
            "manglemin",
            "plot_mangledata",
            "manglemin",
           ]

## offset is calculated as m_AB - m_vega
offset = {
    "U" : 0.79,
    "B" : 0.09,
    "V" : 0.02,
    "R" : 0.21,
    "I" : 0.45,
    "u" : 0.91,
    "g" : 0.08,
    "r" : 0.16,
    "i" : 0.37,
    "z" : 0.54,
    "J" : 0.91,
    "H" : 1.39,
    "Ks" : 1.85 ,
    "0:1u" : 1.25 ,
    "0:1g" : 0.01 ,
    "0:1r" : 0.04 ,
    "0:1i" : 0.27 ,
    "0:1z" : 0.46,
}

# def convert_Vega_to_AB(phot_table, filters = False):
#     """
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     return phot_table
#

# def convert_AB_to_Vega():
#     """
#     Parameters
#     ----------
#     Returns
#     -------
#     """
#
#     return phot_table


def load_vega(path = os.path.join(_default_kcorr_data_path, "alpha_lyr_stis_002.dat"), wmin = 1500*u.angstrom, *args, **kwargs):
    """
    returns spectrum of Vega as a SpectrumClass instance
    """
    vega = SpectrumClass()
    vega.load(path, wmin = wmin, *args, **kwargs)

    return vega


def load_AB(path = os.path.join(_default_kcorr_data_path, "AB_pseudospectrum.dat"), wmin = 1500*u.angstrom, *args, **kwargs):
    """
    returns 'spectrum' as a SpectrumClass instance
    """
    AB = SpectrumClass()
    AB.load(path, wmin = wmin, *args, **kwargs)

    return AB


def generate_AB_pseudospectrum(fnu = False):
    """

    """

    f_nu_AB = 3.63078e-20 ## erg s^-1 cm^-2 Hz^-1
    freq = linspace(2e13, 2e15, num = 1000)[::-1]*u.Hz ## Hz
    f_nu = ones(len(freq))*f_nu_AB


    if fnu:
        table = Table([freq, f_nu], names = ("frequency", "flux"))
    else:
        wavelength = (c/freq).to("Angstrom") ## \AA
        f = freq*f_nu_AB ## erg s^-1 cm^-2
        f_lambda = f/wavelength

        table = Table([wavelength, f_lambda], names = ("wavelength", "flux"))

    return table


def load_atmosphere(path = os.path.join(_default_lsst_throughputs_path, "baseline/atmos_std.dat")):
    """
    reads in atmosphere from LSST_THROUGHPUTS, default is at airmass 1.2
    """

    atmos = BaseFilterClass()
    atmos.load(path, wavelength_u = u.nm, fmt = "ascii.commented_header")

    return atmos


def calc_filter_area(filter_name = False, filter_object=False, filter_path = _default_filter_dir_path):
    """

    :param filter_name:
    :param filter_object:
    :param filter_path:
    :return:
    """

    if filter_object:
        if hasattr(filter_object, "_effective_area"):
            return filter_object._effective_area
        else:
            filter_object.calculate_filter_area()
            return filter_object._effective_area
    else:
        check_file_path(os.path.join(filter_path, filter_name + ".dat"))

        filter_object = load_filter(os.path.join(filter_path, filter_name + ".dat"))
        filter_object.calculate_effective_wavelength()
        filter_area = simps(filter_object.throughput, filter_object.wavelength)

    return filter_area


def calc_spectrum_filter_flux(filter_name=False, filter_object=False, spectrum_object=False,
                              filter_path = _default_filter_dir_path, spectrum_dir=None, spectrum_filename=None,
                              correct_for_area=True):
    """
    returns flux in units of

    :param filter_name:
    :param spectrum_object:
    :param filter_path:
    :return:
    """
    if not filter_object:
        check_file_path(os.path.join(filter_path, filter_name + ".dat"))

        filter_object = load_filter(os.path.join(filter_path, filter_name + ".dat"))
        if not hasattr(filter_object, "lambda_effective"):
            filter_object.calculate_effective_wavelength()

    if not spectrum_object:
        spectrum_path=os.path.join(spectrum_dir, spectrum_filename)

        check_file_path(spectrum_path)

        spectrum_object = SpectrumClass()
        spectrum_object.load(filename=spectrum_filename, path=spectrum_dir)

    if not array_equal(filter_object.wavelength, spectrum_object.wavelength):
        filter_object.resample_response(new_wavelength = spectrum_object.wavelength)

    if hasattr(filter_object, "_effective_area"):
        filter_object.calculate_filter_area()
        filter_area = filter_object._effective_area
        # filter_area = simps(filter_object.throughput, filter_object.wavelength)

    transmitted_spec = filter_object.throughput * spectrum_object.flux
    integrated_flux = simps(transmitted_spec, spectrum_object.wavelength)

    if correct_for_area:
        return  integrated_flux/filter_area
    else:
        return  integrated_flux


def calc_AB_flux(filter_name, filter_path = _default_filter_dir_path, filter_object = False):

    AB = load_AB()

    if not filter_object:
        filter_object = load_filter(os.path.join(filter_path, filter_name + ".dat"))

    filter_object.calculate_effective_wavelength()
    filter_object.resample_response(new_wavelength = AB.wavelength)

    transmitted_spec = filter_object.throughput * AB.flux

    integrated_flux = simps(transmitted_spec, AB.wavelength)

    return integrated_flux


def calc_AB_zp(filter_name=False, filter_object = False):
    """

    """
    if not filter_object and filter_name:
        filter_object = load_filter(os.path.join(_default_filter_dir_path, filter_name + ".dat"))

    integrated_flux = calc_AB_flux(filter_name, filter_object=filter_object)
    area_corr_integrated_flux = integrated_flux / calc_filter_area(filter_name)

    return -2.5 * log10(area_corr_integrated_flux)


def calc_vega_flux(filter_name, filter_object = False,):
    """

    """

    vega = load_vega()

    if not filter_object:
        filter_object = load_filter(os.path.join(_default_filter_dir_path, filter_name + ".dat"))
    # else if hasattr(filter_object, "wavelength"):

    filter_object.resample_response(new_wavelength = vega.wavelength)

    transmitted_spec = filter_object.throughput * vega.flux

    integrated_flux = simps(transmitted_spec, vega.wavelength)

    return integrated_flux


def calc_vega_zp(filter_name, filter_object = False, vega_Vmag = 0.03):
    """

    """

    if not filter_object:
        filter_object = load_filter(os.path.join(_default_filter_dir_path, filter_name + ".dat"))

    integrated_flux = calc_vega_flux(filter_name)
    area_corr_integrated_flux = integrated_flux / calc_filter_area(filter_name)

    # return -2.5 * log10(integrated_V_flux) - vega_Vmag
    return -2.5 * log10(area_corr_integrated_flux)


def calc_vega_mag(filter_name):
    """

    """
    zp = calc_vega_zp(filter_name)
    flux = calc_vega_flux(filter_name)

    mag = -2.5 * log10(flux) - zp
    return mag


def load_dark_sky_spectrum(wmin = 1500*u.angstrom, wmax = 11000*u.angstrom, *args, **kwargs):
    """
    requires https://github.com/lsst/throughputs/ and environment vars LSST_THROUGHPUTS
    and LSST_THROUGHPUTS_BASELINE.

    e.g.:
    setenv LSST_THROUGHPUTS ${HOME}/projects/LSST/throughputs
    setenv LSST_THROUGHPUTS_BASELINE ${LSST_THROUGHPUTS}/baseline
    """
    dark_sky_path = os.path.join(os.environ["LSST_THROUGHPUTS_BASELINE"],"darksky.dat")
    darksky = SpectrumClass()
    darksky.load(dark_sky_path, wavelength_u = u.nm, flux_u = u.cgs.erg / u.si.cm ** 2 / u.si.s / u.nm,
                 fmt = "ascii.commented_header", wmin = wmin, wmax = wmax, *args, **kwargs)

    darksky.success = True

    return darksky


def calc_m_darksky(filter_name=False, filter_object = False, dark_sky = False, vega = False):
    """

    :param filter_name:
    :param filter_object:
    :param dark_sky:
    :param vega:
    :return:
    """
    if not dark_sky:
        dark_sky_path = os.path.join(os.environ["LSST_THROUGHPUTS_BASELINE"], "darksky.dat")
        darksky = SpectrumClass()
        darksky.load(dark_sky_path, wavelength_u=u.nm, flux_u=u.cgs.erg / u.si.cm ** 2 / u.si.s / u.nm,
                     fmt="ascii.commented_header", wmin=3500 * u.angstrom, wmax=11000 * u.angstrom, )

    if filter_object:
        if hasattr(filter_object):
            zp = filter_object.zp_AB
        else:
            filter_object.get_zeropoint()
            zp = filter_object.zp_AB
    else:
        if vega:
            zp = calc_vega_zp(filter_name)
        else:
            zp = calc_AB_zp(filter_name)

    return -2.5 * log10(calc_spectrum_filter_flux(filter_name, darksky)) - zp


def nu_to_lambda(freq):
    """

    :param freq:
    :return:
    """
    wavelength = (c/freq).to("Angstrom")
    return wavelength


def lambda_to_nu(wavelength):
    """

    :param wavelength:
    :return:
    """
    freq = (c/wavelength).to("Hz")
    return freq


## Mangling

def manglespec3(SpectrumObject):
    """

    :param SpectrumObject:
    :return:
    """

    pass


def save_mangle(mS, filename, orig_filename, path=False,
                squash=False, verbose=True, *args, **kwargs):
    """

    :param mS:
    :param filename:
    :param orig_filename:
    :param path:
    :param squash:
    :param verbose:
    :param args:
    :param kwargs:
    :return:
    """

    if hasattr(mS, "data"):
        if verbose: print("has data")
        if not path:
            if verbose: print("No directory specified, assuming " + mS._default_data_dir_path)
            path = mS._default_data_dir_path
        else:
            StringWarning(path)

        outpath = os.path.join(path, filename)

        pcc.check_dir_path(path)

        save_table = Table()

        save_table['wavelength'] = mS.wavelength
        save_table['flux'] = mS.flux

        save_table['wavelength'].format = "5.5f"
        save_table['flux'].format = "5.5e"

        save_table.meta["comments"] = [orig_filename, ]

        if os.path.isfile(outpath):
            if squash:
                print("Overwriting " + outpath)
                save_table.write(outpath, format="ascii.no_header", overwrite=True)
            else:
                warnings.warn("Found existing file matching " + os.path.join(path,
                                                                             filename) + ". Run with squash = True to overwrite")
        else:
            print("Writing " + outpath)
            save_table.write(outpath, format="ascii.no_header")

    else:
        warnings.warn("Doesn't seem to be any data here (empty self.data)")
    pass


def applymangle(params, SpectrumObject):
    """

    :param params:
    :param SpectrumObject:
    :return:
    """

    MangledSpectrumObject = copy.deepcopy(SpectrumObject)
    paramlist = array([params[key].value for key in params.keys()])
    print("params:", paramlist)

    weights = append(append(1.0, paramlist), 1.0)
    print("weights:", weights)

    # SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights)
    SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights, bc_type = "clamped")

    plt.plot(MangledSpectrumObject.wavelength, SplObj(MangledSpectrumObject.wavelength))
    plt.scatter(data_table["lambda_eff"], weights)

    plt.show()

    MangledSpectrumObject.flux = MangledSpectrumObject.flux * SplObj(MangledSpectrumObject.wavelength)

    return MangledSpectrumObject


def calculate_fluxes(data_table, S, verbose=False):
    """

    :param data_table:
    :param S:
    :param verbose:
    :return:
    """
    for i, f in enumerate(data_table["filter_object"]):
        column = Column(zeros(len(data_table)), name=fit_flux)

        if isinstance(f, FilterClass):
            mangledspec_filterflux = calc_spectrum_filter_flux(filter_object=f, spectrum_object=S)
            print(data_table["spec_filterflux"][i], mangledspec_filterflux)
            # data_table["mangledspec_filterflux"][i] = mangledspec_filterflux
            column[i] = mangledspec_filterflux

        else:
            pass
    return column


def manglemin(params, SpectrumObject, data_table, verbose=False, *args, **kwargs):
    """
    """
    MangledSpectrumObject = copy.deepcopy(SpectrumObject)
    paramlist = array([params[key].value for key in params.keys()])

    weights = append(append(1.0, paramlist), 1.0)

    # SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights)
    SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights, bc_type = "clamped")

    MangledSpectrumObject.flux = MangledSpectrumObject.flux * SplObj(MangledSpectrumObject.wavelength)

    specflux = array([calc_spectrum_filter_flux(filter_object=FilterObject, spectrum_object=MangledSpectrumObject) for
         FilterObject in data_table[data_table["mask"]]["filter_object"]])
    if verbose:
        print("params:", paramlist)
        print("weights:", weights)
        print("flux:", specflux)
        print("fitflux:", data_table[data_table["mask"]]["fitflux"].data)

    return data_table[data_table["mask"]]["fitflux"] - specflux

