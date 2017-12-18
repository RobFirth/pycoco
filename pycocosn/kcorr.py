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

import copy
import warnings
import os
from collections import OrderedDict

from astropy import units as u
from astropy.modeling import blackbody as bb
from astropy.constants import c as c
from astropy.table import Table, Column, vstack
from lmfit import minimize, Parameters, fit_report
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import simps, trapz

# from .classes import *
# from .colours import *
# from .defaults import *
# from .functions import *
# from .utils import utils.check_file_path, utils.check_dir_path, b
from . import classes
from . import colours
from . import defaults
from . import functions
from . import utils
from . import extinction

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
            "manglespec3",
            "mangle"
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


def load_vega(path = os.path.join(defaults._default_kcorr_data_path, "alpha_lyr_stis_002.dat"), wmin = 1500*u.angstrom,
              *args, **kwargs):
    """
    returns spectrum of Vega as a SpectrumClass instance
    """
    vega = classes.SpectrumClass()
    vega.load(path, wmin = wmin, *args, **kwargs)

    return vega


def load_AB(path = os.path.join(defaults._default_kcorr_data_path, "AB_pseudospectrum.dat"), wmin = 1500*u.angstrom,
            *args, **kwargs):
    """
    returns 'spectrum' as a SpectrumClass instance
    """
    AB = classes.SpectrumClass()
    AB.load(path, wmin = wmin, *args, **kwargs)

    return AB


def generate_AB_pseudospectrum(fnu = False):
    """

    """

    f_nu_AB = 3.63078e-20 ## erg s^-1 cm^-2 Hz^-1
    freq = np.linspace(2e13, 2e15, num = 1000)[::-1]*u.Hz ## Hz
    f_nu = np.ones(len(freq))*f_nu_AB


    if fnu:
        table = Table([freq, f_nu], names = ("frequency", "flux"))
    else:
        wavelength = (c/freq).to("Angstrom") ## \AA
        f = freq*f_nu_AB ## erg s^-1 cm^-2
        f_lambda = f/wavelength

        table = Table([wavelength, f_lambda], names = ("wavelength", "flux"))

    return table


def load_atmosphere(path = os.path.join(defaults._default_lsst_throughputs_path, "baseline/atmos_std.dat")):
    """
    reads in atmosphere from LSST_THROUGHPUTS, default is at airmass 1.2
    """

    atmos = classes.BaseFilterClass()
    atmos.load(path, wavelength_u = u.nm, fmt = "ascii.commented_header")

    return atmos


def calc_filter_area(filter_name = False, filter_object=False, filter_path = defaults._default_filter_dir_path):
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
        utils.check_file_path(os.path.join(filter_path, filter_name + ".dat"))

        filter_object = functions.load_filter(os.path.join(filter_path, filter_name + ".dat"))
        filter_object.calculate_effective_wavelength()
        filter_area = simps(filter_object.throughput, filter_object.wavelength)

    return filter_area


def calc_spectrum_filter_flux(filter_name=False, filter_object=False, spectrum_object=False,
                              filter_path = defaults._default_filter_dir_path, spectrum_dir=None,
                              spectrum_filename=None, correct_for_area=True, verbose = True):
    """
    returns flux in units of

    :param filter_object:
    :param spectrum_dir:
    :param spectrum_filename:
    :param correct_for_area:
    :param filter_name:
    :param spectrum_object:
    :param filter_path:
    :return:
    """
    if not filter_object:
        utils.check_file_path(os.path.join(filter_path, filter_name + ".dat"))

        filter_object = functions.load_filter(os.path.join(filter_path, filter_name + ".dat"))
        if not hasattr(filter_object, "lambda_effective"):
            filter_object.calculate_effective_wavelength()

    if not spectrum_object:
        spectrum_path=os.path.join(spectrum_dir, spectrum_filename)

        utils.check_file_path(spectrum_path)

        spectrum_object = classes.SpectrumClass()
        spectrum_object.load(filename=spectrum_filename)

    if verbose:
        print("filter = ", filter_object.filter_name)

    if verbose:
        print("min wavelength = ", np.nanmin(filter_object.wavelength))
        print("max wavelength = ", np.nanmax(filter_object.wavelength))

    if not np.array_equal(filter_object.wavelength, spectrum_object.wavelength):
        if verbose: print("resampling the response")
        filter_object.resample_response(new_wavelength = spectrum_object.wavelength)

        if verbose:
            print("min wavelength = ", np.nanmin(filter_object.wavelength))
            print("max wavelength = ", np.nanmax(filter_object.wavelength))

    if hasattr(filter_object, "_effective_area"):
        filter_object.calculate_filter_area()
        filter_area = filter_object._effective_area
        # filter_area = simps(filter_object.throughput, filter_object.wavelength)

        if verbose: print("Filter_area = ", filter_area)

    transmitted_spec = filter_object.throughput * spectrum_object.flux
    integrated_flux = simps(transmitted_spec, spectrum_object.wavelength)
    if verbose: print("Integrated flux = ", integrated_flux)

    if np.isnan(integrated_flux):   ## See Issue #26 on GitHub
        integrated_flux = trapz(transmitted_spec, spectrum_object.wavelength)
        if verbose: print("New integrated flux = ",integrated_flux)

    if correct_for_area:
        return  integrated_flux/filter_area
    else:
        return  integrated_flux


def calc_AB_flux(filter_name, filter_path = defaults._default_filter_dir_path,
                 abpath = os.path.join(defaults._default_kcorr_data_path, "AB_pseudospectrum.dat"),
                 filter_object = False):

    AB = load_AB(path=abpath)

    if not filter_object:
        filter_object = functions.load_filter(os.path.join(filter_path, filter_name + ".dat"))

    filter_object.calculate_effective_wavelength()
    filter_object.resample_response(new_wavelength = AB.wavelength)

    transmitted_spec = filter_object.throughput * AB.flux

    integrated_flux = simps(transmitted_spec, AB.wavelength)

    return integrated_flux


def calc_AB_zp(filter_name=False, filter_object = False,
               abpath=os.path.join(defaults._default_kcorr_data_path, "AB_pseudospectrum.dat")):
    """

    """
    if not filter_object and filter_name:
        filter_object = functions.load_filter(os.path.join(defaults._default_filter_dir_path, filter_name + ".dat"))

    integrated_flux = calc_AB_flux(filter_name, filter_object=filter_object, abpath=abpath)
    area_corr_integrated_flux = integrated_flux / calc_filter_area(filter_name)

    return -2.5 * np.log10(area_corr_integrated_flux)


def calc_vega_flux(filter_name, filter_object = False,):
    """

    """

    vega = load_vega()

    if not filter_object:
        filter_object = functions.load_filter(os.path.join(defaults._default_filter_dir_path, filter_name + ".dat"))
    # else if hasattr(filter_object, "wavelength"):

    filter_object.resample_response(new_wavelength = vega.wavelength)

    transmitted_spec = filter_object.throughput * vega.flux

    integrated_flux = simps(transmitted_spec, vega.wavelength)

    return integrated_flux


def calc_vega_zp(filter_name, filter_object = False, vega_Vmag = 0.03):
    """

    """

    if not filter_object:
        filter_object = functions.load_filter(os.path.join(defaults._default_filter_dir_path, filter_name + ".dat"))

    integrated_flux = calc_vega_flux(filter_name)
    area_corr_integrated_flux = integrated_flux / calc_filter_area(filter_name)

    # return -2.5 * np.log10(integrated_V_flux) - vega_Vmag
    return -2.5 * np.log10(area_corr_integrated_flux)


def calc_vega_mag(filter_name):
    """

    """
    zp = calc_vega_zp(filter_name)
    flux = calc_vega_flux(filter_name)

    mag = -2.5 * np.log10(flux) - zp
    return mag


def load_dark_sky_spectrum(wmin = 1500*u.angstrom, wmax = 11000*u.angstrom, *args, **kwargs):
    """
    requires https://github.com/lsst/throughputs/ and environment vars LSST_THROUGHPUTS
    and LSST_THROUGHPUTS_BASELINE.

    e.g.:
    setenv LSST_THROUGHPUTS ${HOME}/projects/LSST/throughputs
    setenv LSST_THROUGHPUTS_BASELINE ${LSST_THROUGHPUTS}/baseline
    """
    dark_sky_path = os.path.join(defaults._default_lsst_throughputs_path,"darksky.dat")
    darksky = classes.SpectrumClass()
    darksky.load(dark_sky_path, wavelength_u = u.nm, flux_u = u.cgs.erg / u.si.cm ** 2 / u.si.s / u.nm,
                 fmt = "ascii.commented_header", wmin = wmin, wmax = wmax, *args, **kwargs)

    darksky.success = True

    return darksky


def calc_m_darksky(filter_name=False, filter_object = False, dark_sky = False, vega = False, abspath=False,
                   verbose = False):
    """

    :param filter_name:
    :param filter_object:
    :param dark_sky:
    :param vega:
    :return:
    """
    if not dark_sky:
        dark_sky_path = os.path.join(os.environ["LSST_THROUGHPUTS_BASELINE"], "darksky.dat")
        darksky = classes.SpectrumClass()
        darksky.load(dark_sky_path, wavelength_u=u.nm, flux_u=u.cgs.erg / u.si.cm ** 2 / u.si.s / u.nm,
                     fmt="ascii.commented_header", wmin=3500 * u.angstrom, wmax=11000 * u.angstrom, abspath=abspath)

    if filter_object:
        if hasattr(filter_object, "zp_AB"):
            zp = filter_object.zp_AB
        else:
            filter_object.get_zeropoint()
            zp = filter_object.zp_AB
        return -2.5 * np.log10(calc_spectrum_filter_flux(filter_object=filter_object, spectrum_object=darksky, verbose = verbose)) - zp

    else:
        if vega:
            zp = calc_vega_zp(filter_name)
        else:
            zp = calc_AB_zp(filter_name)

        return -2.5 * np.log10(calc_spectrum_filter_flux(filter_name=filter_name, spectrum_object=darksky, verbose=verbose)) - zp


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

def mangle(sn, S, spec_mjd, filters, staticfilter=False, anchor_distance=100, verbose=False):
    """

    :param anchor_distance:
    :param spec_mjd:
    :param filters:
    :param staticfilter:
    :param verbose:
    :param sn:
    :param S:
    :return:
    """

    if hasattr(sn, "lcfit"):

        rows = OrderedDict()
        filter_dict = OrderedDict()

        for i, f in enumerate(filters):
            filter_dict[f] = functions.load_filter(os.path.join(defaults._default_filter_dir_path, f + ".dat"))
            filter_dict[f].calculate_edges()
            #     filter_dict[f].calculate_edges_zero()

            fit_flux = sn.lcfit.spline[f](spec_mjd)

            sn.phot.data_filters[f].resample_response(new_wavelength=S.wavelength)
            S_filter_flux = calc_spectrum_filter_flux(filter_object=sn.phot.data_filters[f], spectrum_object=S, verbose=verbose)
            S_filter_flux_no_area = calc_spectrum_filter_flux(filter_object=sn.phot.data_filters[f],
                                                                  spectrum_object=S,
                                                                  correct_for_area=False, verbose=verbose)
            mS_filter_flux = np.nan

            rows[f] = (fit_flux, S_filter_flux, S_filter_flux_no_area)
            if i == 0:
                data_table = Table(
                    names=("filter", "fitflux", "spec_filterflux", "mangledspec_filterflux", "filter_object", "mask"),
                    dtype=('S12', 'f4', 'f4', 'f4', object, bool))
            data_table.add_row((f, fit_flux, S_filter_flux, mS_filter_flux, filter_dict[f], True))

        for i, f in enumerate(data_table["filter_object"]):
            ## Test extent
            bool_uncontained = np.logical_or(f._lower_edge < S.min_wavelength, f._upper_edge > S.max_wavelength)
            if verbose: print(bool_uncontained)
            if bool_uncontained:
                data_table = data_table[np.where(data_table["filter"] != utils.b(f.filter_name))]

        knot_colours = [j._plot_colour for j in data_table["filter_object"] if hasattr(j, "_plot_colour")]
        data_table.add_column(Column(knot_colours, name="knot_colours"))
        data_table["lambda_eff"] = [i.lambda_effective.value for i in data_table["filter_object"]]

        if not staticfilter:
            w = 0
        else:
            w = np.where(data_table["filter"] == staticfilter)


        scale_factor = 1. / data_table[w]["fitflux"]
        if verbose: print("Scale Factor", scale_factor)
        norm_factor = data_table[w]["fitflux"] / data_table[w]["spec_filterflux"]
        if verbose: print("norm factor", norm_factor)
        data_table["fitflux"] = data_table["fitflux"] * scale_factor
        # "spec flux"
        data_table["spec_filterflux"] = data_table["spec_filterflux"] * scale_factor
        if verbose: print("scaled ", )

        S.flux = S.flux * scale_factor
        S.flux = S.flux * norm_factor
        S.scale_factor = scale_factor
        S.norm_factor = norm_factor

        data_table["spec_filterflux"] = data_table["spec_filterflux"] * norm_factor
        # ## Lower
        anchor_min_wavelength = np.nanmin([i._lower_edge for i in data_table["filter_object"]]) - anchor_distance
        # ## Upper
        anchor_max_wavelength = np.nanmax([i._upper_edge for i in data_table["filter_object"]]) + anchor_distance

        if verbose: print(data_table)
        if len(data_table) < 2:
            S.flux = S.flux / S.scale_factor

            fit_dict = OrderedDict()
            fit_dict["SpectrumObject"] = S
            fit_dict["final_spl"] = lambda x: np.ones(len(x))
            fit_dict["data_table"] = data_table

            return fit_dict

        mc_l, mc_u = functions.calc_linear_terms(data_table[data_table["mask"]], key="fitflux", verbose=verbose)
        anchor_l = mc_l[0] * anchor_min_wavelength + mc_l[1]
        anchor_u = mc_u[0] * anchor_max_wavelength + mc_u[1]


        spl_wav = S.data['wavelength'][
            np.logical_and(S.data['wavelength'] >= anchor_min_wavelength, S.data['wavelength'] <= anchor_max_wavelength)]

        mc_spec_l, mc_spec_u = functions.calc_linear_terms(data_table[data_table["mask"]], key="spec_filterflux", verbose=verbose)
        anchor_spec_l = mc_spec_l[0] * anchor_min_wavelength + mc_spec_l[1]
        anchor_spec_u = mc_spec_u[0] * anchor_max_wavelength + mc_spec_u[1]

        data_table.add_row(("lower_anchor", anchor_spec_l, anchor_spec_l, anchor_spec_u, np.nan, False,
                            colours.hex["batman"], anchor_min_wavelength))
        data_table.add_row(("upper_anchor", anchor_spec_u, anchor_spec_u, anchor_spec_u, np.nan, False,
                            colours.hex["batman"], anchor_max_wavelength))


        data_table.add_index("lambda_eff")
        data_table.sort()

        for i, f in enumerate(data_table["filter_object"]):
            if isinstance(f, classes.FilterClass):
                mangledspec_filterflux = calc_spectrum_filter_flux(filter_object=f, spectrum_object=S, verbose=verbose)
                #         print(data_table["spec_filterflux"][i], mangledspec_filterflux)
                data_table["mangledspec_filterflux"][i] = mangledspec_filterflux
            else:
                pass


        wanted_flux = data_table[data_table["mask"]]["fitflux"].data
        wanted_filters = data_table[data_table["mask"]]["filter_object"].data

        fit_dict = manglespec3(S, spec_mjd, wanted_filters, wanted_flux, data_table)

    return fit_dict


def manglespec3(SpectrumObject, spec_mjd, wanted_filters, wanted_flux, data_table, verbose = False):
    """

    :param verbose:
    :param spec_mjd:
    :param wanted_filters:
    :param wanted_flux:
    :param data_table:
    :param SpectrumObject:

    :return:
    """
    original_spectrum_flux = data_table[data_table["mask"]]["spec_filterflux"].data
    scaled_spectrum_flux = data_table[data_table["mask"]]["mangledspec_filterflux"].data

    if len(scaled_spectrum_flux) == len(wanted_flux):
        params = Parameters()
        for i, flux_tuple in enumerate(zip(scaled_spectrum_flux, wanted_flux)):
            params.add(wanted_filters[i].filter_name, value=flux_tuple[1] / flux_tuple[0])
        else:
            pass

    paramlist = np.array([params[key].value for key in params.keys()])
    data_table["weights"] = Column(np.append(1, np.append(paramlist, 1)), name="weights")

    mc_l, mc_u = functions.calc_linear_terms(data_table[data_table["mask"]], key="weights", verbose=verbose)
    weight_l = mc_l[0] * data_table["lambda_eff"][0] + mc_l[1]
    weight_u = mc_u[0] * data_table["lambda_eff"][-1] + mc_u[1]

    weights = np.append(np.append(weight_l, paramlist), weight_u)
    data_table["weights"] = weights

    ## Do the fit
    out = minimize(manglemin, params, args=(SpectrumObject, data_table), kws=({"verbose": verbose}))
    # out = minimize(manglemin, params, args=(SpectrumObject, data_table), epsfcn=1e-5)
    if verbose: print(fit_report(out))

    paramlist = np.array([out.params[key].value for key in out.params.keys()])

    mc_l, mc_u = functions.calc_linear_terms(data_table, key="weights")
    data_table["weights"][0] = mc_l[0] * data_table["lambda_eff"][0] + mc_l[1]
    data_table["weights"][-1] = mc_u[0] * data_table["lambda_eff"][-1] + mc_u[1]
    weights = data_table["weights"].data
    final_spl = interpolate.CubicSpline(data_table["lambda_eff"], data_table["weights"], bc_type="clamped")

    SpectrumObject.flux = final_spl(SpectrumObject.wavelength) * SpectrumObject.flux / SpectrumObject.scale_factor

    data_table["fitflux"] = data_table["fitflux"] / SpectrumObject.scale_factor
    data_table["spec_filterflux"] = data_table["spec_filterflux"] / SpectrumObject.scale_factor

    # data_table[0]["mangledspec_filterflux"] = data_table[0]["mangledspec_filterflux"] / SpectrumObject.scale_factor
    # data_table[-1]["mangledspec_filterflux"] = data_table[-1]["mangledspec_filterflux"] / SpectrumObject.scale_factor

    # data_table["mangledspec_filterflux"] = data_table["mangledspec_filterflux"] / SpectrumObject.scale_factor
    data_table["mangledspec_filterflux"] = calculate_fluxes(data_table, SpectrumObject)
    fit_dict = OrderedDict()

    fit_dict["SpectrumObject"] = SpectrumObject
    fit_dict["final_spl"] = final_spl
    fit_dict["data_table"] = data_table

    return fit_dict


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
            if verbose: print("No directory specified, assuming " + defaults._default_recon_dir_path)
            path = defaults._default_recon_dir_path
        else:
            errors.StringWarning(path)

        outpath = os.path.join(path, filename)

        utils.check_dir_path(path)

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


def applymangle(params, SpectrumObject, verbose = False):
    """

    :param verbose:
    :param params:
    :param SpectrumObject:
    :return:
    """

    MangledSpectrumObject = copy.deepcopy(SpectrumObject)
    paramlist = np.array([params[key].value for key in params.keys()])
    if verbose: print("params:", paramlist)

    weights = np.append(np.append(1.0, paramlist), 1.0)
    if verbose: print("weights:", weights)

    # SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights)
    SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights, bc_type = "clamped")

    # plt.plot(MangledSpectrumObject.wavelength, SplObj(MangledSpectrumObject.wavelength))
    # plt.scatter(data_table["lambda_eff"], weights)
    # plt.show()

    MangledSpectrumObject.flux = MangledSpectrumObject.flux * SplObj(MangledSpectrumObject.wavelength)

    return MangledSpectrumObject


def calculate_fluxes(data_table, S, verbose=False):
    """

    :param data_table:
    :param S:
    :param verbose:
    :return:
    """
    column = Column(np.zeros(len(data_table)), name="fit_flux")

    for i, f in enumerate(data_table["filter_object"]):

        if isinstance(f, classes.FilterClass):
            mangledspec_filterflux = calc_spectrum_filter_flux(filter_object=f, spectrum_object=S, verbose=verbose)
            if verbose: print(data_table["spec_filterflux"][i], mangledspec_filterflux)
            # data_table["mangledspec_filterflux"][i] = mangledspec_filterflux
            column[i] = mangledspec_filterflux

        else:
            pass
    return column


def manglemin(params, SpectrumObject, data_table, verbose=False, clamped=False, *args, **kwargs):
    """
    """
    MangledSpectrumObject = copy.deepcopy(SpectrumObject)
    paramlist = np.array([params[key].value for key in params.keys()])

    # weights = np.append(np.append(1.0, paramlist), 1.0)
    mc_l, mc_u = functions.calc_linear_terms(data_table[data_table["mask"]], key="weights")
    data_table["weights"][0] = mc_l[0] * data_table["lambda_eff"][0] + mc_l[1]
    data_table["weights"][-1] = mc_u[0] * data_table["lambda_eff"][-1] + mc_u[1]
    weights = data_table["weights"].data
    #     print(weights)
    data_table["weights"][data_table["mask"]] = paramlist

    data_table["mangledspec_filterflux"][0] = data_table["spec_filterflux"][0] * data_table["weights"][0]
    data_table["mangledspec_filterflux"][-1] = data_table["spec_filterflux"][-1] * data_table["weights"][-1]

    if clamped:
        SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights, bc_type = "clamped")
    else:
        SplObj = interpolate.CubicSpline(data_table["lambda_eff"], weights)

    MangledSpectrumObject.flux = MangledSpectrumObject.flux * SplObj(MangledSpectrumObject.wavelength)

    specflux = np.array([calc_spectrum_filter_flux(filter_object=FilterObject, spectrum_object=MangledSpectrumObject, verbose=verbose) for
         FilterObject in data_table[data_table["mask"]]["filter_object"]])
    if verbose:
        print("params:", paramlist)
        print("weights:", weights)
        print("flux:", specflux)
        print("fitflux:", data_table[data_table["mask"]]["fitflux"].data)

    return data_table[data_table["mask"]]["fitflux"] - specflux ## minimising residual - better to do chisq? or minimise the sum?? TODO


## Extending Spectra

def bb_min(params, specphot, filters, wavelength, verbose=False):
    T = params["T"]
    flux_scale = params["flux_scale"]
    EBV = params["EBV"]

    if verbose: print(type(u.AA))

    bb_wavelength = np.array(wavelength) * u.AA
    bb_flux = bb.blackbody_lambda(bb_wavelength, temperature=T*u.Kelvin)
    # bb_flux = bb.blackbody_lambda(bb_wavelength, temperature=T*u.Kelvin)

    bb_spec = classes.SpectrumClass()
    bb_spec.load_table(Table([wavelength, bb_flux], names=("wavelength", "flux")))

    bb_spec.flux = extinction.unred(bb_spec.wavelength, bb_spec.flux, EBV=EBV)

    bb_spec.data["flux"] = bb_spec.data["flux"] / flux_scale
    bb_spec.flux = bb_spec.flux / flux_scale

    bb_spec.get_specphot(filters, verbose=verbose)

    residual = specphot["flux"] - bb_spec.specphot["flux"]

    #     return np.sum((residual)**2)
    return residual


def bb_min_fun(params, filters, wavelength, verbose=False):
    T = params["T"]
    flux_scale = params["flux_scale"]
    EBV = params["EBV"]

    if verbose:print(type(u.AA))

    bb_wavelength = np.array(wavelength) * u.AA
    bb_flux = bb.blackbody_lambda(bb_wavelength, temperature=T * u.Kelvin)
    # bb_flux = bb.blackbody_lambda(wavelength, temperature=T*u.Kelvin)

    bb_spec = classes.SpectrumClass()
    bb_spec.load_table(Table([wavelength, bb_flux], names=("wavelength", "flux")))

    bb_spec.flux = extinction.unred(bb_spec.wavelength, bb_spec.flux, EBV=EBV)

    bb_spec.data["flux"] = bb_spec.data["flux"] / flux_scale
    bb_spec.flux = bb_spec.flux / flux_scale

    bb_spec.get_specphot(filters, verbose=verbose)

    return bb_spec


def fit_bb(S, new_wavelength=False, new_min_wavelength=2000, new_max_wavelength=10000, T_guess=10000,
           flux_scale_guess=1e23, EBV_guess=0.0, correct_for_area=True, filter_dict=False, return_table=False,
           verbose=False):
    """

    :param S:
    :param new_wavelength:
    :param T_guess:
    :param flux_scale_guess:
    :param EBV_guess:
    :param correct_for_area:
    :param filter_dict:
    :param return_table:
    :param verbose:
    :return:
    """
    if not new_wavelength:
        new_wavelength = np.arange(new_min_wavelength, new_max_wavelength ) * u.Angstrom

    if not filter_dict:
        filter_dict = OrderedDict()
        for i, filter_name in enumerate(S._overlapping_filter_list):
            filter_dict[filter_name] = load_filter(os.path.join(defaults._default_filter_dir_path, filter_name + ".dat"))
            filter_dict[filter_name].calculate_edges()

    if not hasattr(S, "specphot"):
        warnings.warn("Spectrum has no specphot - calculating")
        S.get_specphot(filter_objects=filter_dict, correct_for_area=correct_for_area, verbose=verbose)

    params = Parameters()
    params.add("T", value=T_guess, min=0.0, max=50000, vary=True)  ## BB temp
    params.add("flux_scale", value=flux_scale_guess, vary=True)  ## Flux Scaling
    params.add("EBV", value=EBV_guess, vary=True)  ## Extinction

    out = minimize(bb_min, params, args=(S.specphot, filter_dict, S.wavelength),
                   kws=({"verbose": verbose}))

    if verbose: print(fit_report(out))

    best_bb = bb_min_fun(out.params, filter_dict, new_wavelength)

    w_blue = np.where(best_bb.wavelength < np.nanmin(S.wavelength))
    w_red = np.where(best_bb.wavelength > np.nanmax(S.wavelength))

    new_flux = np.append(best_bb.flux[w_blue], np.append(S.flux, best_bb.flux[w_red]))
    new_wavelength = np.append(best_bb.wavelength[w_blue], np.append(S.wavelength, best_bb.wavelength[w_red]))

    S._bb_extended = Table([new_wavelength, new_flux], names=("wavelength", "flux"))
    if return_table:
        return S._bb_extended
    else:
        pass

def flat_extend(S, extension_wavelength=False, new_max_wavelength=10000,
                return_table=False, verbose=False):
    """

    :param S:
    :param new_wavelength:
    :param new_max_wavelength:
    :param return_table:
    :param verbose:
    :return:
    """

    if not extension_wavelength:
        extension_wavelength = np.arange(np.nanmax(S.wavelength), new_max_wavelength) * u.Angstrom

    if new_max_wavelength < np.nanmax(S.wavelength):
        warnings.warn("No need to extend")
        if return_table:
            return S.data
        else:
            pass

    extension_flux = np.ones(len(extension_wavelength)) * np.mean(S.flux)

    extension_spec = classes.SpectrumClass()
    extension_spec.load_table(Table([extension_wavelength, extension_flux], names=("wavelength", "flux")), verbose=verbose)

    extended_spec = classes.SpectrumClass()
    extended_spec.load_table(vstack([S.data, extension_spec.data]), verbose=verbose)

    if return_table:
        return extended_spec.data
    else:
        pass



def linear_extend(S, extension_wavelength=False, new_max_wavelength=10000,
                return_table=False, verbose=False):
    """

    :param S:
    :param new_wavelength:
    :param new_max_wavelength:
    :param return_table:
    :param verbose:
    :return:
    """

    if not extension_wavelength:
        extension_wavelength = np.arange(np.nanmax(S.wavelength), new_max_wavelength ) * u.Angstrom
    if new_max_wavelength < np.nanmax(S.wavelength):
        warnings.warn("No need to extend")
        if return_table:
            return S.data
        else:
            pass

    extension_flux = np.linspace(S.flux[-1], 0, len(extension_wavelength))

    extension_spec = classes.SpectrumClass()
    extension_spec.load_table(Table([extension_wavelength, extension_flux], names=("wavelength", "flux")), verbose=verbose)

    extended_spec = classes.SpectrumClass()
    extended_spec.load_table(vstack([S.data, extension_spec.data]), verbose=verbose)

    if return_table:
        return extended_spec.data
    else:
        pass