"""

"""

import os
import warnings
import subprocess
import copy

import numpy as np

from . import classes
from . import defaults
from . import functions
from . import utils
from . import kcorr

__all__ = [
            "test_LCfit",
            "run_LCfit",
            "run_LCfit_fileinput",
            "run_all_SNe_LCfit",
            "test_specfit",
            "run_specfit",
            "run_cocospecfit",
            "get_all_spec_lists",
            "specfit_all",
            "specfit_sn",
            "run_specphase",
            "check_specphase",
           ]


#  #------------------------------------#  #
#  # CoCo Functions                     #  #
#  #------------------------------------#  #


def test_LCfit(snname, coco_dir = defaults._default_coco_dir_path,
               verbose = False):
    """
    Check to see if a fit has been done. Does this by
    looking for reconstructed LC files

    :param snname:
    :param coco_dir:
    :param verbose:
    :return:
    """


    # try:
    #     if not coco_dir:
    #         coco_dir = defaults._default_coco_dir_path
    #
    # except:
    #     warnings.warn("Something funky with your input")

    utils.check_dir_path(coco_dir)

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


def run_LCfit(path, coco_dir = defaults._default_coco_dir_path, model = False,
              testrun=False, verbose = False):
    """

    :param path:
    :param coco_dir:
    :param model:
    :param verbose:
    :return:
    """

    utils.check_file_path(path)
    utils.relist() ## Check filter file is up to date

    if model:
        models = np.unique([i.split(".")[0] for i in os.listdir(os.path.join(defaults._default_coco_dir_path, "src/models"))])

        if model not in models:
            return False
        callargs = [os.path.join(defaults._default_coco_dir_path, "lcfit"), path, "-m", model]
        print("running with", model)
    else:
        print("No Model supplied - running with default")
        callargs = [os.path.join(defaults._default_coco_dir_path, "lcfit"), path]

    if verbose: print("Running CoCo lcfit on " + path)
    if verbose: print("callargs are ", callargs)

    cwd = os.getcwd()
    os.chdir(coco_dir)

    if not testrun:
        subprocess.call(callargs)
    else:
        warnings.warn("Test run - not generating subprocess")

    os.chdir(cwd)
    if verbose: print("Fit complete")
    pass


def run_LCfit_fileinput(listfile_path, coco_dir = defaults._default_coco_dir_path,
                        data_dir = defaults._default_data_dir_path, verbose = False):
    """

    :param listfile_path:
    :param coco_dir:
    :param data_dir:
    :param verbose:
    :return:
    """

    utils.check_file_path(listfile_path)

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


def run_all_SNe_LCfit(verbose = False):
    """

    :param verbose:
    :return:
    """
    ## TODO
    pass


def test_specfit(snname, coco_dir = False,
               verbose = False):
    """
    Check to see if a fit has been done. Does this by
    looking for reconstructed .spec files

    :param snname:
    :param coco_dir:
    :param verbose:
    :return:
    """

    try:
        if not coco_dir:
            coco_dir = defaults._default_coco_dir_path

    except:
        warnings.warn("Something funky with your input")

    utils.check_dir_path(coco_dir)

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


def run_specfit(SNObject, wantedfilters=False, anchor_distance=1000, save=True, plot = False,
                coco_dir=defaults._default_coco_dir_path, overwrite=False, verbose = False):
    """
    replacement for `run_cocospecfit`. Mangles the spectra in the listfiles. Built for comfort, not speed.

    :param SNObject:
    :param wantedfilters:
    :param anchor_distance:
    :param save:
    :param plot:
    :param coco_dir:
    :param verbose:
    :return:
    """

    if not wantedfilters:
        wantedfilters = SNObject.phot.filter_names.data

    outfile_log = []
    if hasattr(SNObject, "spec") and hasattr(SNObject, "lcfit"):
        if verbose: print("hasattr spec and lcfit")

        if len(SNObject.spec) == 0:
            warnings.warn("No spectra loaded in.")
            return
        else:
            for name, mS in SNObject.spec.items():
                if verbose: print(name, mS)

                S = copy.deepcopy(mS)
                fit_dict = kcorr.mangle(SNObject, mS, mS.mjd_obs, wantedfilters, anchor_distance=anchor_distance, verbose=verbose)
                if plot:
                    functions.plot_mangledata(S, fit_dict["data_table"], mS=fit_dict["SpectrumObject"], spl=fit_dict["final_spl"],
                                        show_linear_extrap=True, normalise=True)

                outfile = SNObject.name + "_" + str(fit_dict["SpectrumObject"].mjd_obs).ljust(12, "0") + ".spec"

                while outfile in outfile_log:
                    j = 1
                    outfile = SNObject.name + "_" + str(fit_dict["SpectrumObject"].mjd_obs + j * 0.00001).ljust(12, "0") + ".spec"
                    j += 1
                if verbose: print(outfile)
                outfile_log.append(outfile)

                if save:
                    kcorr.save_mangle(fit_dict["SpectrumObject"], outfile, fit_dict["SpectrumObject"].infile, squash=overwrite, verbose=verbose)
    else:
        print("SNObject needs lcfit and spectra")
    # utils.check_file_path(path)
    # utils.relist() ## Check filter file is up to date
    # cwd = os.getcwd()
    # os.chdir(coco_dir)
    # if verbose: print("Running CoCo specfit on " + path)
    # subprocess.call([os.path.join(defaults._default_coco_dir_path, "./specfit"), path])
    # os.chdir(cwd)
    pass


def run_cocospecfit(path, coco_dir=defaults._default_coco_dir_path, verbose = False):
    """
    runs CoCo specfit on the listfile supplied in path

    :param path:
    :param coco_dir:
    :param verbose:
    :return:
    """
    utils.check_file_path(path)
    utils.relist() ## Check filter file is up to date
    cwd = os.getcwd()
    os.chdir(coco_dir)
    if verbose: print("Running CoCo specfit on " + path)
    subprocess.call([os.path.join(defaults._default_coco_dir_path, "./specfit"), path])
    os.chdir(cwd)
    pass


def get_all_spec_lists(dirpath = defaults._default_list_dir_path, ignore_m = True, verbose=False):
    """

    :param dirpath:
    :param ignore_m:
    :param verbose:
    :return:
    """

    ignore = [".DS_Store", "master.list", "lightcurves.list"]

    utils.check_dir_path(dirpath)

    if verbose: print(dirpath)

    listfile_list = [i for i in os.listdir(dirpath) if i not in ignore]

    if ignore_m:
        listfile_list = [k for k in listfile_list if "_m.list" not in k]

    fullpath_list = [os.path.join(dirpath, j) for j in listfile_list]

    return fullpath_list


def specfit_all(dirpath=defaults._default_list_dir_path, overwrite=False, verbose=True):
    """

    :param verbose:
    :param dirpath:
    :return:
    """

    fullpath_list = get_all_spec_lists(dirpath)

    snnames = [utils.get_snname_from_listfile(i) for i in fullpath_list]

    for i, sninfo in enumerate(zip(snnames, fullpath_list)):
        snname = sninfo[0]
        listpath = sninfo[1]

        if verbose: print(i, snname, listpath)

        # run_specfit(path, plot=plot, verbose=verbose)
        specfit_sn(snname=snname, listpath=listpath, overwrite=overwrite, verbose=verbose)
        if verbose: print("Done")

    pass


def specfit_sn(SNobject = False , snname = False, listpath = False, photpath = False, fitpath = False,
               anchor_distance=1000, save=True, plot=False, coco_dir=defaults._default_coco_dir_path,
               save_new_list = False, overwrite=False, verbose = False):
    """
    runs CoCo specfit on the listfile supplied in path.

    :param snname:
    :param verbose:
    :return:
    """

    if SNobject:
        snname = SNobject.name
        pass
    elif snname:
        SNobject = classes.SNClass(snname)

        if not photpath:
            photpath = os.path.join(defaults._default_data_dir_path, "lc/" + snname + ".dat")

        SNobject.load_phot(path = photpath, verbose=verbose)

    else:
        warnings.warn("Need to provide either SNobject or snname")
        return

    if not hasattr(SNobject, "lcfit"):
        if not fitpath:
            fitpath = os.path.join(defaults._default_recon_dir_path, snname + ".dat")
        SNobject.get_lcfit(fitpath, verbose=verbose)

    if len(SNobject.spec) == 0:
        if not listpath:
            listpath = os.path.join(defaults._default_list_dir_path, snname + ".list")
        if verbose:
            print("No spectra, Loading from list: ")
            print(listpath)
        SNobject.load_list(listpath, verbose=verbose)
        SNobject.load_spec(verbose=verbose)

    # ## Need to look for the recon lc files for snname
    # # sn = classes.SNClass(snname)
    # lcfit = classes.LCfitClass()
    #
    # path = os.path.join(lcfit.recon_directory, snname+".dat")
    # lcfit.load_formatted_phot(path)
    # lcfit.unpack()
    # lcfit._sort_phot()
    # lcfit.get_fit_splines()

    ## Need to make new recon lc files for mangling - no overlaps
    # lcfit.
    manglefilters = [i for i in SNobject.lcfit.filter_names]
    print(manglefilters)
    if "BessellR" in manglefilters and "SDSS_r" in manglefilters:
        if verbose:
            print("Has both BessellR and SDSS_r - using SDSS_r")
        ## Only use SDSS_r - less hassle
        manglefilters.remove("BessellR")
    if "BessellI" in manglefilters and "SDSS_i" in manglefilters:
        if verbose:
            print("Has both BessellI and SDSS_i - using SDSS_i")
        ## Only use SDSS_i - less hassle
        manglefilters.remove("BessellI")

    filename = snname + "_m.dat"
    outpath = SNobject.lcfit.recon_directory

    if verbose: print(outpath, filename)

    SNobject.lcfit.save(filename, path = outpath, filters = manglefilters, squash = True, verbose=verbose)

    # lcfit.
    ## Need to change the listfile to one that has snname matches the new lc file

    if not listpath:
        listpath = os.path.join(defaults._default_coco_dir_path, "lists", snname + ".list")

    origlist = utils.read_list_file(listpath)
    origlist.rename_column('snname', 'snname_nomangle')
    origlist["snname"] = [j+"_m" for j in origlist["snname_nomangle"]]

    newlistpath = os.path.join(defaults._default_coco_dir_path, "lists", snname + "_m.list")
    newlist = origlist["spec_path", "snname", "mjd_obs", "z"]
    # newlist.write(newlistpath, format = "ascii.fast_commented_header", overwrite = True)
    if save_new_list:
        newlist.write(newlistpath, format = "ascii.fast_commented_header")


    ## Need to call run_specfit with path of new list file
    # SNObject, wantedfilters=False, anchor_distance=1000, save=True, plot = False, coco_dir=defaults._default_coco_dir_path, verbose = True)

    # run_specfit(newlistpath)

    run_specfit(SNObject=SNobject, wantedfilters=manglefilters, save=save, anchor_distance=anchor_distance, plot=plot,
                coco_dir=coco_dir, overwrite=overwrite, verbose=verbose)
    pass


def run_specphase(filtername, phase_path=False, filetype=".dat", coco_dir=defaults._default_coco_dir_path,
                  verbose = False, model=False, recon_dir=defaults._default_recon_dir_path, test=False):
    """
    runs CoCo specphase.

    :param filtername:
    :param phase_path:
    :param filetype:
    :param coco_dir:
    :param verbose:
    :param model:
    :return:
    """
    if not phase_path:
        phase_path=os.path.join(defaults._default_coco_dir_path, "examples/phase.list")

    if model:
        models = np.unique([i.split(".")[0] for i in os.listdir(os.path.join(defaults._default_coco_dir_path, "src/models"))])

        if model not in models:
            print("Model", model, "not recognised.")
            return False

        callargs = [os.path.join(defaults._default_coco_dir_path, "./specphase"), phase_path, filtername, "-m", model]
        if verbose: print("running with", model)
    else:
        if verbose: print("No Model supplied - running with default")
        callargs = [os.path.join(defaults._default_coco_dir_path, "./specphase"), phase_path, filtername]

    filters = utils._get_current_filter_registry()

    if filtername+filetype not in filters:
        warnings.warn("Filtername not recognised")
        return False

    utils.check_file_path(phase_path)

    _check_reconspec_sn_in_listfile(phase_path=phase_path, recon_dir=recon_dir, verbose=verbose)

    utils.relist() ## Check filter file is up to date
    cwd = os.getcwd()
    os.chdir(coco_dir)
    # if verbose: print("Running CoCo specfit on " + path)
    if verbose:
        print(callargs)

    if not test:
        subprocess.call(callargs)
    else:
        warnings.warn("Just testing - skipping actual call")

    os.chdir(cwd)
    pass


def check_specphase(snname, spectra_dir="spectra/", coco_dir=defaults._default_coco_dir_path, absolute_path=False,
                    verbose = False):
    """

    :param verbose:
    :param snname:
    :param spectra_dir:
    :param coco_dir:
    :param absolute_path:
    :return:
    """
    if not absolute_path:
        spectra_dir = os.path.join(coco_dir, spectra_dir)

    utils.check_dir_path(spectra_dir)

    dir_list = [spec for spec in os.listdir(spectra_dir) if spec[:len(snname)] == snname]
    if verbose:
        print(len(dir_list), "files found matching", snname)
        print(dir_list)
    return dir_list


def _check_reconspec_sn_in_listfile(phase_path=os.path.join(defaults._default_coco_dir_path, "examples/phase.list"),
                                    recon_dir=defaults._default_recon_dir_path,
                                    verbose=True):
    """

    :param listfile:
    :param recon_dir:
    :param verbose:
    :return:
    """

    phase_list = np.sort(np.array(utils.read_phasefile(filepath=phase_path)["snname"]))

    recon_sn_list = utils.find_unique_SN_in_recon(dir_path=recon_dir)

    for sn in recon_sn_list:
        if verbose: print("Checking", sn, end="")
        if sn not in phase_list:
            warnings.warn(str(sn) + " not in phase list " + str(phase_path))
            if verbose: print("")
        else:
            if verbose: print("... Found.")
            pass
    pass
