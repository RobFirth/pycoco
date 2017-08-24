# pycoco
___

## v0.7.2
___
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
___
This is the development repo for the python frontend for the core-collapse SNe template code 'CoCo':  https://github.com/UoS-SNe/CoCo
(my fork is currently https://github.com/RobFirth/CoCo).

CoCo was originally started by Natasha Karpenka, and is currently being updated and maintained by S. Prajs (https://github.com/SzymonPrajs).

A paper, Firth et. al. 2017, is currently in prep.
___


 * File I/O and interaction with `CoCo LCfit` output now operational - 11/01/17

 * Usage of a `SN` class now solid

 * Implementing calls to `CoCo LCfit` now

 * Calls to `CoCo specfit` now implemented

 * Mangling now stable

 * Adding tools for calculating magnitude offsets to make it easier to import new data

 * regeneration of filter list file now possible

 * added colours and bandpasses for LSST filters

 * Less dependence on environment variables

 * installation through `setup.py` now possible

 * better handing of CoCo sim outputs

 * SN position and mu now stored in infofile `./testdata/info/info.dat`

 * dark sky calculations and integration with LSST Throughputs now done

 * Improved stability

 * input from `astropy tables` now more straightforward - better integration with `coco.simulate`

 * can now batch fit light curves and spectra from within python
___


To install, run:

```
git clone https://github.com/RobFirth/verbose-enigma.git
```

then:

```
cd verbose-enigma
python setup.py install --user
```

(The --user argument only installs current user only, omitting flag will install for all users on the system if there are appropriate permissions)

_NOTE: make sure that the python used to install is the one that you will use with pycoco_
___

Ideally set the following environment variables:

`COCO_ROOT_DIR` (my default is `~/Code/CoCo/`)
`PYCOCO_FILTER_DIR`(my default is `~/Code/CoCo/data/filters/`)
`PYCOCO_DATA_DIR` (my default is `~/Code/CoCo/data/`)
`SFD_DIR` (my default is `~/data/Dust/sfddata-master/`; see below)
`LSST_THROUGHPUTS` (my default is `${HOME}/projects/LSST/throughputs`)
`LSST_THROUGHPUTS_BASELINE` (my default is `${LSST_THROUGHPUTS}/baseline`)

also `pycoco/` and `CoCo` need to be in your path and pythonpath, i.e.:

 ```
 setenv PATH /Users/berto/Code/pycoco/:$PATH

 setenv PYTHONPATH "/Users/berto/Code/pycoco/:$PYTHONPATH
 ```

___

# Requirements
## python packages

* matplotlib
* numpy
* scipy
* astropy
* sfdmap (https://github.com/kbarbary/sfdmap)

# additionally

* lsst throughputs (https://github.com/lsst/throughputs; `$LSST_THROUGHPUTS` points to this directory)
___


for `sfdmap`, the environment variable `SFD_DIR` needs to point at the path to the parent directory of the appropriate dust map files. See the installation instructions here: https://github.com/kbarbary/sfdmap
___  
