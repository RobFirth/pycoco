"""
Module Docstring
"""
import collections

__all__ = ["_colourmap_name", "rgb", "rgb255", "triplet", "hex"]

_colourmap_name = 'plasma'

hex = collections.OrderedDict()
## from flatuicolors.com
hex['midnightblue'] =       '#2c3e50'
hex['wetasphalt'] =         '#34495e'
hex['wisteria'] =           '#8e44ad'
hex['amethyst'] =           '#9b59b6'
hex['peterriver'] =         '#3498db'
hex['belize'] =             '#2980b9'
hex['emerald'] =            '#40d47e'
hex['nephritis'] =          '#2ecc71'
hex['turquoise'] =          '#1abc9c'
hex['greensea'] =           '#16a085'
hex['sunflower'] =          '#f1c40f'
hex['orange'] =             '#f39c12'
hex['carrot'] =             '#e67e22'
hex['pumpkin'] =            '#d35400'
hex['alizarin'] =           '#e74c3c'
hex['pomegranite'] =        '#c0392b'
hex['clouds'] =             '#ecf0f1'
hex['silver'] =             '#bdc3c7'
hex['concrete'] =           '#95a5a6'
hex['asbestos'] =           '#7f8c8d'
hex['tungsten'] =           '#4F5858'
hex['batman'] =             '#303535'
hex['black'] =              '#000000'

hex['light_wisteria'] =     '#E3D0EA'
hex['light_peterriver'] =   '#CCE5F6'
hex['light_belize'] =       '#CADFED'
hex['light_emerald'] =      '#CFF4DF'
hex['light_sunflower'] =    '#FBF0C3'
hex['light_asbestos'] =     '#DFE2E3'
hex['light_LTr'] =          '#CB8D87'
hex['light_alizarin'] =     '#F6BCB6'

## Others to use
hex['red'] = '#ff0000'
hex['green'] = '#0F971A'
hex['blue'] = '#0000ff'

hex['p48R'] = '#F01414'
hex['P48R'] = '#F01414'
hex['PTF48_R'] = '#F01414'
hex['P48g'] = '#14F014'
hex['PTF48_g'] = '#14F014'
hex['p48g'] = '#14F014'
hex['LTr'] = '#971A0F'
hex['LTg'] = '#0F971A'
hex['LTi'] = '#fd00ff'
hex['LSQgr'] = '#8DC63F'

hex['LCOGTB'] = hex['belize']
hex['LCOGTV'] = hex['sunflower']
hex['LCOGTg'] = hex['nephritis']
hex['LCOGTr'] = hex['pomegranite']
hex['LCOGTi'] = hex['wisteria']

hex['LOG7'] = '#ff6600'
hex['LOG8'] = '#cc0000'
hex['I-shift'] = '#660066'


hex['SDSS_u'] = '#3498db'
hex['SDSS_g'] = '#00ff00'
hex['SDSS_r'] = '#ff0000'
hex['SDSS_i'] = '#fd00ff'
hex['SDSS_z'] = '#660066'

hex["DES_g"] = hex['SDSS_g']
hex["DES_r"] = hex['SDSS_r']
hex["DES_i"] = hex['SDSS_i']
hex["DES_z"] = hex['SDSS_z']

hex['u'] = '#40d47e'
hex['b'] = '#0000ff'
hex['v'] = '#ffff00'
hex['r'] = '#c0392b'
hex['i'] = '#8e44ad'
hex['z'] = '#000000'

hex['U'] = hex['u']
hex['B'] = hex['b']
hex['V'] = hex['v']
hex['R'] = hex['r']
hex['I'] = hex['i']
hex['Z'] = hex['z']

hex['BessellU'] = hex['u']
hex['BessellB'] = hex['b']
hex['BessellV'] = hex['v']
hex['BessellR'] = hex['r']
hex['BessellI'] = hex['i']
hex['BessellZ'] = hex['z']

hex['LSST_u'] = '#3498db'
hex['LSST_g'] = '#00ff00'
hex['LSST_r'] = '#ff0000'
hex['LSST_i'] = '#fd00ff'
hex['LSST_z'] = '#660066'
hex['LSST_y'] = '#000000'

## Random Others
hex['white'] = '#FFFFFF'

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'

def rgb(triplet):
    triplet = triplet.replace('#', '')

    return _HEXDEC[triplet[0:2]]/255.0, _HEXDEC[triplet[2:4]]/255.0, _HEXDEC[triplet[4:6]]/255.0

def rgb255(triplet):
    triplet = triplet.replace('#', '')

    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]

def triplet(rgb, lettercase=LOWERCASE):
    return format(rgb[0]<<16 | rgb[1]<<8 | rgb[2], '06'+lettercase)

RGB = dict()
RGB255 = dict()
for i in hex.keys():
    RGB[i] = rgb(hex[i])
    RGB255[i] = rgb255(hex[i])
