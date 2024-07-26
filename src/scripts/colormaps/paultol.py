r"""
This file contains a collection of colorblind-safe color palettes by Paul Tol.
More information can be found at the website https://personal.sron.nl/~pault/
"""

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

###############################################################################
#                                                                             #
#       QUALITATIVE COLORMAPS                                                 #
#                                                                             #
###############################################################################

# Bright: for lines and labels
bright = ListedColormap(
    ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
     '#BBBBBB'],
    name='paultol_bright')
# High-contrast: for monochrome vision and printouts
highcontrast = ListedColormap(
    ['#004488', '#DDAA33', '#BB5566'],
    name='paultol_highcontrast')
# Vibrant: for TensorBoard
vibrant = ListedColormap(
    ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988',
     '#BBBBBB'],
    name='paultol_vibrant')
# Muted: for more colors
muted = ListedColormap(
    ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255',
     '#44AA99', '#999933', '#AA4499', '#DDDDDD'],
    name='paultol_muted')
# Medium-contrast: for color pairs (1 with 4, 2 with 5, 3 with 6)
mediumcontrast = ListedColormap(
    ['#6699CC', '#004488', '#EECC66', '#994455', '#997700', '#EE99AA'],
    name='paultol_mediumcontrast')
# Pale: for the background of black text
pale = ListedColormap(
    ['#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC', '#DDDDDD'],
    name='paultol_pale')
# Dark: for the background of white text or for text on a white background
dark = ListedColormap(
    ['#222255', '#225555', '#225522', '#666633', '#663333', '#555555'],
    name='paultol_dark')
# Light: for filling labeled cells
light = ListedColormap(
    ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99',
     '#BBCC33', '#AAAA00', '#DDDDDD'],
    name='paultol_light')


###############################################################################
#                                                                             #
#       SEQUENTIAL COLORMAPS                                                  #
#                                                                             #
###############################################################################

# YlOrBr: to be used as given or linearly interpolated
ylorbr = LinearSegmentedColormap.from_list('paultol_ylorbr',
    ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29', 
     '#EC7014', '#CC4C02', '#993404', '#662506'])
ylorbr.set_bad('#888888')
# YlOrBr variant with white value
ylorbr_white = LinearSegmentedColormap.from_list('paultol_ylorbr_white',
    ['#FFFFFF', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29', 
     '#EC7014', '#CC4C02', '#993404', '#662506'])
ylorbr_white.set_bad('#888888')
# Short version of YlOrBr more easily displayed on a white background
ylorbr_short = LinearSegmentedColormap.from_list('paultol_ylorbr_short',
    ['#FEE391', '#FEC44F', '#FB9A29', '#EC7014', '#CC4C02', '#993404', '#662506'])
ylorbr_short.set_bad('#888888')
# Iridescent: colors should be linearly interpolated, optionally extended
# toward white and black
iridescent = LinearSegmentedColormap.from_list('paultol_iridescant',
    ['#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF', '#D0E7CA', 
     '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1', '#8DCBE4', '#81C4E7', 
     '#7BBCE7', '#7EB2E4', '#88A5DD', '#9398D2', '#9B8AC4', '#9D7DB2', 
     '#9A709E', '#906388', '#805770', '#684957', '#46353A'])
iridescent.set_bad('#999999')
# Iridescant variant extended to white and black
iridescent_wb = LinearSegmentedColormap.from_list('paultol_iridescant_wb',
    ['#FFFFFF', '#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF', '#D0E7CA', 
     '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1', '#8DCBE4', '#81C4E7', 
     '#7BBCE7', '#7EB2E4', '#88A5DD', '#9398D2', '#9B8AC4', '#9D7DB2', 
     '#9A709E', '#906388', '#805770', '#684957', '#46353A', '#000000'])
iridescent_wb.set_bad('#999999')
# Incandescent: not printer-friendly
incandescent = LinearSegmentedColormap.from_list('paultol_incandescent',
    ['#CEFFFF', '#C6F7D6', '#A2F49B', '#BBE453', '#D5CE04', '#E7B503', 
     '#F19903', '#F6790B', '#F94902', '#E40515', '#A80003'])
incandescent.set_bad('#888888')
# Smooth rainbow: The colours are meant to be linearly interpolated. Often it 
# is better to use only a limited range of these colours. Starting at purple, 
# bad data can be shown white, whereas starting at off-white, the most distinct 
# grey is given in the circle. If the lowest data value occurs often, start at 
# off-white instead of purple. If the highest data value occurs often, end at 
# red instead of brown. For colour-blind people, the light purples and light 
# blues should not be mixed much.
LinearSegmentedColormap.from_list('paultol_smooth_rainbow',
    ['#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2', '#A778B4', 
     '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', 
     '#4D8AC6', '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190', 
     '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', 
     '#E49C39', '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222', 
     '#B8221E', '#95211B', '#721E17', '#521A13'])
