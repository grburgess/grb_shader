from .multiverse import GodMultiverse,RestoredMultiverse
from .catalog import LocalVolume
from .grb_pop import GRBPop

from .restored_simulation import RestoredUniverse

from .utils.package_data import get_path_of_data_file, get_ghirlanda_model,get_path_of_config_file
from .utils.logging import setup_log

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


#configuration matplotlib
from matplotlib import rc_file
rc_file(get_path_of_config_file('matplotlibrc'))