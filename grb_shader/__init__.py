from .simulation import GBM_GRB_God, RestoredSimulation
from .catalog import LocalVolume
from .grb_pop import GRBPop

from .utils.package_data import get_path_of_data_file, get_ghirlanda_model


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
