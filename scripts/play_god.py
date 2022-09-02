from grb_shader import GodMultiverse
from dask.distributed import LocalCluster, Client

from grb_shader.utils.logging import setup_log
import logging as log


if __name__ == '__main__':
    logger = setup_log(__name__)
    logger.setLevel(log.DEBUG)

    # path where simulations are stored
    sim_path = "/data/eschoe/grb_shader/sims/test/"
    # path of parameter file
    param_file = "ghirlanda2016_c_t90fit_pulse.yml"

    n_sims = 1

    constant_temporal_profile = False
    catalog_selec = False
    internal_parallelization = False
    hard_flux_selec = False

    print('Start simulations')

    multiverse = GodMultiverse(n_sims)

    #multiverse.process_surveys(surveys_path = sim_path)

    multiverse.go(
        param_file=param_file,
        pops_dir=sim_path,
        client=None,
        constant_temporal_profile=constant_temporal_profile,
        catalog_selec=catalog_selec,
        hard_flux_selec = hard_flux_selec,
        internal_parallelization= internal_parallelization,
        )


    print('Successfully done')


