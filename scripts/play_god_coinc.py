from grb_shader import GodMultiverse
from dask.distributed import LocalCluster, Client

from grb_shader.utils.logging import setup_log
import logging as log
import numpy as np


if __name__ == '__main__':
    logger = setup_log(__name__)
    logger.setLevel(log.DEBUG)

    # path where simulations are stored
    sim_path = "/data/eschoe/grb_shader/sims/only_coinc/221110/"
    # path of parameter file
    param_file = 'ghirlanda2016_c_t90fit_r0_inc.yml'

    n_sims =30
    n_cores=30
    uncertainties = np.geomspace(1./60.,3,50)

    constant_temporal_profile = True
    catalog_selec = True
    internal_parallelization = False
    hard_flux_selec = False
    with_unc = True

    print('Start simulations')
    
    i=0
    
    with LocalCluster(n_workers=n_cores,threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            print(client)
    
            for unc in uncertainties:

                multiverse = GodMultiverse(n_sims)

                #multiverse.process_surveys(surveys_path = sim_path)

                multiverse.go_pops(
                    param_file=param_file,
                    pops_dir=sim_path+str(np.round(unc,4)),
                    constant_temporal_profile=constant_temporal_profile,
                    seed=1234,
                    client = client,
                    catalog_selec=catalog_selec,
                    hard_flux_selec = hard_flux_selec,
                    with_unc=with_unc,
                    unc_circular_angle=unc, #deg
                    n_samp=1000
                    )
                
                i+=1

            print('Successfully done')


