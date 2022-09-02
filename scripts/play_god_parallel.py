# +
from grb_shader import GodMultiverse
from dask.distributed import LocalCluster, Client

from cosmogrb.instruments.gbm import GBM_CPL_Constant_Universe
# -

if __name__ == '__main__':

    # path where simulations are stored
    sim_path = "/data/eschoe/grb_shader/sims/220822/ghirl_c_t90fit_pulse/catalogselec/"
    # path of parameter file
    param_file = "ghirlanda2016_c_t90fit_pulse.yml"

    n_cores = 20
    n_sims = 500

    constant_temporal_profile = False
    catalog_selec = True
    internal_parallelization = False
    hard_flux_selec = False

    with LocalCluster(n_workers=n_cores,threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            print(client)
            
            print('Start simulations')

            multiverse = GodMultiverse(n_sims)
            #multiverse.process_surveys(surveys_path = sim_path,client=client)

            multiverse.go(
                param_file=param_file,
                pops_dir=sim_path,
                client=client,
                constant_temporal_profile=constant_temporal_profile,
                catalog_selec=catalog_selec,
                hard_flux_selec = hard_flux_selec,
                internal_parallelization=internal_parallelization
                )

            print('Successfully done')
