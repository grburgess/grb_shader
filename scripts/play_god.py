from grb_shader import God_Multiverse,get_path_of_data_file
from dask.distributed import LocalCluster, Client

if __name__ == '__main__':

    # path where simulations are stored
    sim_path = "/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/catalogselec/"
    # path of parameter file
    param_file = "ghirlanda2016_c_t90fit.yml"

    n_cores = 25
    n_sims = 500

    constant_temporal_profile = True
    catalog_selec = True
    internal_parallelization = False
    hard_flux_selec = False

    with LocalCluster(n_workers=n_cores) as cluster:
        with Client(cluster) as client:
            print('Start simulations')

            multiverse = God_Multiverse(n_sims)

            multiverse.go(
                param_file=param_file,
                pops_dir=sim_path,
                client=client,
                constant_temporal_profile=constant_temporal_profile,
                catalog_selec=catalog_selec,
                hard_flux_selec = hard_flux_selec,
                internal_parallelization= internal_parallelization,
                )


    print('Successfully done')


