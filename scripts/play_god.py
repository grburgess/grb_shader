from grb_shader import GBM_GRB_God,get_path_of_data_file

import popsynth
popsynth.update_logging_level("INFO")
popsynth.silence_progress_bars()
popsynth.silence_warnings()

#path where simulations are stored
sim_path = "/data/eschoe/grb_shader/popsynth/220712_c_ghirl/catalogselec/pop"
param_file = "ghirlanda2016_c_constant.yml"

god = GBM_GRB_God(n_sims=500,constant_profile=False)
god.go_grb_pop(
    param_file=get_path_of_data_file(param_file),
    n_cpus=10, 
    base_file_name=sim_path,
    catalog_selec=True,
    hard_flux_selec=False
    )



