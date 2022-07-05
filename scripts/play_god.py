from grb_shader import play_god,get_path_of_data_file

import popsynth
popsynth.update_logging_level("INFO")
popsynth.silence_progress_bars()
popsynth.silence_warnings()

#path where simulations are stored
sim_path = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/b2/pop"
param_file = "ghirlanda2016_b_triangle.yml"

play_god(
    param_file=get_path_of_data_file(param_file),
    n_sims=500,
    n_cpus=8, 
    base_file_name=sim_path
    )

