from pathlib import Path

from popsynth import Population

import yaml
from joblib import Parallel, delayed

from .grb_pop import GRBPop


def play_god(param_file: str,
             n_sims: int = 10,
             n_cpus: int = 6,
             base_file_name: str = "pop",
             seed: int = 1234

             ) -> None:

    p: Path = Path(param_file)

    with p.open("r") as f:

        setup = yaml.load(f, Loader=yaml.SafeLoader)

    def sim_one(i):
        setup["seed"] = int(seed + i * 10)
        gp = GRBPop.from_dict(setup)

        gp.engage()

        gp.population.writeto(f"{base_file_name}_{int(seed + i *10)}.h5")



class RestoredSimulation(object):

    def __init__(self, population_file: str):

        


        
        
