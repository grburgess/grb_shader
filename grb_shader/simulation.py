import collections
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from popsynth import Population, silence_progress_bars, silence_warnings
from tqdm.auto import tqdm

from .grb_pop import GRBPop
from .catalog import LocalVolume

def play_god(param_file: str,
             n_sims: int = 10,
             n_cpus: int = 6,
             base_file_name: str = "pop",
             seed: int = 1234

             ) -> None:

    p: Path = Path(param_file)

    bp = Path(base_file_name).parent

    bp.mkdir(parents=True, exist_ok=True)
    
    silence_warnings()
    silence_progress_bars()

    with p.open("r") as f:

        setup = yaml.load(f, Loader=yaml.SafeLoader)

    def sim_one(i):
        setup["seed"] = int(seed + i * 10)
        gp = GRBPop.from_dict(setup)

        gp.engage()

        gp.population.writeto(f"{base_file_name}_{int(seed + i *10)}.h5")

    sims = Parallel(n_jobs=n_cpus)(delayed(sim_one)(i)
                              for i in tqdm(range(n_sims), desc="playing god"))


class RestoredSimulation(object):

    def __init__(self, sim_path: str):

        p = Path(sim_path)
        folder = p.parent
        self._populations = []

        for f in tqdm(list(folder.glob(f"{p.name}*.h5")), desc="loading populations"):

            self._populations.append(Population.from_file(f))

        self._catalog = LocalVolume.from_lv_catalog()

        self._count_galaxies()

    def _count_galaxies(self):
        self._hit_galaxy_names = collections.Counter()

        for pop in tqdm(self._populations, desc="counting galaxies"):
            self._catalog.read_population(pop)
            for galaxy in self._catalog.selected_galaxies:

                self._hit_galaxy_names.update([galaxy.name])


    @property
    def populations(self) -> List[Population]:
        return self._populations
    
    @property
    def fractions(self):

        return np.array([population.n_detections/population.n_objects for population in self._populations])

    @property
    def n_detections(self):

        return np.array([population.n_detections for population in self._populations])

    def hist_galaxies(self, n=10, exclude=[]):

        fig, ax = plt.subplots()

        hits = self._hit_galaxy_names.most_common(n)
        names = [x[0] for x in hits if x[0] not in exclude]
        counts = [x[1] for x in hits if x[0] not in exclude]

        x = np.arange(len(names))  # the label locations
        width = 0.35  # the width of the bars

        ax.barh(x, counts, width)

        ax.invert_yaxis()
        ax.set_yticks(x)
        ax.set_yticklabels(names)

        plt.tight_layout()
