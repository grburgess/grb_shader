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

class GBM_GRB_God(object):

    def __init__(
        self,
        n_sims: int = 10,
        constant_profile: bool = False
    ):
        """Generate populations of GRBs and GBM data

        :param n_sims: number of populations that is generated, defaults to 10
        :type n_sims: int, optional
        :param constant_profile: if true: use constant temporal light curve, if false: Norris pulse profile, defaults to False
        :type constant_profile: bool, optional
        """

        self.base_file_name: str = ''

        self.n_sims: int = n_sims

        self.constant_profile: bool = constant_profile
        
        silence_warnings()
        silence_progress_bars()

    def go_grb_pop(
        self,
        param_file: str,
        n_cpus: int = 6,
        base_file_name: str = "pop",
        seed: int = 1234,
        catalog_selec: bool = False,
        hard_flux_selec: bool = False,
        hard_flux_lim: float = 1e-7 #erg cm^-2 s^-1
        ):
        """Generate populations of GRBs

        :param param_file: path to file determining redshift and luminosity distribution parameters
        :type param_file: str
        :param n_cpus: number of cpus used to create populations, defaults to 6
        :type n_cpus: int, optional
        :param base_file_name: prefix for created popsynth files, defaults to "pop"
        :type base_file_name: str, optional
        :param seed: seed for random sampling, defaults to 1234
        :type seed: int, optional
        :param catalog_selec: if true, select GRBs coinciding with location of LV galaxy, defaults to False
        :type catalog_selec: bool, optional
        :param hard_flux_selec: if true, apply hard flux selection, defaults to False
        :type hard_flux_selec: bool, optional
        :param hard_flux_lim: lower limit hard flux selection, defaults to 1e-7#ergcm^-2s^-1
        :type hard_flux_lim: float, optional
        """
        
        self.base_file_name = base_file_name

        bp: Path = Path(self.base_file_name).parent

        bp.mkdir(parents=True, exist_ok=True)

        #read parameter file specifying distributions
        p: Path = Path(param_file)

        with p.open("r") as f:

            setup = yaml.load(f, Loader=yaml.SafeLoader)

        setup["catalog_selec"] = catalog_selec
        setup["hard_flux_selec"] = hard_flux_selec
        setup["hard_flux_lim"] = hard_flux_lim

        #function for one simulated population with params set in file p
        def sim_one(i):
            setup["seed"] = int(seed + i * 10)
            gp = GRBPop.from_dict(setup)

            gp.engage()

            gp.population.writeto(f"{base_file_name}_{int(seed + i *10)}.h5")

        #in parallel 
        sims = Parallel(n_jobs=n_cpus)(delayed(sim_one)(i)
                                for i in tqdm(range(self.n_sims), desc="playing god - GRB pops"))

    def go_gbm_data(self):
        
        pass


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

    @property
    def n_grbs(self):
        return np.array([population.n_objects for population in self._populations])

    def hist_galaxies(self, n=10, exclude=[]):

        fig, ax = plt.subplots()

        hits = self._hit_galaxy_names.most_common(n)
        names = [x[0] for x in hits if x[0] not in exclude]
        counts = [x[1] for x in hits if x[0] not in exclude]

        x = np.arange(len(names))  # the label locations
        width = 0.35  # the width of the bars

        ax.barh(x, counts, width)

        ax.set_xlabel('Number of spatial coincidences')

        ax.invert_yaxis()
        ax.set_yticks(x)
        ax.set_yticklabels(names)

        plt.tight_layout()

        return fig

    def hist_n_GRBs(self, ax=None, **kwargs):

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()

        ax.hist(self.n_grbs, **kwargs)

        ax.set_xlabel('Number GRBs')
        plt.tight_layout()

        return fig

    def hist_n_detections(self, ax=None,**kwargs):

        if ax is None:

            fig, ax = plt.subplots()

            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()

        ax.hist(self.n_detections, **kwargs)

        plt.tight_layout()

        return fig

