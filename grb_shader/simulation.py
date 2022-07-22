import collections
from pathlib import Path
from typing import List
from wsgiref.util import setup_testing_defaults
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dask.distributed import Client

from popsynth import Population, silence_progress_bars, silence_warnings

from tqdm.auto import tqdm

from .grb_pop import GRBPop
from .catalog import LocalVolume
from .utils.package_data import get_path_of_data_file
from .utils.logging import setup_log

logger = setup_log(__name__)

from cosmogrb.instruments.gbm import GBM_CPL_Universe, GBM_CPL_Constant_Universe
from cosmogrb.universe.survey import Survey
from cosmogrb.instruments.gbm.gbm_trigger import GBMTrigger


class God_Multiverse(object):

    def __init__(
        self,
        n_universes: int
    ):
        self._n_universes = n_universes #number of Universes
        
        self._pops_computed = False     #if pops were computed or loaded
        self._universes_computed = False    #if GBM Universes were computed or loaded
        self._surveys_processed = False     #if GBM trigger was applied or files were loaded

        #set by go_pops or load_pops
        self._pops_dir = None
        self._pops_base_file_name = None
        self._population_files = []
        self._constant_temporal_profile = None

        #set by read_pops
        self._populations = []  #list of populations

        #set by go_universes or load_surveys, changed by process_surveys
        self._surveys_path = None
        self._surveys_base_file_name = None
        self._survey_files = []

    @property
    def n_universes(self):
        return self._n_universes

    def _go_pops(
        self,
        param_file: str,
        pops_dir: str,
        constant_temporal_profile: bool,
        base_file_name: str = "pop",
        client: Client = None,
        seed: int = 1234,
        catalog_selec: bool = False,
        hard_flux_selec: bool = False,
        hard_flux_lim: float = 1e-7 #erg cm^-2 s^-1
        ):

        """Generate populations of GRBs

        :param param_file: path to file determining redshift and luminosity distribution parameters
        :type param_file: str
        :param pops_dir: path of populations
        :type pops_dir: str
        :param constant_temporal_profile: shape of temporal profile (True: constant, False: Pulse shape)
        :type constant_temporal_profile: bool
        :param base_file_name: prefix for created popsynth files, defaults to "pop"
        :type base_file_name: str, optional
        :param client: Dask client for parallelization, defaults None
        :type client: Client, optional
        :param seed: seed for random sampling, defaults to 1234
        :type seed: int, optional
        :param catalog_selec: if true, select GRBs coinciding with location of LV galaxy, defaults to False
        :type catalog_selec: bool, optional
        :param hard_flux_selec: if true, apply hard flux selection, defaults to False
        :type hard_flux_selec: bool, optional
        :param hard_flux_lim: lower limit hard flux selection, defaults to 1e-7#ergcm^-2s^-1
        :type hard_flux_lim: float, optional
        :param internal_parallelization: if true, compute GRBs in parallel WITHIN one population
        :type internal_parallelization: bool, optional
        """

        self._pops_computed = True

        self._pops_base_file_name = base_file_name

        self._pops_dir: Path = Path(pops_dir)

        self._constant_temporal_profile = constant_temporal_profile

        self._pops_dir.mkdir(parents=True, exist_ok=True)

        #read parameter file specifying distributions
        p: Path = Path(get_path_of_data_file(param_file))

        with p.open("r") as f:

            setup = yaml.load(f, Loader=yaml.SafeLoader)

        setup["catalog_selec"] = catalog_selec
        setup["hard_flux_selec"] = hard_flux_selec
        setup["hard_flux_lim"] = hard_flux_lim

        #function for one simulated population with params set in file p
        def sim_one_population(i,client=None):
            setup["seed"] = int(seed + i * 10)
            gp = GRBPop.from_dict(setup)

            gp.engage()

            gp.population.writeto(self._pops_dir / f"{self._pops_base_file_name}_{int(seed + i *10)}.h5")

        logger.info(f'Compute {self.n_universes} populations')

        if client is not None:

            #compute individual populations in parallel
            iteration = [i for i in range(0,self._n_universes)]

            futures = client.map(sim_one_population, iteration)
            res = client.gather(futures)

            del futures
            del res

        else:
            #serial
            sims = [sim_one_population(i) for i in range(self._n_universes)]

        self._population_files = list(self._pops_dir.glob(f"{self._pops_base_file_name}*.h5"))

        if len(self._population_files) == 0:
            logger.error('No populations were computed.')
            raise RuntimeError()


    def _load_pops(
        self,
        pop_sim_path: str,
        constant_temporal_profile: bool,
        pop_base_file_name: str = 'pop',
        ):
        """Set class parameters if pops were already computed

        :param pop_sim_path: Path of simulations
        :type pop_sim_path: str
        :param constant_temporal_profile: shape of temporal profile (True: constant, False: Pulse shape)
        :type constant_temporal_profile: bool
        :param pop_base_file_name: base file name of populations, defaults to 'pop'
        :type pop_base_file_name: str, optional
        """

        self._pops_dir = Path(pop_sim_path)

        self._pops_base_file_name = pop_base_file_name

        self._constant_temporal_profile = constant_temporal_profile

        self._population_files = list(self._pops_dir.glob(f"{self._pops_base_file_name}*.h5"))

        if len(self._population_files) == 0:
            raise Exception("No populations were found. Check path, and base file name.")

        self._pops_computed = True

    @property
    def population_files(self) -> List[str]:
        if len(self._population_files) == 0:
            raise Exception("No populations found. Load populations first (load_pops) or compute them (go_pops)")
        else:    
            return self._population_files

    
    def _read_pops(self):
        self._population_files = list(self._pops_dir.glob(f"{self._pops_base_file_name}*.h5"))

        if self._pops_computed:
            for f in tqdm(self._population_files, desc="loading populations"):

                self._populations.append(Population.from_file(f))

        else:
            raise Exception("No populations found. Load populations first (load_pops) or compute them (go_pops)")

    def _go_universes(
        self,
        surveys_path = None,
        surveys_base_file_name: str = 'survey',
        client: Client = None,
        internal_parallelization: bool = False
        ):
        """Compute GBM GRB data for all GRB universes given by different populations

        :param surveys_path: path in which survey is saved, defaults to None
        :type surveys_path: _type_, optional
        :param surveys_base_file_name: base name of generated survey h5 output file, defaults to 'survey'
        :type surveys_base_file_name: str, optional
        :param client: dask client for parallel computation, defaults to None
        :type client: Client, optional
        :param internal_parallelization: if True - compute GRB data within a population in parallel, defaults to False
                                        makes sense if n_sims (number simulations) << n_grbs (GRBs in one population)
        :type internal_parallelization: bool, optional
        :raises RuntimeError: _description_
        :return: _description_
        :rtype: _type_
        """

        self._surveys_base_file_name = surveys_base_file_name

        if surveys_path is None:
            # as default, save simulated GRB files in folder containing population files
            self._surveys_path = self._pops_dir
        else:
            # if save_path was specified, use this folder
            self._surveys_path = Path(surveys_path)
            # if path does not exist yet, create it
            self._surveys_path.mkdir(parents=True, exist_ok=True)

        if self._constant_temporal_profile == True:

            Universe_class = GBM_CPL_Constant_Universe

        else:

            Universe_class = GBM_CPL_Universe
        
        def sim_one_universe(i,client=None):

            pop_path_i = self.population_files[i]
            # take population stem name as name for folder of simulated GRBs
            save_path_stem_i = pop_path_i.stem.strip(self._pops_base_file_name)
            save_path_i = self._surveys_path / save_path_stem_i
            #create new folder if non-existing yet
            save_path_i.mkdir(parents=True, exist_ok=True)

            universe = Universe_class(self.population_files[i],save_path=save_path_i)
            universe.go(client)
            #save as non-processed survey
            universe.save(str(self._surveys_path / f'{surveys_base_file_name}{save_path_stem_i}.h5'))
            return i

        logger.info('Go Universes')
        
        if client is not None:

            if internal_parallelization:
                #compute GRBs in parallel within one universe
                logger.info('Use internal parallelization')
                sims = [sim_one_universe(i,client) for i in range(self._n_universes)]
            else:
                logger.info('Use external parallelization')
                iteration = [i for i in range(0,self._n_universes)]

                futures = client.map(sim_one_universe, iteration)
                res = client.gather(futures)

                del futures
                del res

        else:
            #serial
            logger.info('Use no parallelization')
            sims = [sim_one_universe(i) for i in range(self._n_universes)]

        self._survey_files = list(self._surveys_path.glob(f"{self._surveys_base_file_name}*.h5"))

        if len(self._survey_files) == 0:
            logger.error('No surveys computed. Check directory names again.')
            raise RuntimeError()

        self._universes_computed = True
    
    @property
    def survey_files(self) -> List[str]:

        if len(self._survey_files) == 0:
            logger.error('No surveys found. Load surveys first (load_surveys) or compute them (go_universe)')
            raise RuntimeError()
        else:    
            return self._survey_files

    def _load_surveys(
        self,
        surveys_path: str,
        surveys_base_file_name: str = 'survey'
        ):
        self._surveys_path = surveys_path
        self._surveys_base_file_name = surveys_base_file_name
        self._survey_files = list()

        self._survey_files = list(self._surveys_path.glob(f"{self._surveys_base_file_name}*.h5"))

        if len(self._survey_files) == 0:
            logger.error("No surveys found. Check directories again.")
            raise RuntimeError()

        self._universes_computed = True

    def _process_surveys(
        self,
        client: Client = None,
        threshold_trigger: float = 4.5,
        internal_parallelization: bool = False
    ):
        def process_one_survey(i,client=None):
            survey = Survey.from_file(self.survey_files[i])
            survey.process(GBMTrigger,threshold=threshold_trigger,client=client)
            survey.write(self.survey_files[i])
        
        logger.info('Process survey')

        if client is not None:

            if internal_parallelization:
                logger.info('Use internal parallelization')
                sims = [process_one_survey(i,client=client) for i in range(self._n_universes)]
            
            else:
                logger.info('Use external parallelization')
                iteration = [i for i in range(0,self._n_universes)]

                futures = client.map(process_one_survey, iteration)
                res = client.gather(futures)

                del futures
                del res

        else:
            logger.info('No parallelization')

            sims = [process_one_survey(i) for i in range(self._n_universes)]

        self._surveys_processed = True 

    def _write_summary_file(
        self,
        ):
            #TODO
            pass

    def go(
        self,
        param_file: str,
        pops_dir: str,
        constant_temporal_profile: bool,
        pop_base_file_name: str = "pop",
        client: Client = None,
        seed: int = 1234,
        catalog_selec: bool = False,
        hard_flux_selec: bool = False,
        hard_flux_lim: float = 1e-7, #erg cm^-2 s^-1
        surveys_path = None,
        surveys_base_file_name: str = 'survey',
        threshold_trigger: float = 4.5,
        internal_parallelization: bool = False
        ):

        self._go_pops(
            param_file=param_file,
            pops_dir=pops_dir,
            constant_temporal_profile=constant_temporal_profile,
            base_file_name=pop_base_file_name,
            client=client,
            seed=seed,
            catalog_selec=catalog_selec,
            hard_flux_selec=hard_flux_selec,
            hard_flux_lim=hard_flux_lim
            )
        self._go_universes(
            surveys_path=surveys_path,
            surveys_base_file_name=surveys_base_file_name,
            client=client,
            internal_parallelization=internal_parallelization
            )
        self._process_surveys(
            client=client,
            threshold_trigger=threshold_trigger,
            internal_parallelization=internal_parallelization)

#if n_cpus > n_sims: ?
    
#TODO: test
#TODO: write pytests
#TODO: write summary summary.h5 file containing final detected GRB file directories for every universe (throw out universes that are not containing GRBs)
#TODO: Use files in summary file to analyse GRBs - New Restores simulation class?
#TODO: for pops without catalog_selection, apply internal parallelization when computing GBM GRB data
#TODO: Add condition if popsynth selected no GRBs
#TODO: add plotting functions from underneath

    


#class Restored_GRBPops(object):
#
#    def __init__(
#        self, 
#        pop_sim_path: str,
#        constant_profile: bool,
#        pop_base_file_name: str = 'pop',
#        universe_sim_path: str = None
#        ):
#
#        # path of populations
#        self._pop_sim_path = Path(pop_sim_path)
#
#        # list of all populations
#        self._populations = []
#
#        self._pop_base_file_name = pop_base_file_name
#
#        if self._pop_sim_path.is_dir(): 
#
#            self._pop_sim_folder = self._pop_sim_path
#            # list of all population files
#            self._population_files = list(self._pop_sim_folder.glob(f"{self._pop_base_file_name}*.h5"))
#
#        else:
#            #e.g. if str pop_sim_path includes base file name 
#            self._pop_sim_folder = self._pop_sim_path.parent
#            # list of all population files
#            self._population_files = list(self._pop_sim_folder.glob(f"{self._pop_sim_path.name}*.h5"))
#
#        self._n_sims = len(self._population_files)
#
#        if self._n_sims == 0:
#
#            raise Exception("No population files found in specified path. Check base file name of population files. If not 'pop', specify pop_sim_path, e.g. dir_to_pops/pops_base_file_name")
#
#        self._universe_sim_path = universe_sim_path
#        
#        self._isread = False
#
#        #self._catalog = LocalVolume.from_lv_catalog()
#
#        self._universes = []
#
#        self._constant_profile = constant_profile
#
#    def play_god_universes(
#        self,
#        save_path: str = None,
#        universe_base_file_name: str = 'universe',
#        client: Client = None
#    ):
#
#        if save_path is None:
#            # as default, save simulated GRB files in folder containing population files
#            self._universe_sim_path = self._pop_sim_folder
#        else:
#            # if save_path was specified, use this folder
#            self._universe_sim_path = Path(save_path)
#            # if path does not exist yet, create it
#            self._universe_sim_path.mkdir(parents=True, exist_ok=True)
#
#        if self._constant_profile == True:
#
#            Universe_class = GBM_CPL_Constant_Universe
#
#        else:
#
#            Universe_class = GBM_CPL_Universe
#        
#        def sim_one_universe(i):
#
#            pop_path_i = self._population_files[i]
#            #print(pop_path_i)
#            #return pop_path_i
#            # take population stem name as name for folder of simulated GRBs
#            save_path_stem_i = pop_path_i.stem.strip(self._pop_base_file_name)
#            save_path_i = self._universe_sim_path / save_path_stem_i
#            #create new folder if non-existing yet
#            save_path_i.mkdir(parents=True, exist_ok=True)
#
#            universe = Universe_class(self._population_files[i],save_path=save_path_i)
#            universe.go()
#            universe.save(str(self._universe_sim_path / f'{universe_base_file_name}{save_path_stem_i}.h5'))
#            return i
#        
#        if client is not None:
#            iteration = [i for i in range(0,self._n_sims)]
#
#            futures = client.map(sim_one_universe, iteration)
#            res = client.gather(futures)
#
#            del futures
#            del res
#
#        else:
#
#            sims = [sim_one_universe(i) for i in range(self._n_sims)]
#
#
#    def play_god_surveys(
#        self,
#        universes_path: str = None,
#        save_path: str = None,
#        universe_base_file_name: str = 'universe',
#        client: Client = None
#    ):
#        if universes_path is None:
#            universes_path = self._pop_sim_folder
#        
#        universe_files = list(universes_path.glob(f"{universe_base_file_name}*.h5"))
#
#        if len(universe_files) == 0:
#
#            raise Exception("No universe files found in specified path. Check also base file name of universe files. If not 'universe', specify universe_base_file_name")
#
#        if save_path is None:
#            # as default, save simulated survey files in folder containing population files
#            save_p = self._pop_sim_folder
#        else:
#            # if save_path was specified, use this folder
#            save_p = Path(save_path)
#            save_p.mkdir(parents=True, exist_ok=True)
#
#        def sim_one_survey(i):
#
#            universe_file_i = Path(universe_files[i])
#
#            survey = Survey.from_file(universe_file_i)
#            survey.process(GBMTrigger, threshold=4.5)
#            survey.write(f"survey{universe_file_i.stem.strip(universe_base_file_name)}.h5")
#            return i
#
#        if client is not None:
#            print('Hello')
#            iteration = [i for i in range(0,self._n_sims)]
#
#            futures = client.map(sim_one_survey, iteration)
#            res = client.gather(futures)
#
#            del futures
#            del res
#
#        else:
#
#            sims = [sim_one_survey(i) for i in range(self._n_sims)]
#
#    def _read_pops(self):
#
#        for f in tqdm(self._population_files, desc="loading populations"):
#
#            self._populations.append(Population.from_file(f))
#
#        self._isread = True
#
#    def _count_galaxies(self):
#
#        if self._isread == False:
#            self._read_pops()
#
#        self._hit_galaxy_names = collections.Counter()
#
#        for pop in tqdm(self._populations, desc="counting galaxies"):
#            self._catalog.read_population(pop)
#            for galaxy in self._catalog.selected_galaxies:
#
#                self._hit_galaxy_names.update([galaxy.name])
#
#    @property
#    def populations(self) -> List[Population]:
#
#        if self._isread == False:
#            self._read_pops()
#
#        return self._populations
#
#    @property
#    def population_files(self) -> List[Population]:
#
#        return self._population_files
#    
#    @property
#    def fractions(self):
#
#        if self._isread == False:
#            self._read_pops()
#
#        return np.array([population.n_detections/population.n_objects for population in self._populations])
#
#    @property
#    def n_detections(self):
#
#        if self._isread == False:
#            self._read_pops()
#
#        return np.array([population.n_detections for population in self._populations])
#
#    @property
#    def n_grbs(self):
#        
#        if self._isread == False:
#            self._read_pops()
#
#        return np.array([population.n_objects for population in self._populations])
#
#    def hist_galaxies(self, n=10, exclude=[]):
#
#        if self._isread == False:
#            self._read_pops()
#
#        fig, ax = plt.subplots()
#
#        hits = self._hit_galaxy_names.most_common(n)
#        names = [x[0] for x in hits if x[0] not in exclude]
#        counts = [x[1] for x in hits if x[0] not in exclude]
#
#        x = np.arange(len(names))  # the label locations
#        width = 0.35  # the width of the bars
#
#        ax.barh(x, counts, width)
#
#        ax.set_xlabel('Number of spatial coincidences')
#
#        ax.invert_yaxis()
#        ax.set_yticks(x)
#        ax.set_yticklabels(names)
#
#        plt.tight_layout()
#
#        return fig
#
#    def hist_n_GRBs(self, ax=None, **kwargs):
#
#        if self._isread == False:
#            self._read_pops()
#
#        if ax is None:
#
#            fig, ax = plt.subplots()
#            ax.set_xlabel('Number GRBs')
#
#        else:
#            
#            fig = ax.get_figure()
#
#        ax.hist(self.n_grbs, **kwargs)
#
#        ax.set_xlabel('Number GRBs')
#        plt.tight_layout()
#
#        return fig
#
#    def hist_n_detections(self, ax=None,**kwargs):
#
#        if self._isread == False:
#            self._read_pops()
#
#        if ax is None:
#
#            fig, ax = plt.subplots()
#
#            ax.set_xlabel('Number GRBs')
#
#        else:
#            
#            fig = ax.get_figure()
#
#        ax.hist(self.n_detections, **kwargs)
#
#        plt.tight_layout()
#
#        return fig