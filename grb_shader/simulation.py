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

#TODO: write summary summary.h5 file containing final detected GRB file directories for every universe (throw out universes that are not containing GRBs)
#TODO: Use files in summary file to analyse GRBs - New Restores simulation class?
#TODO: for pops without catalog_selection, apply internal parallelization when computing GBM GRB data --> Done but too high memory usage
#TODO: add plotting functions from underneath

class GodMultiverse(object):

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
            [sim_one_population(i) for i in tqdm(range(self._n_universes), 'Simulate populations')]

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
            #del universe

        logger.info('Go Universes')
        
        if client is not None:

            if internal_parallelization:
                #compute GRBs in parallel within one universe
                logger.info('Use internal parallelization')
                for i in tqdm(range(self._n_universes),desc='Go universe - internally parallel'):
                    sim_one_universe(i=i,client=client)
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
            sims = [sim_one_universe(i) for i in tqdm(range(self._n_universes),desc='Go universe - serial')]

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
        self._surveys_path = Path(surveys_path)
        self._surveys_base_file_name = surveys_base_file_name

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

        logger.info('Process surveys')

        def process_one_survey(i,client=None,serial=True):
            survey = Survey.from_file(self.survey_files[i])
            survey.process(GBMTrigger,threshold=threshold_trigger,client=client,serial=serial)
            survey.write(self.survey_files[i])
            del survey
        

        if client is not None:

            if internal_parallelization:
                logger.info('Use internal parallelization')

                for i in tqdm(range(self._n_universes),desc='Process Surveys - internalls parallel'):
                    process_one_survey(i=i,client=client,serial=False)
            
            else:
                logger.info('Use external parallelization')
                iteration = [i for i in range(0,self._n_universes)]

                futures = client.map(process_one_survey, iteration)
                res = client.gather(futures)

                del futures
                del res

        else:
            logger.info('No parallelization')

            sims = [process_one_survey(i) for i in tqdm(range(self._n_universes),desc='Process Surveys - serial')]

        self._surveys_processed = True 

    def process_surveys(
        self,
        surveys_path: str,
        client: Client = None,
        surveys_base_file_name: str = 'survey',
        threshold_trigger: float = 4.5,
        internal_parallelization: bool = False
    ):
        self._load_surveys(
            surveys_path=surveys_path, 
            surveys_base_file_name= surveys_base_file_name
        )

        self._process_surveys(
            client=client,
            threshold_trigger=threshold_trigger,
            internal_parallelization=internal_parallelization)

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

class RestoredMultiverse(object):

    def __init__(
        self,
        path_survey_files: str,
        survey_base_file_name: str = 'survey',
        pop_base_file_name: str = 'pop',
        path_pops_files: str = None,
    ):
        self._survey_path = Path(path_survey_files)

        self._surveys = []
        self._survey_files = sorted(list(self._survey_path.glob(f"{survey_base_file_name}*.h5")))
        if len(self._survey_files) == 0:
            logger.warning('No Survey files found')

        self._n_hit_grbs = np.zeros(len(self.survey_files))

        for i, f in enumerate(tqdm(self.survey_files, desc="Loading surveys")):
            survey = Survey.from_file(f)
            self._surveys.append(survey)
            self._n_hit_grbs[i] = survey.n_grbs

        if path_pops_files is None:
            self._pops_path = self._survey_path
        else:
            self._pops_path = path_pops_files

        self._populations = []
        self._population_files = sorted(list(self._pops_path.glob(f"{pop_base_file_name}*.h5")))


        #There has to be always same number of population and survey files
        if len(self._survey_files) > 0:
            assert len(self._population_files) == len(self._survey_files)

        self._n_sim_grbs = np.zeros(len(self.population_files))
        self._n_detected_grbs = np.zeros(len(self.population_files))

        for i, f in enumerate(tqdm(self._population_files, desc="Loading populations")):
            pop = Population.from_file(f)
            self._populations.append(pop)
            self._n_sim_grbs[i] = pop.n_objects
            self._n_detected_grbs[i] = pop.n_detections

        self._catalog = LocalVolume.from_lv_catalog()

        self._galaxies_were_counted = False

    def _count_galaxies(self):

        self._galaxynames_hit = collections.Counter()
        self._galaxynames_hit_and_detected = collections.Counter()

        for i, pop in enumerate(tqdm(self._populations, desc="counting galaxies")):
            survey = self._surveys[i]
            self._catalog.read_population(pop)

            #Number of selected galaxies has to be same as number of simulated GRBs
            assert len(self._catalog.selected_galaxies) == survey.n_grbs

            for i,galaxy in enumerate(self._catalog.selected_galaxies):
                if survey.mask_detected_grbs[i]:
                    self._galaxynames_hit_and_detected.update([galaxy.name])

                self._galaxynames_hit.update([galaxy.name])

        self._galaxies_were_counted = True

    @property
    def n_hit_grbs(self):
        # if catalog selection: = number of GRBs that hit a galaxy
        # without: same as n_sim_grbs
        return self._n_hit_grbs

    @property
    def n_detected_grbs(self):
        # number of detected GRBs 
        # (only GRBs that hit galaxy if catalog selection )
        return self._n_detected_grbs

    @property
    def n_sim_grbs(self):
        # total number of GRBs in the population (from integrated spatial distribution)
        return self._n_sim_grbs

    @property
    def fractions_det(self):
        #fraction of detected GRBs 
        return self.n_detected_grbs/self.n_sim_grbs

    @property
    def fractions_det(self):
        #fraction of GRBs that hit galaxies
        return self.n_hit_grbs/self.n_sim_grbs

    @property
    def populations(self) -> List[Path]:
        return self._populations

    @property
    def survey_files(self) -> List[Path]:
        return self._survey_files

    @property
    def population_files(self) -> List[Population]:
        return self._population_files

    def hist_galaxies(self, n=10, width=0.6,exclude=[],ax=None,**kwargs):
        #histogram the n most often hit galaxies in all simulated universes

        if not self._galaxies_were_counted:
            self._count_galaxies()

        if ax is None:

            fig, ax = plt.subplots()

            ax.set_xlabel('Number of spatial coincidences')

        else:
            
            fig = ax.get_figure()
        
        hits = self._galaxynames_hit.most_common(n)
        names = [x[0] for x in hits if x[0] not in exclude]
        counts = [x[1] for x in hits if x[0] not in exclude]

        x = np.arange(len(names))  # the label locations

        ax.barh(x, counts, width,**kwargs)

        ax.invert_yaxis()
        ax.set_yticks(x)
        ax.set_yticklabels(names)
        plt.tight_layout()
        return fig
    
    def hist_galaxies_detected(self, n=10, exclude=[],width=0.6,ax=None,**kwargs):
        #histogram only detected GRBs
        #histogram the n most often hit galaxies in all simulated universes

        if not self._galaxies_were_counted:
            self._count_galaxies()

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_xlabel('Number of spatial coincidences')

        else:
            
            fig = ax.get_figure()
        
        hits = self._galaxynames_hit_and_detected.most_common(n)
        names = [x[0] for x in hits if x[0] not in exclude]
        counts = [x[1] for x in hits if x[0] not in exclude]

        x = np.arange(len(names))  # the label locations

        ax.barh(x, counts,width,**kwargs)
        
        ax.set_xlabel('Number of spatial coincidences')
        ax.invert_yaxis()
        ax.set_yticks(x)
        ax.set_yticklabels(names)
        plt.tight_layout()

        return fig

    def hist_n_sim_grbs(self, ax=None, **kwargs):

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()
        
        labels, counts = np.unique(self.n_sim_grbs, return_counts=True)
        ax.bar(labels,counts,**kwargs)

        ax.set_xlabel('Number GRBs')
        plt.tight_layout()

        return fig

    def hist_n_det_grbs(self, ax=None,width=0.55,**kwargs):

        if ax is None:

            fig, ax = plt.subplots()

            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()

        labels, counts = np.unique(self.n_detected_grbs, return_counts=True)
        ax.bar(labels,counts,width,**kwargs)

        plt.tight_layout()

        return fig

    def hist_n_hit_grbs(self, ax=None,**kwargs):

        if ax is None:

            fig, ax = plt.subplots()
            
            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()

        labels, counts = np.unique(self.n_hit_grbs, return_counts=True)
        ax.bar(labels,counts,**kwargs)

        plt.tight_layout()

        return fig
   