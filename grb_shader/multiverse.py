import collections
from pathlib import Path
from typing import List
from wsgiref.util import setup_testing_defaults
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import yaml

from dask.distributed import Client

from popsynth import Population, silence_progress_bars, silence_warnings

from tqdm.auto import tqdm

from .grb_pop import GRBPop
from .catalog import LocalVolume
from .utils.package_data import get_path_of_data_file
from .utils.logging import setup_log
from .plotting.plotting_functions import array_to_cmap
from .plotting.minor_symlog_locator import MinorSymLogLocator

logger = setup_log(__name__)

import cosmogrb
from cosmogrb.instruments.gbm import GBM_CPL_Universe, GBM_CPL_Constant_Universe
from cosmogrb.universe.survey import Survey
from cosmogrb.instruments.gbm.gbm_trigger import GBMTrigger




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
        hard_flux_lim: float = 1e-7, #erg cm^-2 s^-1
        with_unc: bool = False,
        unc_circular_angle:float=1., #deg
        n_samp:int=100
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
        setup["with_unc"] = with_unc
        setup["unc_circular_angle"] = unc_circular_angle
        setup["n_samp"] = n_samp

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

    def reload_pops(
        self,
        pop_sim_path: str,
        constant_temporal_profile: bool,
        pop_base_file_name: str = 'pop'
        ):

        self._load_pops(pop_sim_path, constant_temporal_profile, pop_base_file_name)
        
        self._read_pops()


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
            logger.info('Simulate GBM_CPL_Constant_Universe')

            Universe_class = GBM_CPL_Constant_Universe

        else:
            logger.info('Simulate GBM_CPL_Universe')

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
        
        #if client is not None:

        if internal_parallelization:
            
            #compute GRBs in parallel within one universe
            logger.info('Use internal parallelization')

            for i in tqdm(range(self._n_universes),desc='Go universe - internally parallel'):

                sim_one_universe(i=i,client=client)
            
            #external parallelization gives memory error
            #else:
#
            #    logger.info('Use external parallelization')
#
            #    iteration = [i for i in range(0,self._n_universes)]
#
            #    futures = client.map(sim_one_universe, iteration)
            #    res = client.gather(futures)
#
            #    del futures
            #    del res

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
        # add survey file paths

        self._surveys_path = Path(surveys_path)
        self._surveys_base_file_name = surveys_base_file_name

        self._survey_files = list(self._surveys_path.glob(f"{self._surveys_base_file_name}*.h5"))

        if len(self._survey_files) == 0:
            logger.error(f"No surveys found in {self._surveys_path.glob}. Check directories again.")
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

                for i in tqdm(range(self._n_universes),desc='Process Surveys - internally parallel'):
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
        surveys_path: str=None,
        client: Client = None,
        surveys_base_file_name: str = 'survey',
        threshold_trigger: float = 4.5,
        internal_parallelization: bool = False,
        save_det_grbs_as_fits: bool = False
    ):
        if self._surveys_path is None:
            self._load_surveys(
                surveys_path=surveys_path, 
                surveys_base_file_name= surveys_base_file_name
            )

        self._process_surveys(
            client=client,
            threshold_trigger=threshold_trigger,
            internal_parallelization=internal_parallelization)

        if save_det_grbs_as_fits:
            self._save_detected_grbs_as_fits(
                client=client,
                internal_parallelization=internal_parallelization
            )

    def _save_detected_grbs_as_fits(self,client=None,internal_parallelization=False):

        logger.info('Convert detected GRB files to fits files')

        def convert_one_survey(i,client=None,serial=True):
            survey = Survey.from_file(self.survey_files[i])
            survey_name = self.survey_files[i].stem

            if not survey.is_processed:
                raise Exception('Survey is not processed yet')

            #convert detected GRBs to HDF5 format to allow import to threeml
            if survey.n_detected > 0:
                grb_file = Path(survey.files_detected_grbs[0])
                destination = grb_file.parent

                for j in tqdm(range(len(survey.files_detected_grbs)),desc=f"Saving GRB files of {survey_name} to fits"):
                    cosmogrb.grbsave_to_gbm_fits(survey.files_detected_grbs[j],destination=destination)

                del survey
            else:
                logger.info(f'No detected GRBs in {survey_name}')
        
        if client is not None:

            if internal_parallelization:
                logger.info('Use internal parallelization')

                for i in tqdm(range(self._n_universes),desc='Process Surveys - internally parallel'):
                    convert_one_survey(i=i,client=client,serial=False)
            
            else:
                logger.info('Use external parallelization')
                iteration = [i for i in range(0,self._n_universes)]

                futures = client.map(convert_one_survey, iteration)
                res = client.gather(futures)

                del futures
                del res

        else:
            logger.info('No parallelization')

            sims = [convert_one_survey(i) for i in tqdm(range(self._n_universes),desc='Process Surveys - serial')]
    
    def go_pops(
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
        with_unc: bool = False,
        unc_circular_angle:float=1., #deg
        n_samp:int=100
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
            hard_flux_lim=hard_flux_lim,
            with_unc=with_unc,
            unc_circular_angle=unc_circular_angle, #deg
            n_samp=n_samp
            )

    def go_universes(
        self,
        pops_dir: str=None,
        constant_temporal_profile: bool=None,
        surveys_path = None,
        surveys_base_file_name: str = 'survey',
        client: Client = None,
        internal_parallelization: bool = False,
        pop_base_file_name: str = "pop",

    ):
        if self._pops_dir is None:

            logger.info('load pops')

            self._load_pops(
            pop_sim_path=pops_dir,
            constant_temporal_profile= constant_temporal_profile,
            pop_base_file_name=pop_base_file_name,
            )

        self._surveys_path = surveys_path
        self._surveys_base_file_name = surveys_base_file_name

        logger.info('go universes')
        self._go_universes(
            surveys_path=self._surveys_path,
            surveys_base_file_name=self._surveys_base_file_name,
            client=client,
            internal_parallelization=internal_parallelization
            )

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
        with_unc: bool = False,
        unc_circular_angle:float=1., #deg
        n_samp:int=100,
        surveys_path = None,
        surveys_base_file_name: str = 'survey',
        threshold_trigger: float = 4.5,
        internal_parallelization: bool = False,
        save_det_grbs_as_fits: bool = True
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
            hard_flux_lim=hard_flux_lim,
            with_unc=with_unc,
            unc_circular_angle=unc_circular_angle, #deg
            n_samp=n_samp
            )
        
        #internal parallelization does not work yet -> problem with memory

        self._go_universes(
            surveys_path=surveys_path,
            surveys_base_file_name=surveys_base_file_name,
            client=client,
            internal_parallelization=False
            )

        if hard_flux_selec == False:
            self._process_surveys(
                client=client,
                threshold_trigger=threshold_trigger,
                internal_parallelization=internal_parallelization)
            
            if save_det_grbs_as_fits:
                self._save_detected_grbs_as_fits(
                    client=client,
                    internal_parallelization=internal_parallelization
                )
        else:
            logger.info('Do not use GBM trigger as hard flux selection was already applied as alternative')

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

        if path_pops_files is None:
            self._pops_path = self._survey_path
        else:
            self._pops_path = path_pops_files

        self._populations = []
        self._population_files = sorted(list(self._pops_path.glob(f"{pop_base_file_name}*.h5")))

        self._n_detected_grbs = np.zeros(len(self.population_files))
        for i, f in enumerate(tqdm(self.survey_files, desc="Loading surveys")):
            survey = Survey.from_file(f)
            self._surveys.append(survey)
            self._n_hit_grbs[i] = survey.n_grbs
            self._n_detected_grbs[i] = survey.n_detected


        #There has to be always same number of population and survey files
        if len(self._survey_files) > 0:
            assert len(self._population_files) == len(self._survey_files)

        self._n_sim_grbs = np.zeros(len(self.population_files))
        

        for i, f in enumerate(tqdm(self._population_files, desc="Loading populations")):
            pop = Population.from_file(f)
            self._populations.append(pop)
            self._n_sim_grbs[i] = pop.n_objects
            #self._n_detected_grbs[i] = pop.n_detections

        #load local volume galaxies catalog
        self._catalog = LocalVolume.from_lv_catalog()

        self._galaxies_were_counted = False
        
        self._count_galaxies()

    def _count_galaxies(self):

        self._galaxynames_hit = collections.Counter()
        self._galaxynames_hit_and_detected = collections.Counter()

        for i, pop in enumerate(tqdm(self._populations, desc="counting galaxies")):
            survey = self._surveys[i]
            self._catalog.read_population(pop,unc_angle=0.)

            #Number of selected galaxies has to be same as number of simulated GRBs
            assert len(self._catalog.selected_galaxies) == survey.n_grbs

            for i,galaxy in enumerate(self._catalog.selected_galaxies):
                if survey.mask_detected_grbs[i]:
                    self._galaxynames_hit_and_detected.update([galaxy[0].name])

                self._galaxynames_hit.update([galaxy[0].name])

        self._galaxies_were_counted = True
       
    @property 
    def galaxynames_hit(self):
        return self._galaxynames_hit   
    
    @property 
    def galaxynames_hit_and_detected(self):
        return self._galaxynames_hit_and_detected


    @property
    def n_universes(self):
        return len(self._survey_files)

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
    def fractions_hit(self):
        #fraction of GRBs that hit galaxies
        return self.n_hit_grbs/self.n_sim_grbs

    @property
    def populations(self) -> List[Population]:
        return self._populations
    
    @property
    def surveys(self) -> List[Survey]:
        return self._surveys

    @property
    def survey_files(self) -> List[Path]:
        return self._survey_files

    @property
    def population_files(self) -> List[Path]:
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
        ax.set_xscale('symlog',linthresh=1)
        ax.set_xlabel('# Spatial Coincidences')
        ax.xaxis.set_minor_locator(MinorSymLogLocator(1))
        ax.grid(which='major',axis='x')
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
        ax.set_xscale('symlog',linthresh=1)
        ax.set_xlabel('# Spatial Coincidences')
        ax.xaxis.set_minor_locator(MinorSymLogLocator(1))
        ax.grid(which='major',axis='x')
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

    def _plot_distance_area_n(self,n,distances,areas,uselog=True, cmap='viridis',ax=None,**kwargs):
        """
        Plot number of coincidences as function of distance to galaxy 
        and its angular area on the sky

        :param n: number of coincidences
        :type n: int
        :param distances: distance to galaxies
        :type distances: list
        :param areas: elliptic angular areas on sky
        :type areas: list
        :param uselog: use logarithmic scale for colors, defaults to True
        :type uselog: bool, optional
        :param cmap: color map, defaults to 'viridis'
        :type cmap: str, optional
        :param ax: Axis, defaults to None
        :type ax: optional
        """
        if ax is None:

            fig, ax = plt.subplots()
            
            ax.set_xlabel('Number GRBs')

        else:
            
            fig = ax.get_figure()
        
        _, colors = array_to_cmap(n, cmap=cmap, use_log=uselog)
    
        if uselog:
            norm = mpl.colors.LogNorm(vmin=min(n), vmax=max(n))
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            norm = mpl.colors.Normalize(vmin=min(n), vmax=max(n))

        #Add an axis for colorbar on right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax, label='Number')

        for i in range(len(distances)):
            ax.scatter(x=float(distances[i]), y=float(areas[i]),color=colors[i],s=30,alpha=0.9,**kwargs)

        ax.set_xlabel('Distance [Mpc]')
        ax.set_ylabel(r'Angular Area [rad$^2$]')

        return fig

    def plot_distance_area_n(self,uselog=True, cmap='viridis',ax=None,**kwargs):

        hit_gal_names = list(self._galaxynames_hit.keys())

        hit_gal_n = list(self._galaxynames_hit.values())

        hit_gal_dist = np.zeros(len(hit_gal_names))
        hit_gal_area = np.zeros(len(hit_gal_names))

        for i, name in enumerate(hit_gal_names):
            hit_gal_dist[i] = self._catalog.galaxies[name].distance
            hit_gal_area[i] = self._catalog.galaxies[name].area

        fig = self._plot_distance_area_n(hit_gal_n,hit_gal_dist,hit_gal_area,uselog=uselog, cmap=cmap,ax=ax,**kwargs)

        return fig 

    def plot_det_distance_area_n(self,uselog=True, cmap='viridis',ax=None):

        hit_gal_names_det = list(self._galaxynames_hit_and_detected.keys())

        hit_gal_n_det = list(self._galaxynames_hit_and_detected.values())

        hit_gal_dist_det = np.zeros(len(hit_gal_names_det))
        hit_gal_area_det = np.zeros(len(hit_gal_names_det))

        for i, name in enumerate(hit_gal_names_det):
            hit_gal_dist_det[i] = self._catalog.galaxies[name].distance
            hit_gal_area_det[i] = self._catalog.galaxies[name].area

        fig = self._plot_distance_area_n(hit_gal_n_det,hit_gal_dist_det,hit_gal_area_det,uselog=uselog, cmap=cmap,ax=ax)

        return fig 