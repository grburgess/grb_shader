from pathlib import Path
from typing import Dict, List

import numpy as np
import popsynth as ps
import yaml
from popsynth.selection_probability import UnitySelection
from .catalog_selector import CatalogSelector
from .profiles.Ep_profiles import Log10normalEp, LognormalEp, BplEp
from .profiles.temporal_profiles import *

class GRBPop(object):

    def __init__(
        self, 
        base_population: ps.PopulationSynth, 
        observed_quantities: List[ps.AuxiliarySampler], 
        catalog_selec: bool = False,
        hard_flux_selec: bool = False,
        hard_flux_lim: float = 1e-7 #erg cm^-2 s^-1
        ):
        """Generate population of GRBs

        :param base_population: base GRB population (ParetoSFR or BPLSFR)
        :type base_population: ps.PopulationSynth
        :param observed_quantities: parameters that are sampled
        :type observed_quantities: List[ps.AuxiliarySampler]
        :param catalog_selec: if true, select GRBs coinciding with location of LV galaxy, defaults to False
        :type catalog_selec: bool, optional
        :param hard_flux_selec: if true, set hard flux limit for detection, defaults to False
        :type hard_flux_selec: bool, optional
        :param hard_flux_lim: flux limit, defaults to 1e-7 #ergcm^-2s^-1
        :type hard_flux_lim: float, optional
        """

        self._population: ps.Population = None

        self._population_gen: ps.PopulationSynth = base_population

        # add the observed qauntities

        for o in observed_quantities:

            self._population_gen.add_observed_quantity(o)

        if hard_flux_selec == True:
            # Add possible hard flux selector from GBM for testing
            flux_select = ps.HardFluxSelection()
            flux_select.boundary =  hard_flux_lim 
            self._population_gen.set_flux_selection(flux_select)
        else: 
            # We are not going to select on flux
            # because cosmo GRB will do that for us
            flux_selector = UnitySelection()
            self._population_gen.set_flux_selection(flux_selector)

        if catalog_selec == True:
            # build the catalog selections
            self._catalog_selector: CatalogSelector = CatalogSelector()
            self._population_gen.add_spatial_selector(self._catalog_selector)

        # angle_sampler = AngleSampler()
        # angle_sampler.set_angles(self._catalog_selector.catalog.angles)
        # self._population_gen.add_observed_quantity(angle_sampler)

    def engage(self) -> None:
        """
        Sample from all distributions and create a 'Population' object
        """
        self._population = self._population_gen.draw_survey()

    @property
    def population(self) -> ps.Population:
        return self._population

    @property
    def population_generator(self) -> ps.PopulationSynth:
        return self._population_gen

    @property
    def catalog_selector(self) -> CatalogSelector:
        return self._catalog_selector

    @classmethod
    def from_yaml(cls, file_name) -> "GRBPop":
        """
        Read parameters for redshift and luminosity distributions 
        from file_name.yml and save in dict
        Generate GRBPop class from created dictionary

        :file_name: string, file name and directory to yaml data file
        """
        p: Path = Path(file_name)

        with p.open("r") as f:

            inputs: Dict = yaml.load(stream=f, Loader=yaml.SafeLoader)

        return cls.from_dict(inputs)

    @classmethod
    def from_dict(cls, inputs: Dict) -> "GRBPop":
        """
        Construct population of GRBs from dictionary

        :param inputs: Dictionary containing parameter names and values
        :type inputs: Dict
        :return: GRB population object
        :rtype: GRBPop
        """

        seed = inputs["seed"]
        catalog_selec = inputs["catalog_selec"]
        hard_flux_selec = inputs["hard_flux_selec"]
        hard_flux_lim = inputs["hard_flux_lim"]


        #look up in dict which is defined below class if constant or pulse
        #set corresponding samplers with defined parameters from yml file
        base_gen = _base_gen_lookup[inputs["generator"]["flavor"]](seed=seed,
                                                                   **inputs["generator"]["parameters"])

        # set distribution for peak energy ep
        ep_profile = _ep_lookup[inputs["spectral"]["flavor"]](
            **inputs["spectral"]["ep"])

        # set alpha distribution for Band function spectrum
        alpha = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="alpha", observed=False)

        alpha.lower = inputs["spectral"]["alpha"]["lower"]
        alpha.upper = inputs["spectral"]["alpha"]["upper"]
        alpha.mu = inputs["spectral"]["alpha"]["mu"]
        alpha.tau = inputs["spectral"]["alpha"]["tau"]

        if inputs["temporal profile"]["flavor"] == 'triangle_cor':
            temporal_profile = _temporal_lookup[inputs["temporal profile"]["flavor"]](ep_profile=ep_profile.quantities[0],
            **inputs["temporal profile"]["parameters"])
        else:
            temporal_profile = _temporal_lookup[inputs["temporal profile"]["flavor"]](**inputs["temporal profile"]["parameters"])
        
        observed_quantities = [alpha]
        observed_quantities.extend(temporal_profile.quantities)

        #if Ep is not sampled as secondary, add it as observed quantity to sample it too
        if ep_profile.quantities[0].is_secondary == False:
             observed_quantities.extend(ep_profile.quantities)

        return cls(base_gen, observed_quantities, catalog_selec, hard_flux_selec, hard_flux_lim)


_base_gen_lookup = dict(pareto_sfr=ps.populations.ParetoSFRPopulation,
                        bpl_sfr=ps.populations.BPLSFRPopulation)

_ep_lookup = dict(log10normal=Log10normalEp,
                    lognormal=LognormalEp,
                    bpl=BplEp)

_temporal_lookup = dict(constant_lognormal=ConstantProfile_Lognormal,
                        constant_log10normal=ConstantProfile_Log10normal,
                        constant_lognormal_trunc=ConstantProfile_Lognormal_Trunc,
                        constant_log10normal_trunc=ConstantProfile_Log10normal_Trunc,
                        pulse_lognormal=PulseProfile_Lognormal,
                        pulse_log10normal=PulseProfile_Log10normal,
                        pulse_lognormal_trunc=PulseProfile_Lognormal_Trunc,
                        pulse_log10normal_trunc=PulseProfile_Log10normal_Trunc,
                        triangle_cor=TriangleProfile_Cor)