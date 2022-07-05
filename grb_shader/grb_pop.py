from pathlib import Path
from typing import Dict, List

import numpy as np
import popsynth as ps
import yaml
from popsynth.selection_probability import UnitySelection
from .catalog_selector import CatalogSelector
from .profiles.Ep_profiles import Log10normalEp, LognormalEp, BplEp
from .profiles.temporal_profiles import ConstantProfile_Lognormal, TriangleProfile_Cor, PulseProfile_Lognormal, ConstantProfile_Log10normal

class GRBPop(object):
    # generate population of GRBs

    def __init__(self, base_population: ps.PopulationSynth, observed_quantities: List[ps.AuxiliarySampler]):

        self._population: ps.Population = None

        self._population_gen: ps.PopulationSynth = base_population

        # add the observed qauntities

        for o in observed_quantities:

            self._population_gen.add_observed_quantity(o)

        # Add possible hard flux selector from GBM
        # flux_select = ps.HardFluxSelection()
        # flux_select.boundary = 1e-7 #erg cm^-2 s^-1
        # self._population_gen.set_flux_selection(flux_select)

        # We are not going to select on flux
        # because cosmo GRB will do that for us
        flux_selector = UnitySelection()
        self._population_gen.set_flux_selection(flux_selector)

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

        return cls(base_gen, observed_quantities)


_base_gen_lookup = dict(pareto_sfr=ps.populations.ParetoSFRPopulation,
                        bpl_sfr=ps.populations.BPLSFRPopulation)

_ep_lookup = dict(log10normal=Log10normalEp,
                    lognormal=LognormalEp,
                    bpl=BplEp)

_temporal_lookup = dict(constant_lognormal=ConstantProfile_Lognormal,
                        constant_log10normal=ConstantProfile_Log10normal,
                        pulse_lognormal=PulseProfile_Lognormal,
                        triangle_cor=TriangleProfile_Cor)
