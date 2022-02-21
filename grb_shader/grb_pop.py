from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import popsynth as ps
import yaml
from joblib import wrap_non_picklable_objects
from popsynth.selection_probability import UnitySelection

from .samplers import CatalogSelector, DurationSampler


class GRBPop(object):

    def __init__(self, base_population: ps.PopulationSynth, observed_quantities: List[ps.AuxiliarySampler]):

        self._population: ps.Population = None

        self._population_gen: ps.PopulationSynth = base_population

        # add the observed qauntities

        for o in observed_quantities:

            self._population_gen.add_observed_quantity(o)

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

        p: Path = Path(file_name)

        with p.open("r") as f:

            inputs: Dict = yaml.load(stream=f, Loader=yaml.SafeLoader)

        return cls.from_dict(inputs)

    @classmethod
    def from_dict(cls, inputs: Dict) -> "GRBPop":

        seed = inputs["seed"]

        base_gen = _base_gen_lookup[inputs["generator"]["flavor"]](seed=seed,
                                                                   **inputs["generator"]["parameters"])

        ep = ps.aux_samplers.Log10NormalAuxSampler(
            name="log_ep", observed=False)

        # set the ep

        ep.mu = np.log10(inputs["spectral"]["ep"]["mu"])
        ep.tau = inputs["spectral"]["ep"]["tau"]

        # set the alpha

        alpha = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="alpha", observed=False)

        alpha.lower = -1.5
        alpha.upper = 0.1
        alpha.mu = inputs["spectral"]["alpha"]["mu"]
        alpha.tau = inputs["spectral"]["alpha"]["tau"]

        temporal_profile = _temporal_lookup[inputs["temporal profile"]["flavor"]](
            **inputs["temporal profile"]["parameters"])

        observed_quantities = [ep, alpha]
        observed_quantities.extend(temporal_profile.quantities)

        return cls(base_gen, observed_quantities)


class TemporalProfile(object):

    def __init__(self, **params) -> None:

        self._qauntites: List = None

        self._construct(**params)

    def _construct(self):

        pass

    @property
    def quantities(self) -> List:
        return self._quantities


class PulseProfile(TemporalProfile):

    def _construct(self, t90_mu, t90_tau, t_rise_mu, t_rise_tau):

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False
                                                          )

        trise.lower = 0.01
        trise.upper = 5.0
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        t90 = ps.aux_samplers.LogNormalAuxSampler(
            name="log_t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = 1.5
        tau.upper = 2.5
        tau.mu = 2

        self._quantities = [duration, tdecay]


class ConstantProfile(TemporalProfile):

    def _construct(self, log_t90_mu, log_t90_tau):

        t90 = ps.aux_samplers.Log10NormalAuxSampler(
            name="t90", observed=False)

        t90.mu = log_t90_mu
        t90.tau = log_t90_tau

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]


_base_gen_lookup = dict(pareto_sfr=ps.populations.ParetoSFRPopulation,
                        bpl_sfr=ps.populations.BPLSFRPopulation
                        )

_temporal_lookup = dict(constant=ConstantProfile,
                        pulse=PulseProfile

                        )
