from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import popsynth as ps
import yaml
from popsynth.selection_probability import SpatialSelection, UnitySelection
from tqdm.auto import tqdm

from .samplers import CatalogSelector

_base_gen_lookup = dict(pareto_sfr=ps.populations.ParetoSFRPopulation)
_temporal_lookup = dict(constant=ConstantProfile,
                        pulse=PulseProfile

                        )


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

    def engage(self) -> None:

        self._population = self._population_gen.draw_survey(
            no_selection=False, boundary=1e-2)

    @property
    def population(self) -> ps.Population:
        return self._population

    @property
    def population_generator(self) -> ps.PopulationSynth:
        return self._population_gen

    @property
    def catalog_selector(self) -> CatalogSelector:
        return self._catalog_selector

    @classmethod:
    def from_yaml(cls, file_name) -> "GRBPop":

        file_name: Path = Path(file_name)

        with file_name.open("r") as f:

            inputs: Dict = yaml.load(stream=f, Loader=yaml.SafeLoader)

        base_gen = _base_gen_look_up[inputs["generator"]["flavor"]](
            **inputs["generator"]["parameters"])

        ep = ps.aux_samplers.LogNormalAuxSampler(name="log_ep", observed=False)

        # set the ep

        ep.mu = inputs["spectral"]["ep"]["mu"]
        ep.tau = inputs["spectral"]["ep"]["tau"]

        # set the alpha

        alpha = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="alpha", observed=False)

        alpha.lower = -1.5
        alpha.upper = 0.1
        alpha.mu = inputs["spectral"]["alpha"]["mu"]
        alpha.tau = inputs["spectral"]["alpha"]["tau"]

        temporal_profile = _temporal_look_up[inputs["temporal profile"]["flavor"]](
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

        self._qauntites = [duration, tdecay]


class ContantProfile(TemporalProfile):

    def _construct(self, log_t90_mu, lof_t90_tau):

        t90 = ps.aux_samplers.Log10NormalAuxSampler(
            name="t90", observed=False)

        t90.mu = log_t90_mu
        t90.tau = log_t90_tau

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._qauntites = [duration]
