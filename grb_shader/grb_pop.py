from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import popsynth as ps
import yaml
from joblib import wrap_non_picklable_objects
from popsynth.selection_probability import UnitySelection

from .samplers import CatalogSelector, DurationSampler, TDecaySampler, EisoSampler, TriangleT90Sampler_Cor


class GRBPop(object):
    # generate population of GRBs

    def __init__(self, base_population: ps.PopulationSynth, observed_quantities: List[ps.AuxiliarySampler]):

        self._population: ps.Population = None

        self._population_gen: ps.PopulationSynth = base_population

        # add the observed qauntities

        for o in observed_quantities:

            self._population_gen.add_observed_quantity(o)

        # We are not going to select on flux
        # because cosmo GRB will do that for us

        flux_selector = UnitySelection()

        #flux_select = ps.HardFluxSelection()
        #flux_select.boundary = 1e-7 #erg cm^-2 s^-1
        #self._population_gen.set_flux_selection(flux_select)

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


class BaseProfile(object):

    def __init__(self, ep_profile=None, **params) -> None:

        self._qauntites: List = None

        if ep_profile is None:
            self._construct(**params)
        else:
            self._construct(ep_profile,**params)

    def _construct(self):

        pass

    @property
    def quantities(self) -> List:
        return self._quantities


class PulseProfile(BaseProfile):

    def _construct(self, t90_mu, t90_tau, t_rise_mu, t_rise_tau, t_rise_lower, t_rise_upper):
        """
        Samplers for pulse profile parameters (Norris, 2005)

        Sample T_90 from LogNormal(``t90_mu``, ``t90_tau``)
        Sample rise time t_rise from truncated Normal()

        :param t90_mu: mean of t90 
        :type t90_mu: float
        :param t90_tau: tau of t90
        :type t90_tau: float
        :param t_rise_mu: mean of t_rise
        :type t_rise_mu: float
        :param t_rise_tau: tau of t90 
        :type t_rise_tau: float
        :param t_rise_lower: lower limit for t_rise
        :type t_rise_lower: float
        :param t_rise_upper: upper limit for t_rise
        :type t_rise_upper: float
        """

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False)

        trise.lower = t_rise_lower
        trise.upper = t_rise_upper
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        t90 = ps.aux_samplers.LogNormalAuxSampler(name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        """
        ???
        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = 1.5
        tau.upper = 2.5
        tau.mu = 2
        """

        self._quantities = [duration, tdecay]


class ConstantProfile(BaseProfile):

    def _construct(self, t90_mu, t90_tau):
        """
        Samplers for constant temporal profile parameters
        """

        t90 = ps.aux_samplers.LogNormalAuxSampler(
            name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class TriangleProfile_Cor(BaseProfile):
    """
    Assume Ep-Eiso correlation and triangle shape of light curve
    """

    def _construct(self, ep_profile, q_a, m_a, tau):

        eiso = EisoSampler()
        eiso.set_secondary_sampler(ep_profile)

        eiso.q_a = q_a
        eiso.m_a = m_a
        eiso.tau = tau

        t90 = TriangleT90Sampler_Cor()

        t90.set_secondary_sampler(eiso)

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class Log10normalEp(BaseProfile):

    def _construct(self, mu, tau):
        """
        Sample Ep from Log10Normal(mu, tau)
        """
        ep = ps.aux_samplers.Log10NormalAuxSampler(
            name="ep", observed=False)

        ep.mu = np.log10(mu)
        ep.tau = tau

        self._quantities = [ep]

class LognormalEp(BaseProfile):

    def _construct(self, mu, tau):
        """
        Sample Ep from LogNormal(mu, tau)
        """
        ep = ps.aux_samplers.LogNormalAuxSampler(
            name="ep", observed=False)

        ep.mu = np.log(mu)
        ep.tau = tau

        self._quantities = [ep]

class BplEp(BaseProfile):

    def _construct(self, Epmin, alpha, Epbreak, beta, Epmax):
        """
        Sample Ep from Broken power law distribution
        """
        ep = ps.aux_samplers.BrokenPowerLawAuxSampler(
            name="ep", observed=False)

        ep.xmin = Epmin
        ep.alpha = alpha
        ep.xbreak = Epbreak
        ep.beta = beta
        ep.xmax = Epmax


        self._quantities = [ep]


_base_gen_lookup = dict(pareto_sfr=ps.populations.ParetoSFRPopulation,
                        bpl_sfr=ps.populations.BPLSFRPopulation)

_ep_lookup = dict(log10normal=Log10normalEp,
                    lognormal=LognormalEp,
                    bpl=BplEp)

_temporal_lookup = dict(constant=ConstantProfile,
                        pulse=PulseProfile,
                        triangle_cor=TriangleProfile_Cor)
