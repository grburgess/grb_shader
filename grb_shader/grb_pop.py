from typing import List, Optional, Union

from tqdm.auto import tqdm
import numpy as np
import popsynth as ps
from popsynth.selection_probability import SpatialSelection, UnitySelection

from .catalog import Galaxy, LocalVolume


class TDecaySampler(ps.AuxiliarySampler):
    def __init__(self):
        """
        samples the decay of the of the pulse
        """

        super(TDecaySampler, self).__init__(
            name="tdecay", observed=False)

    def true_sampler(self, size):

        t90 = 10 ** self._secondary_samplers["log_t90"].true_values
        trise = self._secondary_samplers["trise"].true_values

        self._true_values = (
            1.0 / 50.0 * (10 * t90 + trise + np.sqrt(trise)
                          * np.sqrt(20 * t90 + trise))
        )


class DurationSampler(ps.AuxiliarySampler):
    def __init__(self):
        "samples how long the pulse last"

        super(DurationSampler, self).__init__(
            name="duration", observed=False
        )

    def true_sampler(self, size):

        t90 = 10 ** self._secondary_samplers["log_t90"].true_values

        self._true_values = 1.5 * t90


class CatalogSelector(SpatialSelection):

    def __init__(self) -> None:

        super(CatalogSelector, self).__init__(name="catalog_selector")

        self._catalog = LocalVolume.from_lv_catalog()
        self._selected_galaxies: List[Galaxy] = []

    def draw(self, size) -> None:

        # loop through the sky positions

        self._selection = np.zeros(size, dtype=bool)

        pbar = tqdm(total=size, desc="Scanning catalog")

        
        for i, (ra, dec), in enumerate(zip(self._spatial_distribution.ra, self._spatial_distribution.dec)):

            test, galaxy = self._catalog.intercepts_galaxy(ra, dec)

            if test:

                self._selection[i] = True

                self._selected_galaxies.append(galaxy)

            pbar.update(1)
                
    @property
    def selected_galaxies(self) -> List[Galaxy]:

        return self._selected_galaxies


class GRBPop(object):

    def __init__(self):

        r0_true = 5
        rise_true = 1.
        decay_true = 4.0
        peak_true = 1.5

        # the luminosity
        Lmin_true = 1e51
        alpha_true = 1.5
        r_max = 7.0

        pop_gen = ps.populations.ParetoSFRPopulation(
            r0=r0_true,
            rise=rise_true,
            decay=decay_true,
            peak=peak_true,
            Lmin=Lmin_true,
            alpha=alpha_true,
            r_max=r_max,
        )

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False
                                                          )

        trise.lower = 0.01
        trise.upper = 5.0
        trise.mu = 1
        trise.tau = 1.0

        t90 = ps.aux_samplers.LogNormalAuxSampler(
            name="log_t90", observed=False)

        t90.mu = 10
        t90.tau = 0.25

        ep = ps.aux_samplers.LogNormalAuxSampler(name="log_ep", observed=False)

        ep.mu = 300.0
        ep.tau = 0.5

        alpha = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="alpha", observed=False)

        alpha.lower = -1.5
        alpha.upper = 0.1
        alpha.mu = -1
        alpha.tau = 0.25

        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = 1.5
        tau.upper = 2.5
        tau.mu = 2
        tau.tau = 0.25

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        pop_gen.add_observed_quantity(ep)
        pop_gen.add_observed_quantity(tau)
        pop_gen.add_observed_quantity(alpha)
        pop_gen.add_observed_quantity(tdecay)
        pop_gen.add_observed_quantity(duration)

        flux_selector = UnitySelection()

        pop_gen.set_flux_selection(flux_selector)

        catalog_selector = CatalogSelector()

        pop_gen.add_spatial_selector(catalog_selector)

        pop = pop_gen.draw_survey(no_selection=False, boundary=1e-2)

        self.pop = pop

        self.pop_gen = pop_gen
