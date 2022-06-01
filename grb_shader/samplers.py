from typing import List, Optional, Union

import ligo.skymap.plot
import numpy as np
import scipy.stats as stats
import popsynth as ps
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from popsynth.selection_probability import SpatialSelection

from .catalog import Galaxy, LocalVolume


class TDecaySampler(ps.AuxiliarySampler):
    # inherits from ps.AuxiliarySampler
    _auxiliary_sampler_name = "TDecaySampler"
    def __init__(self):
        """
        samples the decay of the pulse
        """

        #call super class's __init__ method
        super(TDecaySampler, self).__init__(name="tdecay", observed=False)

    def true_sampler(self, size):

        t90 = 10 ** self._secondary_samplers["t90"].true_values
        trise = self._secondary_samplers["trise"].true_values

        self._true_values = (
            1.0 / 50.0 * (10 * t90 + trise + np.sqrt(trise)
                          * np.sqrt(20 * t90 + trise))
        )


class DurationSampler(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "DurationSampler"
    def __init__(self):
        "samples how long the pulse lasts"

        super(DurationSampler, self).__init__(name="duration", observed=False)

    def true_sampler(self, size):

        t90 = self._secondary_samplers["t90"].true_values

        # add that other 10 %

        self._true_values = 1.1 * t90

class TriangleT90Sampler_Cor(ps.AuxiliarySampler):
    """
    Assume Ep-Eiso correlation
    Assume that pulse has a triangle shape
    T_90 = 2*E_iso/L
    (Case a of Ghirlanda et al., 2016)
    """

    _auxiliary_sampler_name = "TriangleDurationSampler"
    def __init__(self):
        "samples how long the pulse last"

        super(TriangleT90Sampler_Cor, self).__init__(
            name="t90", 
            observed=False,
            uses_luminosity=True
            )

    def true_sampler(self, size):

        eiso = self._secondary_samplers["Eiso"].obs_values

        duration = 2*(eiso)/(self._luminosity)

        self._true_values = duration

class EisoSampler(ps.AuxiliarySampler):
    """
    
    """
    _auxiliary_sampler_name = "EisoSampler"

    #Best fit values for Ep-Eiso correlation 
    #Default: Ghirlanda et al., 2016, case a values
    q_a = ps.auxiliary_sampler.AuxiliaryParameter(default=0.033)
    m_a = ps.auxiliary_sampler.AuxiliaryParameter(default=0.91,vmin=0)
    #tau of lognormal distribution from which observed value for E_iso is computed
    tau = ps.auxiliary_sampler.AuxiliaryParameter(default=0.2,vmin=0)

    def __init__(self):
        """Sample E_iso"""

        super(EisoSampler, self).__init__(name="Eiso", observed=True)

    def true_sampler(self, size):

        ep = self._secondary_samplers["ep"].true_values #keV

        #from Ep-Eiso correlation
        eiso = 1./self.m_a * ( np.log10(ep/670.) - self.q_a) * 1e51 #erg

        self._true_values = eiso

    def observation_sampler(self, size):
        #lognormal distribution whose central value is given by the true value
        self._obs_values = np.exp(stats.norm.rvs(loc=self._true_values/1.e52, scale=self.tau, size=size)
        )*1.e52


class CatalogSelector(SpatialSelection):
    def __init__(self) -> None:
        """
        Select GRBs that lie on sky inside cones of Local Volume galaxies 
        """

        super(CatalogSelector, self).__init__(name="catalog_selector")

        self._catalog = LocalVolume.from_lv_catalog()
        self._catalog.prepare_for_popynth()

        self._selected_galaxies: List[Galaxy] = []

    @property
    def catalog(self):
        return self._catalog

    def draw(self, size) -> None:

        # loop through the sky positions

        self._selection = np.zeros(size, dtype=bool)

#        pbar = tqdm(total=size, desc="Scanning catalog")

        for i, (ra, dec), in enumerate(
            zip(self._spatial_distribution.ra, self._spatial_distribution.dec)
        ):

            flag, galaxy = self._catalog.intercepts_galaxy_numba(ra, dec)
            #flag, galaxy = self._catalog.intercepts_galaxy(ra, dec)

            if flag:

                self._selection[i] = True

                self._selected_galaxies.append(galaxy)

 #           pbar.update(1)

    @property
    def selected_galaxies(self) -> List[Galaxy]:

        return self._selected_galaxies

    def show_selected_galaxies(self):

        fig, ax = plt.subplots(
            subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        selected_ra = self._spatial_distribution.ra[self._selection]
        selected_dec = self._spatial_distribution.dec[self._selection]

        # colors = plt.cm.Set1(np.linspace(
        #     0, 1, len(self._selected_galaxies)))

        names = np.unique([galaxy.name for galaxy in self._selected_galaxies])

        colors = plt.cm.Set1(list(range(len(names))))

        _plotted_galaxies = []

        i = 0

        for ra, dec, galaxy in zip(
            selected_ra,
            selected_dec,
            self._selected_galaxies,

        ):
            if galaxy.name not in _plotted_galaxies:

                a = galaxy.radius * (1 / 60)
                b = a * galaxy.ratio

                ellipse = Ellipse(
                    (galaxy.center.ra.deg, galaxy.center.dec.deg),
                    a,
                    b,
                    galaxy.angle,
                    alpha=0.5,
                    color=colors[i],
                    label=galaxy.name,
                    transform=ax.get_transform("icrs")

                )
                e = ax.add_patch(ellipse)
                # e.set_transform(ax.get_transform("icrs"))

                i += 1

                _plotted_galaxies.append(galaxy.name)

            ax.scatter(
                ra,
                dec,
                transform=ax.get_transform("icrs"),
                color="k",
                edgecolor="k",
                s=2,
            )

        ax.legend()

        return fig, ax

        # ax.scatter(
        #     ra, dec, s=10, color="k", edgecolor="k", transform=ax.get_transform("icrs")
        # )
