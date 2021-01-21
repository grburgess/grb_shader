from typing import List, Optional, Union

import ligo.skymap.plot
import numba as nb
import numpy as np
import popsynth as ps
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from popsynth.selection_probability import SpatialSelection, UnitySelection
from tqdm.auto import tqdm

from .catalog import Galaxy, LocalVolume


class TDecaySampler(ps.AuxiliarySampler):
    def __init__(self):
        """
        samples the decay of the of the pulse
        """

        super(TDecaySampler, self).__init__(name="tdecay", observed=False)

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

        super(DurationSampler, self).__init__(name="duration", observed=False)

    def true_sampler(self, size):

        t90 = self._secondary_samplers["t90"].true_values

        # add that otehr 10 %

        self._true_values = 1.1 * t90


class CatalogSelector(SpatialSelection):
    def __init__(self) -> None:

        super(CatalogSelector, self).__init__(name="catalog_selector")

        self._catalog = LocalVolume.from_lv_catalog()
        self._catalog.prepare_for_popynth()

        self._selected_galaxies: List[Galaxy] = []

    def draw(self, size) -> None:

        # loop through the sky positions

        self._selection = np.zeros(size, dtype=bool)

        pbar = tqdm(total=size, desc="Scanning catalog")

        for i, (ra, dec), in enumerate(
            zip(self._spatial_distribution.ra, self._spatial_distribution.dec)
        ):

            flag, galaxy = self._catalog.intercepts_galaxy_numba(ra, dec)

            if flag:

                self._selection[i] = True

                self._selected_galaxies.append(galaxy)

            pbar.update(1)

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
                )
                e = ax.add_patch(ellipse)
                e.set_transform(ax.get_transform("icrs"))

                i+=1

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

    def show_all_galaxies(self, frame="icrs"):

        fig, ax = plt.subplots(
            subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        ra = self._spatial_distribution.ra
        dec = self._spatial_distribution.dec

        for _, galaxy in self._catalog.galaxies.items():

            a = galaxy.radius * (1 / 60)
            b = a * galaxy.ratio

            ellipse = Ellipse(
                (galaxy.center.ra.deg, galaxy.center.dec.deg),
                a,
                b,
                galaxy.angle,
                alpha=0.5,
                label=galaxy.name,
            )
            e = ax.add_patch(ellipse)
            e.set_transform(ax.get_transform(frame))

        # ax.scatter(
        #     ra, dec, s=10, color="k", edgecolor="k", transform=ax.get_transform("icrs")
        # )
