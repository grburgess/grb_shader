from typing import List, Optional, Union

import numpy as np
import popsynth as ps
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
            1.0 / 50.0 * (10 * t90 + trise + np.sqrt(trise) * np.sqrt(20 * t90 + trise))
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
        self._selected_galaxies: List[Galaxy] = []

    def draw(self, size) -> None:

        # loop through the sky positions

        self._selection = np.zeros(size, dtype=bool)

        pbar = tqdm(total=size, desc="Scanning catalog")

        for i, (ra, dec), in enumerate(
            zip(self._spatial_distribution.ra, self._spatial_distribution.dec)
        ):

            test, galaxy = self._catalog.intercepts_galaxy(ra, dec)

            if test:

                self._selection[i] = True

                self._selected_galaxies.append(galaxy)

            pbar.update(1)

    @property
    def selected_galaxies(self) -> List[Galaxy]:

        return self._selected_galaxies

    def show_selected_galaxies(self):

        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse

        ax = plt.subplot(projection="hammer")

        ax.figure.set_size_inches((7, 5))

        selected_ra = np.deg2rad(self._spatial_distribution.ra[self._selection])
        selected_dec = np.deg2rad(self._spatial_distribution.dec[self._selection])

        colors = plt.cm.viridis(np.linspace(0, 1, len(self._selected_galaxies)))

        for ra, dec, galaxy, color in zip(
            selected_ra,
            selected_dec,
            self._selected_galaxies,
            colors,
        ):

            a = np.deg2rad(galaxy.radius * (1 / 60))
            b = a * galaxy.ratio

            ellipse = Ellipse(
                (np.pi - galaxy.center.ra.rad, galaxy.center.dec.rad),
                a,
                b,
                galaxy.angle,
                alpha=0.5,
                color=color,
                label=galaxy.name,
            )
            ax.add_patch(ellipse)
            ax.scatter(np.pi - ra, dec, color="k", edgecolor="k", s=10)

        ax.legend()
        ax.get_xaxis().set_visible(False)

        return ax
