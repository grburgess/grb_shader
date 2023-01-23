from typing import List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from popsynth.selection_probability import SpatialSelection
from .catalog import Galaxy, LocalVolume
from popsynth.selection_probability import SelectionParameter
from tqdm.auto import tqdm

class CatalogSelector(SpatialSelection):
    #choose how large error circle around GRB is
    unc_circular_angle = SelectionParameter(default=0.,vmin=0, vmax=180) #degrees
    
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

            flag, galaxy = self._catalog.intercepts_galaxy(ra, dec, self.unc_circular_angle)

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
