from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type, Union

import astropy.units as u
import ipyvolume as ipv
import ipywidgets as widgets
import numba as nb
import numpy as np
import pandas as pd
import pythreejs
from astropy.coordinates import SkyCoord

from grb_shader.utils.disk import Sphere
from grb_shader.utils.package_data import get_path_of_data_file


@dataclass
class Galaxy(object):
    name: str
    distance: float
    center: SkyCoord
    radius: float
    ratio: float
    angle: float = 0


    def __post_init__(self):
           
        # Some useful ellipse properties

        self.a = self.radius / 60  # deg

        self.b = self.a * self.ratio  # deg

        self.area = np.pi * np.deg2rad(self.a) * np.deg2rad(self.b)  # rad

    def contains_point(self, ra: float, dec: float) -> bool:
        """
        does this galaxy contain this point?

        Assumes galaxy is an ellipse with object's properties.

        :param ra:
        :param dec:
        """


        cos_angle = np.cos(np.pi - np.deg2rad(self.angle))
        sin_angle = np.sin(np.pi - np.deg2rad(self.angle))

        # Get xy dist from point to center
        x = ra - self.center.ra.deg
        y = dec - self.center.dec.deg

        # Transform to along major/minor axes
        x_t = x * cos_angle - y * sin_angle
        y_t = x * sin_angle + y * cos_angle

        # Get normalised distance of point to center
        r_norm = x_t ** 2 / (self.a / 2) ** 2 + y_t ** 2 / (self.b / 2) ** 2

        if r_norm <= 1:

            return True

        else:

            return False


        # a = self.radius * (1 / 60)  # deg

        # return _contains_point(ra,
        #                        dec,
        #                        a,
        #                        self.center.ra.deg,
        #                        self.center.dec.deg,
        #                        np.deg2rad(self.angle),
        #                        self.ratio
        #                        )


_exclude = ["LMC", "SMC", ]
#_exclude = []


@dataclass
class LocalVolume(object):
    galaxies: Dict[str, Galaxy]

    @classmethod
    def from_lv_catalog(cls) -> "LocalVolume":
        """
        Construct a LocalVolume from the LV catalog
        """
        output = OrderedDict()

        table = pd.read_csv(
            get_path_of_data_file("lv_catalog.txt"),
            delim_whitespace=True,
            header=None,
            na_values=-99.99,
            names=["name", "skycoord", "radius", "ratio", "distance"],
        )

        for rrow in table.iterrows():

            row = rrow[1]

            sk = parse_skycoord(row["skycoord"], row["distance"])

            if (not np.isnan(row["radius"])) and (row["name"] not in _exclude):

                galaxy = Galaxy(name=row["name"],
                                distance=row["distance"],
                                center=sk,
                                radius=row["radius"],
                                ratio=row["ratio"])

                output[row["name"]] = galaxy

        return cls(output)

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)

    def sample_angles(self, seed=1234) -> None:
        """
        Sample random orientations for galaxies.
        """

        if seed:
            np.random.seed(seed)

        for name, galaxy in self.galaxies.items():

            galaxy.angle = np.random.uniform(0, 360)

    def prepare_for_popynth(self) -> None:
        """
        extract info for fast reading

        """

        # sample the angles

        self.sample_angles()

        self._radii = np.empty(self.n_galaxies)
        self._angles = np.empty(self.n_galaxies)
        self._ra = np.empty(self.n_galaxies)
        self._dec = np.empty(self.n_galaxies)
        self._ratio = np.empty(self.n_galaxies)

        for i, (name, galaxy) in enumerate(self.galaxies.items()):

            # convert to degree from arcmin
            self._radii[i] = galaxy.radius * (1. / 60.)

            # convert to radian
            self._angles[i] = np.deg2rad(galaxy.angle)

            self._ra[i] = galaxy.center.ra.deg
            self._dec[i] = galaxy.center.dec.deg

            self._ratio[i] = galaxy.ratio

    def intercepts_galaxy_numba(
        self, ra: float, dec: float
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        """

        flag, idx = _intercepts_galaxy(ra,
                                       dec,
                                       self._radii,
                                       self._ra,
                                       self._dec,
                                       self._angles,
                                       self._ratio,
                                       self.n_galaxies
                                       )

        if idx > 0:
            out = list(self.galaxies.values())[idx]

        else:

            out = None

        return flag, out

    def intercepts_galaxy(
        self, ra: float, dec: float
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        """

        for name, galaxy in self.galaxies.items():

            if galaxy.contains_point(ra, dec):
                return True, galaxy

        else:

            return False, None

    def __dir__(self):
        # Get the names of the attributes of the class
        l = list(self.__class__.__dict__.keys())

        # Get all the children
        l.extend([x.name for k, x in self.galaxies.items()])

        return l

    def __getattr__(self, name):
        if name in self.galaxies:
            return self.galaxies[name]
        else:
            return super().__getattr__(name)

    def display(self):

        fig = ipv.figure()

        ipv.pylab.style.box_off()
        ipv.pylab.style.axes_off()
        ipv.pylab.style.set_style_dark()
        # ipv.pylab.style.background_color(background_color)

        xs = []
        ys = []
        zs = []

        for k, v in self.galaxies.items():

            x, y, z = v.center.cartesian.xyz.to("Mpc").value

            # sphere = Sphere(x,y,z, radius=v.radius/1000.,  color="white")
            # sphere.plot()

            xs.append(x)
            ys.append(y)
            zs.append(z)

        ipv.scatter(
            np.array(xs),
            np.array(ys),
            np.array(zs),
            marker="sphere",
            size=0.5,
            color="white",
        )

        fig.camera.up = [1, 0, 0]
        control = pythreejs.OrbitControls(controlling=fig.camera)
        fig.controls = control
        control.autoRotate = True
        fig.render_continuous = True
        control.autoRotate = True
        toggle_rotate = widgets.ToggleButton(description="Rotate")
        widgets.jslink((control, "autoRotate"), (toggle_rotate, "value"))
        r_value = toggle_rotate

        ipv.xyzlim(12)

        ipv.show()

        return r_value


def parse_skycoord(x: str, distance: float) -> SkyCoord:
    """
    parse the archaic sailor version of
    coordinate into an astropy SkyCoord
    """

    sign = "+" if ("+" in x) else "-"

    ra, dec = x.split(sign)

    ra_string = f"{ra[:2]}h{ra[2:4]}min{ra[4:]}s"
    dec_str = f"{sign}{dec[:2]}.{dec[2:]}"

    sk = SkyCoord(
        f"{ra_string} {dec_str}",
        distance=distance * u.Mpc,
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )

    return sk


@nb.jit(fastmath=False)
def _intercepts_galaxy(ra, dec, radii, ra_center, dec_center, angle, ratio, N):

    for n in range(N):

        if _contains_point(ra,
                           dec,
                           radii[n],
                           ra_center[n],
                           dec_center[n],
                           angle[n],
                           ratio[n]
                           ):

            return True, n

    return False, -1


@nb.jit(fastmath=False)
def _contains_point(ra, dec, radius, ra_center, dec_center, angle, ratio):

    # assume radius is in degree
    a = radius

    b = a * ratio  # deg

    cos_angle = np.cos(np.pi - angle)
    sin_angle = np.sin(np.pi - angle)

    # Get xy dist from point to center
    x = ra - ra_center
    y = dec - dec_center

    # Transform to along major/minor axes
    x_t = x * cos_angle - y * sin_angle
    y_t = x * sin_angle + y * cos_angle

    # Get normalised distance of point to center
    r_norm = (x_t/ (a / 2))**2 + (y_t/(b / 2))**2

    if r_norm <= 1:

        return True

    else:

        return False


__all__ = ["Galaxy", "LocalVolume"]
