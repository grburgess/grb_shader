from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Type, Union

import astropy.units as u
import ipyvolume as ipv
import ipywidgets as widgets
import numba as nb
import numpy as np
import pandas as pd
import pythreejs
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from popsynth import Population

from grb_shader.utils.package_data import get_path_of_data_file
from scipy.spatial.transform import Rotation


@dataclass
class Galaxy(object):
    name: str
    distance: float
    center: SkyCoord
    diameter: float
    ratio: float
    angle: float = 0

    def __post_init__(self):

        # Some useful ellipse properties
        #major angular diameter
        self.a = self.diameter / 60  # deg
        #minor angular diameter
        self.b = self.a * self.ratio  # deg

        self.area = np.pi * np.deg2rad(self.a) * np.deg2rad(self.b)  # rad

        self.vec_gcen = self.sph2cart(self.center.ra.deg, self.center.dec.deg)

        self.rotation_to_ez_from_gcenter = self.rotation_to_from(np.array([0,0,1]),self.vec_gcen)

    def rotation_to_from(self, v1: np.ndarray, v2: np.ndarray, tol: float = 1e-9) -> Rotation:
        """
        Compute the rotation from v2 to v1 using quaternions

        :param v1: vector to rotate to (3D numpy array)
        :param v2: vector to rotate from (3D numpy array)
        :param tol: tolerance below which vectors are assumed parallel or anti-parallel
            (from dot product)
        """

        #normalise vectors
        v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        #check if vectors are parallel or antiparallel
        dot_product = np.sum(v1*v2)

        if np.abs(dot_product-1)<tol:
            #v1 and v2 are parallel -> no rotation needed (theta = 0)
            quat = np.array([0,0,0,1])

        elif np.abs(dot_product+1)<tol:
            #v1 and v2 are antiparallel -> rotate by 180 degrees
            #for construction of rotation axis, any second vector can be chosen,
            #so, use x-axis as a helper vector
            crossproduct = np.cross(v2, [1,0,0], axis=-1)
            crossproduct /= np.linalg.norm(crossproduct)
            #construct unit quaternion with theta=180 degrees
            quat = np.hstack((crossproduct, 0))

        else:
            #compute unit rotation vector from cross product
            crossproduct = np.cross(v2, v1, axis=-1)
            crossproduct /= np.linalg.norm(crossproduct)
            #compute rotation angle from dot product
            theta = np.arccos(np.sum(v1 * v2))
            #construct unit quaternion
            quat = np.hstack((np.sin(theta/2) * crossproduct, np.cos(theta/2)))

        rot = Rotation.from_quat(quat)

        return rot

    def sph2cart(self,ra:float, dec: float) -> np.ndarray:
        """
        Convert spherical to cartesian unit vectors (r=1)

        :param ra: right ascension of point - in degrees
        :param dec: declination of point - in degrees
        """
        #convert coordinates to rad
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        #dec_rad = [-pi/2,pi/2] but theta=[0,pi] (spherical coordinates)
        theta_rad = dec_rad + np.pi/2.

        #convert spherical to Cartesian coordinates
        x = np.cos(ra_rad) * np.sin(theta_rad)
        y = np.sin(ra_rad) * np.sin(theta_rad)
        z = np.cos(theta_rad)
        
        return np.array([x,y,z])

    def contains_point(self, ra: float, dec: float) -> bool:
        """
        does this galaxy contain this point?

        Assumes galaxy is an ellipse with object's properties.

        :param ra: right ascension of GRB
        :param dec: declination of GRB
        """
        #Compute Cartesian unit vector pointing in GRB direction
        vec_GRB = self.sph2cart(ra, dec)

        #rotate GRB vector by angle between z-axis and center of galaxy 
        self.rotation_to_ez_from_gcenter.apply(vec_GRB)

        #consider elliptic cone with random angle between x-axis and semi-major axis
        cos_angle = np.cos(np.deg2rad(self.angle))
        sin_angle = np.sin(np.deg2rad(self.angle))

        #rotate x and y coordinates by angle
        x_t = vec_GRB[0] * cos_angle - vec_GRB[1] * sin_angle
        y_t = vec_GRB[0] * sin_angle + vec_GRB[1] * cos_angle

        #define tangent of semi-major angular distance and semi-minor angular distance
        tan_alpha = np.tan(np.deg2rad(self.a/2.))
        tan_beta = np.tan(np.deg2rad(self.b/2.))

        # Get normalised distance of point to center
        r_norm = (x_t / tan_alpha) ** 2 + (y_t / tan_beta) ** 2

        #test if GRB lies within cone
        if np.sqrt(r_norm) <= vec_GRB[2]:

            return True

        else:

            return False


_exclude = ["LMC", "SMC", ]
#_exclude = []


@dataclass
class LocalVolume(object):
    galaxies: Dict[str, Galaxy]
    _population: Optional[Population] = None

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
            names=["name", "skycoord", "diameter", "ratio", "distance"],
        )

        for rrow in table.iterrows():

            row = rrow[1]

            sk = parse_skycoord(row["skycoord"], row["distance"])

            if (not np.isnan(row["diameter"])) and (row["name"] not in _exclude):

                galaxy = Galaxy(name=row["name"],
                                distance=row["distance"],
                                center=sk,
                                diameter=row["diameter"],
                                ratio=row["ratio"])

                output[row["name"]] = galaxy

        return cls(output)

    def read_population(self, population: Population) -> None:

        self._population: Population = population
        self.prepare_for_popynth()

        self._selected_galaxies: List[Galaxy] = []
        for i, (ra, dec), in enumerate(
            zip(self._population.ra[self._population.selection],
                self._population.dec[self._population.selection])
        ):

            flag, galaxy = self.intercepts_galaxy_numba(ra, dec)
            #flag, galaxy = self._catalog.intercepts_galaxy(ra, dec)

            if flag:

                self._selected_galaxies.append(galaxy)

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)

    @property
    def selected_galaxies(self):
        return self._selected_galaxies

    def sample_angles(self, seed=1234) -> None:
        """
        Sample random orientations for galaxies.
        """

        if seed:
            np.random.seed(seed)

        for name, galaxy in self.galaxies.items():

            galaxy.angle = np.random.uniform(0, 180)

    @property
    def angles(self):

        return np.array([galaxy.angle for _, galaxy in self.galaxies.items()])

    @property
    def diameters(self):

        return np.array([galaxy.diameter for _, galaxy in self.galaxies.items()])

    @property
    def areas(self):

        return np.array([galaxy.area for _, galaxy in self.galaxies.items()])

    def set_angles(self, angles):

        for i, (_, galaxy) in enumerate(self.galaxies):

            galaxy.angle = angles[i]

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
            self._radii[i] = galaxy.diameter * (1. / 60.)

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

            # sphere = Sphere(x,y,z, diameter=v.diameter/1000.,  color="white")
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

        if self._population is not None:
            xs = []
            ys = []
            zs = []

            
            for v in self._selected_galaxies:

                x, y, z = v.center.cartesian.xyz.to("Mpc").value

                # sphere = Sphere(x,y,z, diameter=v.diameter/1000.,  color="white")
                # sphere.plot()

                xs.append(x)
                ys.append(y)
                zs.append(z)

            ipv.scatter(
                np.array(xs),
                np.array(ys),
                np.array(zs),
                marker="sphere",
                size=1.4,
                color="yellow",
            )

            selection = self._population.selection

            xs = []
            ys = []
            zs = []

            for ra, dec in zip(self._population.ra[~selection], self._population.dec[~selection]):

                sk = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                              frame="icrs", distance=25 * u.Mpc)

                x, y, z = sk.cartesian.xyz.to("Mpc").value

                xs.append(x)
                ys.append(y)
                zs.append(z)

            ipv.scatter(
                np.array(xs),
                np.array(ys),
                np.array(zs),
                marker="sphere",
                size=0.25,
                color="red",
            )

            xs = []
            ys = []
            zs = []


            for ra, dec in zip(self._population.ra[selection], self._population.dec[selection]):

                sk = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                              frame="icrs", distance=25 * u.Mpc)

                x, y, z = sk.cartesian.xyz.to("Mpc").value

                xs.append(x)
                ys.append(y)
                zs.append(z)

            ipv.scatter(
                np.array(xs),
                np.array(ys),
                np.array(zs),
                marker="sphere",
                size=1.4,
                color="yellow",
            )

            # for x, y, x in zip(xs, ys, zs):

            #     ipv.plot(
            #         np.array([x, 0]),
            #         np.array([y, 0]),
            #         np.array([z, 0]),
            #         color="orange

#                )

            ipv.xyzlim(30)
        else:

            ipv.xyzlim(12)

        fig.camera.up = [1, 0, 0]
        control = pythreejs.OrbitControls(controlling=fig.camera)
        fig.controls = control
        control.autoRotate = True
        fig.render_continuous = True
        control.autoRotate = True
        toggle_rotate = widgets.ToggleButton(description="Rotate")
        widgets.jslink((control, "autoRotate"), (toggle_rotate, "value"))
        #r_value = toggle_rotate

        #ipv.pylab.save("/home/eschoe/test.html")
        ipv.show()

        return #r_value

    def show_selected_galaxies(self):

        if self._population is None:

            return

        fig, ax = plt.subplots(
            subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        selected_ra = self._population.ra[self._population.selection]
        selected_dec = self._population.dec[self._population.selection]

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

                a = galaxy.diameter * (1 / 60)
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

    def show_all_galaxies(self, frame="icrs"):

        fig, ax = plt.subplots(
            subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        ra = self._spatial_distribution.ra
        dec = self._spatial_distribution.dec

        for _, galaxy in self._catalog.galaxies.items():

            a = galaxy.diameter * (1 / 60)
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


def parse_skycoord(x: str, distance: float) -> SkyCoord:
    """
    parse the archaic sailor version of
    coordinate into an astropy SkyCoord
    """

    sign = "+" if ("+" in x) else "-"

    ra, dec = x.split(sign)

    ra_string = f"{ra[:2]}h{ra[2:4]}m{ra[4:]}s"
    dec_str = f"{sign}{dec[:2]}d{dec[2:4]}m{dec[4:]}s"

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
def _contains_point(ra, dec, diameter, ra_center, dec_center, angle, ratio):

    # assume diameter is in degree
    a = diameter

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
    r_norm = (x_t / (a / 2))**2 + (y_t/(b / 2))**2

    if r_norm <= 1:

        return True

    else:

        return False


__all__ = ["Galaxy", "LocalVolume"]
