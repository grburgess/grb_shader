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
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from popsynth import Population

from grb_shader.utils.package_data import get_path_of_data_file
from scipy.spatial.transform import Rotation
from .plotting.plotting_functions import array_to_cmap


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

        self.area = np.pi * np.deg2rad(self.a) * np.deg2rad(self.b)  # rad2
        
        #Compute Cartesian unit vector pointing in galaxy center
        self.vec_gcen = self.center.cartesian.xyz.to_value()

    def contains_point(self, ra: float, dec: float) -> bool:
        """
        does this galaxy contain this point?

        Assumes galaxy is an ellipse with object's properties.

        :param ra: right ascension of GRB
        :param dec: declination of GRB
        """
        x_center, y_center, z_center = self.vec_gcen
        
        sk_coord_grb = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        x,y,z = sk_coord_grb.cartesian.xyz.to_value()
        
        return _contains_point(x, y, z, 
                               self.diameter, 
                               x_center, y_center, z_center, 
                               self.angle, 
                               self.ratio)


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

    def read_population(self, population: Population,without_unc=True,unc_angle=1,n_samp=100) -> None:

        self._population: Population = population
        self.prepare_for_popynth()
        
        self._selection = np.zeros(self._population.n_detections, dtype=bool)
        self._selected_galaxies: List[Galaxy] = []
        for i, (ra, dec), in enumerate(
            zip(self._population.ra[self._population.selection],
                self._population.dec[self._population.selection])
        ):
            if without_unc:
                flag, galaxy = self.intercepts_galaxy(ra, dec)
            else:
                flag, galaxy = self.intercepts_galaxy_with_grb_uncertainty(ra, dec,unc_angle,n_samp)
            #flag2, galaxy2 = self.intercepts_galaxy_old(ra, dec)

            if flag:
                #print('New:', i, galaxy.name,'Gal a,b:',galaxy.a,galaxy.b)
                self._selected_galaxies.append(galaxy)
                self._selection[i] = True

            #if flag2:
            #    print('Old:', i, galaxy2.name)
    
    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)
    
    @property
    def selection(self):
        return self._selection

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
    
    @property
    def centers(self):

        return np.array([galaxy.center for _, galaxy in self.galaxies.items()])

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
        self._x = np.empty(self.n_galaxies)
        self._y = np.empty(self.n_galaxies)
        self._z = np.empty(self.n_galaxies)

        for i, (name, galaxy) in enumerate(self.galaxies.items()):

            # convert to degree from arcmin
            self._radii[i] = galaxy.diameter * (1. / 60.)

            # convert to radian
            self._angles[i] = np.deg2rad(galaxy.angle)

            self._ra[i] = galaxy.center.ra.deg
            self._dec[i] = galaxy.center.dec.deg

            self._vec = galaxy.center.cartesian.xyz.to_value()
            self._x[i] = self._vec[0]
            self._y[i] = self._vec[1]
            self._z[i] = self._vec[2]

            self._ratio[i] = galaxy.ratio
    
    def intercepts_galaxy(
        self, ra: float, dec: float
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        """
        
        sk_coord_grb = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        vec_GRB = sk_coord_grb.cartesian.xyz.to_value()
        vec_GRB /= np.linalg.norm(vec_GRB)

        flag, idx = _intercepts_galaxy(vec_GRB[0],
                                       vec_GRB[1],
                                       vec_GRB[2],
                                       self._radii,
                                       self._x,
                                       self._y,
                                       self._z,
                                       self._angles,
                                       self._ratio,
                                       self.n_galaxies
                                       )

        if idx > 0:
            out = list(self.galaxies.values())[idx]

        else:

            out = None

        return flag, out
    
    def intercepts_galaxy_with_grb_uncertainty(
        self, ra: float, dec: float,angle:float=1,n:int=100
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        """
        test_ra,test_dec = sample_err_circ(ra,dec,angle,n)
        
        sk_coord_grb = SkyCoord(ra=test_ra*u.degree, dec=test_dec*u.degree, frame='icrs')

        flag, idx = _intercepts_galaxy_unc(
            sk_coord_grb.cartesian.x,
            sk_coord_grb.cartesian.y,
            sk_coord_grb.cartesian.z,
            self._radii,
            self._x,
            self._y,
            self._z,
            self._angles,
            self._ratio,
            self.n_galaxies,
            n)

        if idx > 0:
            out = list(self.galaxies.values())[idx]

        else:

            out = None

        return flag, out

    def intercepts_galaxy_old(
        self, ra: float, dec: float
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        """

        flag, idx = _intercepts_galaxy_old(ra,
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

            selection = self._selection

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

    def show_selected_galaxies(self,projection = 'astro degrees mollweide',radius=10.0,center=None,ax=None,grb_color="k"):

        if self._population is None:

            return
        
        assert projection in [
            "astro degrees aitoff",
            "astro degrees mollweide",
            "astro hours aitoff",
            "astro hours mollweide",
            "astro degrees globe",
            "astro hours globe",
            "astro degrees zoom",
            "astro hours zoom"
        ]
        
        skw_dict = dict(projection=projection)
        
        if projection in ["astro degrees globe", "astro hours globe",
                          "astro degrees zoom","astro hours zoom"]:

            if center is None:

                center = SkyCoord(0, 0, unit="deg")

            skw_dict = dict(projection=projection, center=center)

        if projection == "astro zoom":

            assert radius is not None, "you must specify a radius"

            skw_dict = dict(projection=projection, center=center, radius=radius)

        #fig = plt.figure(figsize=(9, 4), dpi=100)
        
        #ax = plt.axes(skw_dict)
        
        if ax is None:
        
            fig, ax = plt.subplots(subplot_kw=skw_dict)

            fig.set_size_inches((8, 4))
        else:
            
            fig = ax.get_figure()

        selected_ra = self._population.ra[self._selection]
        selected_dec = self._population.dec[self._selection]

        names = np.unique([galaxy.name for galaxy in self._selected_galaxies])

        #colors = plt.cm.Set1(list(range(len(names))))

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
                    angle=galaxy.angle,
                    alpha=0.5,
                    color=f'C0{i}',
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
                color=grb_color,
                edgecolor=grb_color,
                s=2,
            )
        ax.grid()
        #ax.legend()

        return fig, ax
    
    def show_all_galaxies_area(self, frame="icrs",cmap='viridis',uselog=True):
        # scatter plot of all galaxies, coloring by area
        fig, ax =plt.subplots(figsize=(7,5))
        ax = plt.subplot(111,projection="astro degrees mollweide")
        
        _, colors = array_to_cmap(self.areas, cmap=cmap, use_log=uselog)
        
        if uselog:
            norm = mpl.colors.LogNorm(vmin=min(self.areas), vmax=max(self.areas))
        else:
            norm = mpl.colors.Normalize(vmin=min(self.areas), vmax=max(self.areas))

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, label=r'Angular Area [rad$^2$]',
                    pad=0.1,fraction=0.08,orientation="horizontal")

        i = 0
        for _, galaxy in self.galaxies.items():
                
            ax.scatter(x=float(galaxy.center.ra.deg), 
                        y=float(galaxy.center.dec.deg),
                        color=colors[i],s=7,
                        zorder=np.log(self.areas[i])+20,
                        #edgecolor='k',
                        transform=ax.get_transform(frame))
            
            i+=1
        return fig, ax

    def show_all_galaxies(self, frame="icrs"):

        fig, ax = plt.subplots(
            subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        i = 0
        for _, galaxy in self.galaxies.items():

            a = galaxy.diameter * (1 / 60)
            b = a * galaxy.ratio

            ellipse = Ellipse(
                (galaxy.center.ra.deg, galaxy.center.dec.deg),
                a,
                b,
                galaxy.angle,
                alpha=0.9,
                label=galaxy.name,
            )
            e = ax.add_patch(ellipse)
            e.set_transform(ax.get_transform(frame))
            
            i+=1
            

            #ax.scatter(
            #    galaxy.center.ra.deg,
            #    galaxy.center.dec.deg,
            #    transform=ax.get_transform("icrs"),
            #    color="k",
            #    edgecolor="k",
            #    s=5,
            #)
        return fig, ax


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

@nb.njit(fastmath=False)
def _calc_cross(vec_1,vec_2):
    """Cross product of 2 times 3 dimensinal vectors
    optimized performance using numba
    """
    res=np.empty((3,1),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in nb.prange(vec_2.shape[0]):
            res[0]=vec_1[1] * vec_2[2] - vec_1[2] * vec_2[1]
            res[1]=vec_1[2] * vec_2[0] - vec_1[0] * vec_2[2]
            res[2]=vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]
    
    return res

@nb.njit(fastmath=False)
def _rotate_to_from_numba(v1, v2):
    """
    Determine rotation axis and angle that is needed to rotate
    vector v2 to vector v1
    
    Parameters
    ----------
    v1: the vector (array) to rotate to
    v2: the vector (array) to rotate from
    tol: tolerance below which vectors are assumed to be either parallel or
         anti-parallel
    
    Returns
    -------
    theta: needed rotation angle (rad)
    crossproduct: normalized vector corresponding to rotation axis
    """
    tol = 1e-12
    #normalise vectors
    v1 /= np.sqrt(np.dot(v1, v1))
    v2 /= np.sqrt(np.dot(v2, v2))

    #check if vectors are parallel or antiparallel
    dot_product = np.sum(v1*v2)

    if np.abs(dot_product-1)<tol:
        #v1 and v2 are parallel -> no rotation needed (theta = 0)
        theta = 0
        crossproduct = np.array([0.,0.,1.],dtype='float64').flatten()


    elif np.abs(dot_product+1)<tol:
        #v1 and v2 are antiparallel -> rotate by 180 degrees
        #for construction of rotation axis, any second vector can be chosen,
        #so, use x-axis as a helper vector
        crossproduct = _calc_cross(v2, np.array([1.,0.,0.])).flatten()
        crossproduct /= np.sqrt(np.dot(crossproduct, crossproduct))

    else:
        #compute unit rotation vector from cross product
        crossproduct = _calc_cross(v2, v1).flatten()
        crossproduct /= np.sqrt(np.dot(crossproduct, crossproduct))
        #compute rotation angle from dot product
        theta = np.arccos(np.sum(v1 * v2))
        
    return theta,crossproduct

@nb.njit(fastmath=False)
def _rotation_matrix(axis, theta):
    """
    Determine rotation matrix associated with counterclockwise rotation about
    the given axis by theta (rad).
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

@nb.njit(fastmath=False)
def _intercepts_galaxy(x, y, z, radii, x_center, y_center, z_center, angle, ratio, N):

    for n in range(N):

        if _contains_point(x,
                           y,
                           z,
                           radii[n],
                           x_center[n],
                           y_center[n],
                           z_center[n],
                           angle[n],
                           ratio[n]
                           ):

            return True, n

    return False, -1

@nb.njit(fastmath=False)
def _contains_point(x, y, z, diameter, x_center, y_center, z_center, angle, ratio) -> bool:
    """
    does this galaxy contain this point?

    Assumes galaxy is an ellipse with object's properties.

    :param ra: right ascension of GRB
    :param dec: declination of GRB
    """
    # assume diameter is in degree
    a = diameter

    b = a * ratio  # deg

    # Cartesian unit vector pointing in GRB direction
    vec_GRB = np.array([x, y, z])

    vec_gcen = np.array([x_center, y_center, z_center])
    vec_gcen /= np.sqrt(np.dot(vec_gcen, vec_gcen))
    
    if np.rad2deg(np.arccos(np.dot(vec_GRB,vec_gcen))) < b/2:
        #print(np.rad2deg(np.arccos(np.dot(vec_GRB,vec_gcen))), ' < b/2 = ', b/2.)
        return True
    
    elif np.rad2deg(np.arccos(np.dot(vec_GRB,vec_gcen))) > a/2:
        #print(np.rad2deg(np.arccos(np.dot(vec_GRB,vec_gcen))), ' > a/2 = ', a/2.)
        return False

    #rotate GRB vector by angle between z-axis and center of galaxy 
    theta, axis = _rotate_to_from_numba(np.array([0.,0.,1.]),vec_gcen)
    vec_GRB_rot = np.dot(_rotation_matrix(axis, theta), vec_GRB)

    #consider elliptic cone with random angle between x-axis and semi-major axis
    #cos_angle = np.cos(np.pi - angle)
    #sin_angle = np.sin(np.pi - angle)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    #rotate x and y coordinates by angle
    x_t = vec_GRB_rot[0] * cos_angle - vec_GRB_rot[1] * sin_angle
    y_t = vec_GRB_rot[0] * sin_angle + vec_GRB_rot[1] * cos_angle

    #define tangent of semi-major angular distance and semi-minor angular distance
    tan_alpha = np.tan(np.deg2rad(a/2.))
    tan_beta = np.tan(np.deg2rad(b/2.))

    # Get normalised distance of point to center
    r_norm = (x_t / tan_alpha) ** 2 + (y_t / tan_beta) ** 2

    #test if GRB lies within cone
    if np.sqrt(r_norm) < vec_GRB_rot[2]:
        
        #print(np.sqrt(r_norm), ' < ', vec_GRB_rot[2])

        return True

    else:

        return False
"""
def sample_err_circ(ra,dec,angle=1,n=100):
    
    r = np.sqrt(np.random.uniform(0,angle**2,size=n))
    ang = np.pi * np.random.uniform(0,2,size=n)
    
    delta_ra = r*np.cos(ang)
    ra_new = (ra+delta_ra)%360
    
    delta_dec = r*np.sin(ang)
    dec_new = (dec+delta_dec+90)%180-90
    
    test_ra = np.append(ra,ra_new)
    test_dec = np.append(dec,dec_new)
    
    return(test_ra,test_dec)

@nb.njit(fastmath=False)
def _intercepts_galaxy_unc(x,y,z, radii, x_center, y_center, z_center, angle, ratio, N,n_samp):

    for n in range(N):
        
        for i in range(n_samp):

            if _contains_point(x[i],
                               y[i],
                               z[i],
                            radii[n],
                            x_center[n],
                            y_center[n],
                            z_center[n],
                            angle[n],
                            ratio[n]
                            ):

                return True, n

    return False, -1
"""


@nb.jit(fastmath=False)
def _intercepts_galaxy_old(ra, dec, radii, ra_center, dec_center, angle, ratio, N):

    for n in range(N):

        if _contains_point_old(ra,
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
def _contains_point_old(ra, dec, diameter, ra_center, dec_center, angle, ratio,error=0.):

    # assume diameter is in degree
    a = diameter/2

    b = a * ratio/2  # deg

    cos_angle = np.cos(np.pi - angle)
    sin_angle = np.sin(np.pi - angle)

    # Get xy dist from point to center
    x = ra - ra_center
    y = dec - dec_center

    # Transform to along major/minor axes
    x_t = x * cos_angle - y * sin_angle
    y_t = x * sin_angle + y * cos_angle

    # Get normalised distance of point to center
    r_norm = (x_t / a)**2 + (y_t/b)**2

    if r_norm <= 1:
        
        return True

    elif error == 0.:
        
        return False
    
    elif error > 0.:
        #e: eccentricity: 
        e = np.sqrt(a**2-b**2)
        
        if error > np.sqrt(x_t**2+y_t**2):
            #error is larger than distance between center
            #of galaxy and GRB 
            return True
        
        else: 
            length = np.sqrt(x_t^2 + y_t^2) - error
            unit_vector_x = 1/np.sqrt(x_t^2 + y_t^2) *x_t
            unit_vector_y = 1/np.sqrt(x_t^2 + y_t^2) *y_t
            
            r_norm_2 = (length * unit_vector_x / a) **2 + (length * unit_vector_y / b)**2
        
            if r_norm_2 <= 1:
            #test whether point P1 lies within ellipse
            #P1: intersection of circle edge with line
            #connecting center of ellipse and circle
                       
                return True
        
        f1_x = -e
        f1_y = 0
        
        dist = np.sqrt((x_t-f1_x)**2+ (y_t-f1_y)**2)
        
        if error>dist:
            return True
        else:
            length = dist - error
            unit_vector_x = (x_t-f1_x)/dist
            unit_vector_y = (y_t-f1_y)/dist
            
            r_norm = (length * unit_vector_x / a) **2 + (length * unit_vector_y / b)**2
        
            if r_norm <= 1:
            #test whether point P2 lies within ellipse
            #P2: intersection of circle edge with line
            #connecting left focus point of ellipse and circle center
                       
                return True
            
        f2_x = e
        f2_y = 0
        
        dist = np.sqrt((x_t-f2_x)**2+ (y_t-f2_y)**2)
        
        if error>dist:
            return True
        else:
            length = dist - error
            unit_vector_x = (x_t-f2_x)/dist
            unit_vector_y = (y_t-f2_y)/dist
            
            r_norm = (length * unit_vector_x / a) **2 + (length * unit_vector_y / b)**2
        
            if r_norm <= 1:
            #test whether point P3 lies within ellipse
            #P3: intersection of circle edge with line
            #connecting right focus point of ellipse and circle center
                       
                return True
            
    else:

        return False


__all__ = ["Galaxy", "LocalVolume"]

