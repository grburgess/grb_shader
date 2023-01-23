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

        self.area = np.pi/4 * np.deg2rad(self.a) * np.deg2rad(self.b)  # rad2
        
        #Compute Cartesian unit vector pointing in galaxy center
        self.vec_gcen = self.center.cartesian.xyz.to_value()

    def contains_point(self, ra: float, dec: float,grb_circ_unc=0.) -> bool:
        """
        does this galaxy contain this point?

        Assumes galaxy is an ellipse with object's properties.

        :param ra: right ascension of GRB
        :param dec: declination of GRB
        """
        
        return _contains_point(ra,
                           dec,
                           self.diameter,
                           self.ra,
                           self.dec,
                           self.angle,
                           self.ratio,
                           grb_circ_unc
                           )


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

    def read_population(self, population: Population,unc_angle=None) -> None:

        self._population: Population = population
        self.prepare_for_popynth()
        
        self._selection = np.zeros(self._population.n_detections, dtype=bool)
        self._n_hit_galaxies = np.zeros(self._population.n_detections, dtype=int)
        self._selected_galaxies: List[Galaxy] = []
        
        if unc_angle is None:
            unc_angle = self._population.pop_synth['spatial selection']['SpatialSelection']['unc_circular_angle']
            
        for i, (ra, dec), in enumerate(
            zip(self._population.ra[self._population.selection],
                self._population.dec[self._population.selection])
        ):
            flag, galaxy = self.intercepts_galaxy_all(ra, dec,unc_angle)

            if flag:
                #print('New:', i, galaxy.name,'Gal a,b:',galaxy.a,galaxy.b)
                self._selected_galaxies.append(galaxy)
                self._selection[i] = True
                self._n_hit_galaxies[i] = len(galaxy)

            #if flag2:
            #    print('Old:', i, galaxy2.name)
    
    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)
    
    @property
    def selection(self):
        #True at indices of GRBs that intersect with at least one GRB
        return self._selection

    @property
    def selected_galaxies(self):
        return self._selected_galaxies
    
    @property
    def n_hit_galaxies(self):
        #how many galaxies were hit by one GRB
        return self._n_hit_galaxies

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

            self._ratio[i] = galaxy.ratio

    def intercepts_galaxy(
        self, ra: float, dec: float, grb_circ_unc:float = 0.
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
                                       self.n_galaxies,
                                       grb_circ_unc
                                       )

        if idx > 0:
            #print('Flag intercepts_galaxy',flag)
            out = list(self.galaxies.values())[idx]

        else:

            out = None

        return flag, out
    
    def intercepts_galaxy_all(
        self, ra: float, dec: float, grb_circ_unc:float = 0.
    ) -> Tuple[bool, Union[Galaxy, None]]:
        """
        Test if the sky point intecepts a galaxy in the local volume
        and if so return that galaxy
        
        Find multiple galaxies that can be hit and return list
        """

        flag, idx = _intercepts_galaxy_all(ra,
                                       dec,
                                       self._radii,
                                       self._ra,
                                       self._dec,
                                       self._angles,
                                       self._ratio,
                                       self.n_galaxies,
                                       grb_circ_unc
                                       )

        if len(idx) > 0:
            out = []
            for i in range(len(idx)):
                out += [list(self.galaxies.values())[idx[i]]]

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

    def show_selected_galaxies(self, projection = 'astro degrees mollweide',radius=10.0,center=None,ax=None,grb_color="k"):

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

        names = np.unique([galaxy[0].name for galaxy in self._selected_galaxies])

        #colors = plt.cm.Set1(list(range(len(names))))

        _plotted_galaxies = []

        i = 0

        for ra, dec, galaxy in zip(
            selected_ra,
            selected_dec,
            self._selected_galaxies,

        ):
            
            if galaxy[0].name not in _plotted_galaxies:

                a = galaxy[0].diameter * (1 / 60)
                b = a * galaxy[0].ratio

                ellipse = Ellipse(
                    (galaxy[0].center.ra.deg, galaxy[0].center.dec.deg),
                    a,
                    b,
                    angle=galaxy[0].angle,
                    alpha=0.5,
                    color=f'C0{i}',
                    label=galaxy[0].name,
                    transform=ax.get_transform("icrs"),
                    zorder=3.

                )
                
                e = ax.add_patch(ellipse)
                # e.set_transform(ax.get_transform("icrs"))

                i += 1

                _plotted_galaxies.append(galaxy[0].name)

            ax.scatter(
                ra,
                dec,
                transform=ax.get_transform("icrs"),
                color=grb_color,
                edgecolor=grb_color,
                s=2,
                zorder=3.
            )
        ax.grid()
        #ax.legend()

        return fig, ax
    
    def show_all_galaxies_area(self, ax=None,frame="icrs",cmap='viridis',uselog=True):
        # scatter plot of all galaxies, coloring by area
        if ax is None:
            fig, ax =plt.subplots(figsize=(7,5))
            ax = plt.subplot(111,projection="astro degrees mollweide")
        else:
            fig = ax.get_figure()
        
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

    def show_all_galaxies(self, ax=None, color='C00',frame="icrs",scatter=True,s=1):
        if ax is None:
            fig, ax = plt.subplots(
                subplot_kw={"projection": "astro degrees mollweide"})
            fig.set_size_inches((7, 5))
        else:
            fig = ax.get_figure()

        i = 0
        for _, galaxy in self.galaxies.items():

            a = galaxy.diameter * (1 / 60)
            b = a * galaxy.ratio

            ellipse = Ellipse(
                (galaxy.center.ra.deg, galaxy.center.dec.deg),
                a,
                b,
                galaxy.angle,
                color=color,
                alpha=0.6,
                label=galaxy.name,
                zorder=3
            )
            e = ax.add_patch(ellipse)
            e.set_transform(ax.get_transform(frame))
            
            i+=1
            
            if scatter:
                ax.scatter(
                    galaxy.center.ra.deg,
                    galaxy.center.dec.deg,
                    transform=ax.get_transform("icrs"),
                    color=color,
                    edgecolor=color,
                    s=s,
                )
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


@nb.jit(fastmath=False)
def _intercepts_galaxy(ra, dec, radii, ra_center, dec_center, angle, ratio, N,grb_circ_unc):

    for n in range(N):

        if _contains_point(ra,
                           dec,
                           radii[n],
                           ra_center[n],
                           dec_center[n],
                           angle[n],
                           ratio[n],
                           grb_circ_unc
                           ):
            #print('coinc')

            return True, n

    return False, -1

@nb.jit(fastmath=False)
def _intercepts_galaxy_all(ra, dec, radii, ra_center, dec_center, angle, ratio, N,grb_circ_unc):
    
    list_gals = []
    flag = False
    for n in range(N):

        if _contains_point(ra,
                           dec,
                           radii[n],
                           ra_center[n],
                           dec_center[n],
                           angle[n],
                           ratio[n],
                           grb_circ_unc
                           ):
            flag = True
            list_gals.append(n)
            
    return flag,list_gals


@nb.jit(fastmath=False)
def _contains_point(ra, dec, diameter, ra_center, dec_center, angle, ratio,error=0.):

    # assume diameter is in degree
    a = diameter/2

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
    r_norm = (x_t / a)**2 + (y_t/b)**2

    if r_norm <= 1:
        
        return True

    elif error == 0.:
        
        return False
    
    elif error > 0.:
        #e: eccentricity: 
        e = np.sqrt(a**2-b**2)
        
        if error > np.sqrt(x_t**2+y_t**2):
            #print(error,'>', np.sqrt(x_t**2+y_t**2))
            #error is larger than distance between center
            #of galaxy and GRB 
            return True
        
        if error <= np.sqrt(x_t**2+y_t**2):
            #test whether point P1 lies within ellipse
            #P1: intersection of circle edge with line
            #connecting center of ellipse and circle
            #print('Test1')
            length = np.sqrt(x_t**2 + y_t**2) - error
            unit_vector_x = x_t/np.sqrt(x_t**2 + y_t**2)
            unit_vector_y = y_t/np.sqrt(x_t**2 + y_t**2)
            
            r_norm_p1 = (length * unit_vector_x / a) **2 + (length * unit_vector_y / b)**2
            #print(r_norm_2)
        
            if r_norm_p1 <= 1:
                #print(r_norm_2,'<',1)
                       
                return True
        
        f1_x = -e
        f1_y = 0
        
        dist_f1_grb_center = np.sqrt((x_t-f1_x)**2+ (y_t-f1_y)**2)
        
        if error > dist_f1_grb_center:
            #print(error, '>', dist_f1_grb_center)
            return True
        
        if error < dist_f1_grb_center:
            #test whether point P2 lies within ellipse
            #P2: intersection of circle edge with line
            #connecting left focus point of ellipse and circle center
            
            #print('Test2')
            
            length_f1_p2 = dist_f1_grb_center - error
            unit_vector_f1_p2_x = (x_t-f1_x)/dist_f1_grb_center
            unit_vector_f1_p2_y = (y_t-f1_y)/dist_f1_grb_center

            x_p2 = length_f1_p2 * unit_vector_f1_p2_x + f1_x
            y_p2 = length_f1_p2 * unit_vector_f1_p2_y + f1_y
            
            r_norm_p2 = (x_p2 / a) **2 + (y_p2 / b)**2
        
            if r_norm_p2 <= 1:
                #print(r_norm_p2,'<',1)                       
                return True
            
        f2_x = e
        f2_y = 0
        
        dist_f2_grb_center = np.sqrt((x_t-f2_x)**2+ (y_t-f2_y)**2)
        
        if error > dist_f2_grb_center:
            #print('coinc')
            
            return True
        
        if error < dist_f2_grb_center:
            #test whether point P3 lies within ellipse
            #P3: intersection of circle edge with line
            #connecting right focus point of ellipse and circle center
            
            #print('Test3')
            
            length_f2_p3 = dist_f2_grb_center - error
            unit_vector_f2_p3_x = (x_t-f2_x)/dist_f2_grb_center
            unit_vector_f2_p3_y = (y_t-f2_y)/dist_f2_grb_center

            x_p3 = length_f2_p3 * unit_vector_f2_p3_x + f2_x
            y_p3 = length_f2_p3 * unit_vector_f2_p3_y + f2_y
            
            r_norm_p3 = (x_p3 / a) **2 + (y_p3 / b)**2
        
            if r_norm_p3 <= 1:
                #print('Test3')
                       
                return True
            
            else:
                return False
        else:
            return False
            
    else:

        return False


__all__ = ["Galaxy", "LocalVolume"]

