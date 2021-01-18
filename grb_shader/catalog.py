from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type, Union

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from grb_shader.utils.package_data import get_path_of_data_file


@dataclass
class Galaxy(object):
    name: str
    distance: float
    center: SkyCoord
    radius: float
    ratio: float

    def contains_point(self, ra: float, dec: float) -> bool:
        """
        does this galaxy contain this point?
        
        NOTE: This is currently dumb as it assumes the galaxy is a
        disk!


        :param ra:
        :param dec:
        """
        point = SkyCoord(ra, dec, unit="deg", frame="icrs")

        sep = self.center.separation(point)

        if sep.arcminute < self.radius:

            return True

        else:

            return False


@dataclass
class LocalVolume(object):
    galaxies: Dict[str, Galaxy]

    @classmethod
    def from_file(cls, file_name) -> "LocalVolume":
        """
        Construct a LocalVolume from the LV catalog
        """
        output = {}

        table = pd.read_csv(get_path_of_data_file("lv_catalog.txt"),
                            delim_whitespace=True,
                            header=None,
                            na_values=-99.99,
                            names=["name",
                                   "skycoord",
                                   "radius",
                                   "ratio",
                                   "distance"])

        for rrow in table.iterrows():

            row = rrow[1]

            sk = parse_skycoord(row["skycoord"])

            galaxy = Galaxy(name=row["name"],
                            distance=row["distance"],
                            center=sk,
                            radius=row["radius"],
                            ratio=row["ratio"])

            output[row["name"]] = galaxy

        return cls(output)

    def intercepts_galaxy(self, ra: float, dec: float) -> Tuple[bool, Union[Galaxy, None]]:
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


def parse_skycoord(x: str) -> SkyCoord:
    """
    parse the archaic sailor version of 
    coordinate into an astropy SkyCoord
    """
    
    sign = "+" if ("+" in x) else "-"

    ra, dec = x.split(sign)

    ra_string = f"{ra[:2]}h{ra[2:4]}min{ra[4:]}s"
    dec_str = f"{sign}{dec[:2]}.{dec[2:]}"

    sk = SkyCoord(f"{ra_string} {dec_str}", frame="icrs",
                  unit=(u.hourangle, u.deg))

    return sk


__all__ = ["Galaxy", "LocalVolume"]
