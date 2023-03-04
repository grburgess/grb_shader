import numpy as np

from .catalog import LocalVolume

def select_gals_sfr(galaxies, lim_halpha, lim_uv):
    """Find galaxies in galaxies that fulfill
    SFR limit given by Halpha and UV SFR
    
    If galaxy has no given SFR, it cannot be rejected
    based on the SFR
    
    If only upper limit is given for galaxy, the galaxy
    is only rejected if upper values is lower than upper limit
    given by lim_halpha and lim_uv

    :param galaxies: list of galaxy names
    :type galaxies: list
    :param lim_halpha: lower limit of H alpha SFR
    :type lim_halpha: float
    :param lim_uv: lower limit of UV SFR
    :type lim_uv: float
    :return: list of selected galaxies and discarded galaxies
    """
    
    lv = LocalVolume.from_lv_catalog()
    
    gal_with_sfr = lv.galaxies_with_sfr
    gal_names_with_sfr = list(gal_with_sfr.keys())
    
    selected_gals = []
    discarded_gals = []
    #select galaxies with larger SFR than given limit or if no SFR given in LV catalog
    #discard galaxy if upper limit or value is smaller than chosen threshold
    for gal in galaxies:
        #print(gal)

        if gal in gal_names_with_sfr:

            #hit_analyzable_gals_wsfr += [gal]

            sfr_halpha = gal_with_sfr[gal].logSFR
            sfr_halpha_type = gal_with_sfr[gal].logSFR_type
            sfr_uv = gal_with_sfr[gal].logFUVSFR
            sfr_uv_type = gal_with_sfr[gal].logFUVSFR_type

            if (((not np.isnan(sfr_halpha)) and (sfr_halpha < lim_halpha)) and (sfr_halpha_type != '>')):
                #Does not fulfill criterion -> discard
                #print('Halpha',sfr_halpha,lim_halpha)
                discarded_gals += [gal]

            elif (((not np.isnan(sfr_uv)) and (sfr_uv < lim_uv)) and (sfr_uv_type != '>')):
                #Does not fulfill criterion -> discard
                #print('UV',sfr_uv,lim_uv)
                discarded_gals += [gal]
            else:
                #Fulfills SFR criterion -> select
                #print('SFR larger than both limits',gal)
                selected_gals += [gal]

        else:
            # galaxy cannot be rejected as no SFR is given -> select
            #print('no SFR given',gal)
            selected_gals += [gal]
    
    return selected_gals, discarded_gals