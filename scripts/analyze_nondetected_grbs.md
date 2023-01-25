---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from cosmogrb.universe.survey import Survey
import matplotlib.pyplot as plt
import popsynth as ps
import numpy as np
from grb_shader import RestoredMultiverse
%matplotlib widget
import cosmogrb
from gbmgeometry import *
from cosmogrb.utils.package_utils import get_path_of_data_file
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import astropy.units as u
from joblib import Parallel, delayed
from grb_shader.grb_analysis import GRBAnalysis_constant
```

# Hard flux selection (1e-7 erg cm^-2 s^-1)

```python
dir_hard_flux_selec = "/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/hardfluxselec/"
pop_hard_flux_selec = dir_hard_flux_selec + "pop_1234.h5"

pop = ps.Population.from_file(pop_hard_flux_selec)
print(f'Total number GRBs in Universe: \t{pop.n_objects}')
print(f'Number detected GRBs: \t\t{pop.n_detections}')
print(f'Number non-detected GRBs: \t{pop.n_non_detections}')

survey_file_hard_flux_selec = dir_hard_flux_selec + "survey_1234.h5"
survey_hard_flux_selec = Survey.from_file(survey_file_hard_flux_selec)
#names of GRBs that were selected by hard flux selection:
grb_names_hard_flux_selec = survey_hard_flux_selec.names_grbs
#print(f'Detected GRBs: \t{grb_names_hard_flux_selec}')
```

```python
dir_sim = "/data/eschoe/grb_shader/sims/220902/noselec_const_no_trunc/"
pop_file_no_selec_pulse = dir_sim + "pop_1234.h5"

survey_file_no_selec_pulse = dir_sim + 'survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec_pulse = Survey.from_file(survey_file_no_selec_pulse)
survey_no_selec_pulse.info()

det_grbs_hard_flux_selec = len(set(survey_no_selec_pulse.names_detected_grbs) & set(grb_names_hard_flux_selec))
print(f'Number GRBs selected by hard flux selection and GBM Triggter \t: {det_grbs_hard_flux_selec}')
print(f'Number GRBs not selected by hard flux selection but selected by GBM Trigger: {survey_no_selec_pulse.n_detected-det_grbs_hard_flux_selec}')

#files of non detected GRBs
non_det_GRBs_no_selec_pulse = survey_no_selec_pulse.names_grbs[~survey_no_selec_pulse.mask_detected_grbs]
det_GRBs_no_selec_pulse = survey_no_selec_pulse.names_grbs[survey_no_selec_pulse.mask_detected_grbs]
```

## Compute difference in location of GRBs and spacecraft


### non-detected GRBs

```python
posthist_file_cosmogrb = get_path_of_data_file('posthist.h5')
pi = PositionInterpolator.from_poshist_hdf5(posthist_file_cosmogrb)

min_sep_angles = np.zeros(len(non_det_GRBs_no_selec_pulse))
is_occulted = np.zeros(len(non_det_GRBs_no_selec_pulse))

for i in tqdm(range(len(min_sep_angles))):
    chosen_grb = non_det_GRBs_no_selec_pulse[i]
    
    time_adjustment = survey_no_selec_pulse[chosen_grb].grb.time_adjustment
    myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)
    
    ra = survey_no_selec_pulse[chosen_grb].grb.ra
    dec = survey_no_selec_pulse[chosen_grb].grb.dec
    grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')
    
    is_occulted[i] = pi.is_earth_occulted(ra=ra,dec=dec,t=time_adjustment)
    
    min_sep_angles[i] = min(np.array(myGBM.get_separation(grb)))
```

### Detected GRBs

```python
min_sep_angles_det = np.zeros(len(det_GRBs_no_selec_pulse))
is_occulted_det = np.zeros(len(det_GRBs_no_selec_pulse))

for i in tqdm(range(len(min_sep_angles_det))):
    chosen_grb = det_GRBs_no_selec_pulse[i]
    
    time_adjustment = survey_no_selec_pulse[chosen_grb].grb.time_adjustment
    myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)
    
    ra = survey_no_selec_pulse[chosen_grb].grb.ra
    dec = survey_no_selec_pulse[chosen_grb].grb.dec
    grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')
    
    is_occulted_det[i] = pi.is_earth_occulted(ra=ra,dec=dec,t=time_adjustment)
    
    min_sep_angles_det[i] = min(np.array(myGBM.get_separation(grb)))
```

```python
is_occulted_det = is_occulted_det.astype(bool)
```

### Plot histogram separation angle

```python
is_occulted = is_occulted.astype(bool)
fig, ax = plt.subplots()
ax.hist(min_sep_angles,bins=40,alpha=0.2,label='all non-detected',color='C00')
ax.hist(min_sep_angles[~is_occulted],bins=40,label='not occulted by Earth',alpha=0.3,color='C00')
ax.hist(min_sep_angles[is_occulted],bins=40,label='occulted by Earth',alpha=0.5,color='C00')

ax.hist(min_sep_angles_det,bins=40,label='all detected',alpha=0.5,color='C01')
ax.hist(min_sep_angles_det[~is_occulted_det],bins=40,label='not occulted by Earth',alpha=0.7,color='C01')
ax.hist(min_sep_angles_det[is_occulted_det],bins=40,label='all detected',alpha=1,color='C01')

ax.legend()
ax.set_xlabel('Minimum separation angle')
plt.savefig('/data/eschoe/grb_shader/figs/sep_angles_const.png',dpi=300)
plt.show()
```

```python
print(f'Number of non-detected GRBs being occulted by sun: {int(sum(is_occulted))} =\
 {np.round(int(sum(is_occulted))/len(non_det_GRBs_no_selec_pulse)*100,2)}% of all \
 non-detected GRBs')

print(f'Number of detected GRBs being occulted by sun: {int(sum(is_occulted_det))} =\
 {np.round(int(sum(is_occulted_det))/len(det_GRBs_no_selec_pulse)*100,2)}% of all \
detected GRBs')
```

## Plot direction of rays

```python
chosen_grb = 'SynthGRB_11'
print(chosen_grb)
    
time_adjustment = survey_no_selec_pulse[chosen_grb].grb.time_adjustment
myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)

ra = survey_no_selec_pulse[chosen_grb].grb.ra
dec = survey_no_selec_pulse[chosen_grb].grb.dec
grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')

min_sep_angles[i] = min(np.array(myGBM.get_separation(grb)))

f = Fermi(quaternion = pi.quaternion(time_adjustment), sc_pos = pi.sc_pos(time_adjustment)*u.m)
f.add_ray(ray_coordinate=grb)
f.compute_intersections()

f.plot_fermi(color_dets_different=True, plot_det_label=False, with_intersections=True, with_rays=True);
```

# GBM Trigger Selection - Constant Profile

```python
#dir_sim = "/data/eschoe/grb_shader/sims/220902/noselec_const/"
#pop_file_no_selec_pulse = dir_sim + "pop_1234.h5"
#
#survey_file_no_selec_pulse = dir_sim + 'survey_1234.h5'
##selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger
#
#survey_no_selec_pulse = Survey.from_file(survey_file_no_selec_pulse)
#survey_no_selec_pulse.info()
#
#det_grbs_hard_flux_selec = len(set(survey_no_selec_pulse.names_detected_grbs) & set(grb_names_hard_flux_selec))
#print(f'Number GRBs selected by hard flux selection and GBM Triggter \t: {det_grbs_hard_flux_selec}')
#print(f'Number GRBs not selected by hard flux selection but selected by GBM Trigger: {survey_no_selec_pulse.n_detected-det_grbs_hard_flux_selec}')
#
##files of non detected GRBs
#non_det_GRBs_no_selec_pulse = survey_no_selec_pulse.names_grbs[~survey_no_selec_pulse.mask_detected_grbs]
#det_GRBs_no_selec_pulse = survey_no_selec_pulse.names_grbs[survey_no_selec_pulse.mask_detected_grbs]
```

```python
dir_sim = "/data/eschoe/grb_shader/sims/220921/noselec_const_no_trunc/"
survey_file_no_selec_const = dir_sim + 'survey_1234.h5'

survey_no_selec_const = Survey.from_file(survey_file_no_selec_const)
non_det_grbs = survey_no_selec_const.names_grbs[~survey_no_selec_pulse.mask_detected_grbs]
survey_no_selec_const.info()
```

```python
det_grbs = survey_no_selec_const.names_grbs[survey_no_selec_const.mask_detected_grbs]
```

```python
chosen_grb_ind = 10
grb1 = GRBAnalysis_constant(
    dir_sim = dir_sim, 
    name_chosen_grb = non_det_grbs[chosen_grb_ind],
    pulse_profile=False)
```

## plot simulated light curve of non_detected

```python
grb1.plot_photon_lc();
```

```python
chosen_grb_ind = 18
grb2 = GRBAnalysis_constant(
    dir_sim = dir_sim, 
    name_chosen_grb = non_det_grbs[chosen_grb_ind],
    pulse_profile=False)
grb2.plot_photon_lc();
```

## Compute difference in location of GRBs and spacecraft


### Non-detected GRBs

```python
posthist_file_cosmogrb = get_path_of_data_file('posthist.h5')
pi = PositionInterpolator.from_poshist_hdf5(posthist_file_cosmogrb)

min_sep_angles = np.zeros(len(non_det_grbs))
is_occulted = np.zeros(len(non_det_grbs))

for i in tqdm(range(len(min_sep_angles))):
    chosen_grb = non_det_grbs[i]
    
    time_adjustment = survey_no_selec_const[chosen_grb].grb.time_adjustment
    myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)
    
    ra = survey_no_selec_const[chosen_grb].grb.ra
    dec = survey_no_selec_const[chosen_grb].grb.dec
    grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')
    
    is_occulted[i] = pi.is_earth_occulted(ra=ra,dec=dec,t=time_adjustment)
    
    min_sep_angles[i] = min(np.array(myGBM.get_separation(grb)))
```

### Detected GRBs

```python
min_sep_angles_det = np.zeros(len(det_grbs))
is_occulted_det = np.zeros(len(det_grbs))

for i in tqdm(range(len(min_sep_angles_det))):
    chosen_grb = det_grbs[i]
    
    time_adjustment = survey_no_selec_const[chosen_grb].grb.time_adjustment
    myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)
    
    ra = survey_no_selec_const[chosen_grb].grb.ra
    dec = survey_no_selec_const[chosen_grb].grb.dec
    grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')
    
    is_occulted_det[i] = pi.is_earth_occulted(ra=ra,dec=dec,t=time_adjustment)
    
    min_sep_angles_det[i] = min(np.array(myGBM.get_separation(grb)))
```

### Plot separation angle

```python
is_occulted = is_occulted.astype(bool)
is_occulted_det =is_occulted_det.astype(bool)
fig, ax = plt.subplots()
ax.hist(min_sep_angles,bins=40,alpha=0.2,label='all non-detected',color='C00')
ax.hist(min_sep_angles[~is_occulted],bins=40,label='not occulted by Earth',alpha=0.3,color='C00')
ax.hist(min_sep_angles[is_occulted],bins=40,label='occulted by Earth',alpha=0.5,color='C00')

ax.hist(min_sep_angles_det,bins=40,label='all detected',alpha=0.5,color='C01')
ax.hist(min_sep_angles_det[~is_occulted_det],bins=40,label='not occulted by Earth',alpha=0.7,color='C01')
ax.hist(min_sep_angles_det[is_occulted_det],bins=40,label='all detected',alpha=1,color='C01')

ax.legend()
ax.set_xlabel('Minimum separation angle')
plt.savefig('/data/eschoe/grb_shader/figs/sep_angles_const.png',dpi=300)
plt.show()
```

# Study spectra

```python

```
