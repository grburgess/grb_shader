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

# Imports

```python
from grb_shader.grb_analysis import GRBAnalysis_constant
from cosmogrb.universe.survey import Survey
from cosmogrb import grbsave_to_gbm_fits
%matplotlib widget
import matplotlib.pyplot as plt
from gbmgeometry import *
from cosmogrb.utils.package_utils import get_path_of_data_file
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from pathlib import Path
from threeML import *
```

<!-- #region tags=[] -->
# Pulse Profile GRB
<!-- #endregion -->

```python tags=[]
dir_sim2 = '/data/eschoe/grb_shader/sims/221016/c_noselec_pulse_noocc_inc/'
survey_file_no_selec_pulse2 = dir_sim2 + 'survey_1234.h5'

survey_no_selec_pulse2 = Survey.from_file(survey_file_no_selec_pulse2)
det_grbs2 = survey_no_selec_pulse2.names_detected_grbs
survey_no_selec_pulse2.info()
```

```python
grb1 = GRBAnalysis_constant(
    dir_sim = dir_sim2, 
    name_chosen_grb = det_grbs2[1],
    pulse_profile=True)
```

## Latent parameters

```python
grb1.chosen_grb
```

## Count light curve and spectrum

```python
grb1.plot_photon_lc(dt=.1);
```

```python
fig, ax = plt.subplots(5,3,sharex=False,sharey=False,figsize=(7,7))
grb1.plot_photon_spectrum(axes=ax);
ax[-1,-1].legend()
```

```python jupyter={"outputs_hidden": true} tags=[]
grb1.fit_cpl_ep()
```

```python
grb1.corner_plot()
```

# Constant GRBs

```python
dir_sim = "/data/eschoe/grb_shader/sims/220921/noselec_const_no_trunc/"
survey_file_no_selec_const = dir_sim + 'survey_1234.h5'

survey_no_selec_const = Survey.from_file(survey_file_no_selec_const)
det_grbs = survey_no_selec_const.names_detected_grbs
survey_no_selec_const.info()
#grb2_name = det_grbs[3]
```

```python
print(len(survey_no_selec_const.population.duration[survey_no_selec_const.population.duration>2]))
```

```python
#print(survey_no_selec_const.population.duration[survey_no_selec_const.mask_detected_grbs])
```

```python
chosen_grb_ind = 4
grb1 = GRBAnalysis_constant(
    dir_sim = dir_sim, 
    name_chosen_grb = det_grbs[chosen_grb_ind],
    pulse_profile=False)
```

## Latent parameters

```python
grb1.chosen_grb
```

## Plot count spectrum and light curve

```python
fig = grb1.plot_photon_lc(alpha=0.9);
ax = fig.get_axes()
#h, l = ax[0].get_legend_handles_labels()
#fig.legend(h,l,loc='lower right',bbox_to_anchor=[0.985, 0.084])
plt.tight_layout()
plt.savefig(f'/data/eschoe/grb_shader/figs/{grb1._name_chosen_grb}_photon_lc.pdf')
```

```python
fig = grb1.plot_photon_spectrum(alpha=0.9);
ax = fig.get_axes()
ax[0].set_ylim(1e-3,7)
ax[1].set_ylim(1e-3,7)
ax[0].set_xlim(200,40000)
ax[1].set_xlim(200,40000)
#h, l = ax[0].get_legend_handles_labels()
#fig.legend(h,l,loc='lower right',bbox_to_anchor=[0.985, 0.084])
#fig.get_legend().remove()

plt.savefig(f'/data/eschoe/grb_shader/figs/{grb1._name_chosen_grb}_photon_spectra.pdf')
```

## Closest detectors

```python
print(grb1.closest_dets)
```

## Orientation of detectors 

```python tags=[]
posthist_file_cosmogrb = get_path_of_data_file('posthist.h5')
pi = PositionInterpolator.from_poshist_hdf5(posthist_file_cosmogrb)

time_adjustment = survey_no_selec_const[grb1._name_chosen_grb].grb.time_adjustment
myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)

ra = survey_no_selec_const[grb1._name_chosen_grb].grb.ra
dec = survey_no_selec_const[grb1._name_chosen_grb].grb.dec
grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')

min_sep_angle = min(np.array(myGBM.get_separation(grb)))
print(min_sep_angle)
f = Fermi(quaternion = pi.quaternion(time_adjustment), sc_pos = pi.sc_pos(time_adjustment)*u.m)
f.add_ray(ray_coordinate=grb)
f.compute_intersections()

f.plot_fermi(color_dets_different=True, detectors=grb1.closest_dets,plot_det_label=True, with_intersections=True, with_rays=True);
```

<!-- #region tags=[] -->
## Fit Cutoffpowerlaw_Ep
<!-- #endregion -->

### convert to fits format

```python
#grb_file = Path(survey_no_selec_const.files_detected_grbs[0])
#destination = grb_file.parent
#grbsave_to_gbm_fits(survey_no_selec_const.files_detected_grbs[chosen_grb_ind],destination=destination)
```

```python
grb1.fit_cpl_ep(plot_lightcurve=True,
                  plot_count_spectrum=True,
                  savefigs=True,
                  dir_figs='/data/eschoe/grb_shader/figs/',
                  n_live_points=1000)

```

```python
grb1.bayes_results[0].results.get_highest_density_posterior_interval('ps.spectrum.main.Cutoff_powerlaw_Ep.xp')
```

```python
grb1.bayes_results[0].results.get_data_frame()['value']['ps.spectrum.main.Cutoff_powerlaw_Ep.xp']
```

```python
grb1.bayes_results[0].results.get_equal_tailed_interval('ps.spectrum.main.Cutoff_powerlaw_Ep.xp')
```

```python
fig = display_spectrum_model_counts(
    grb1.bayes_results[0],
    min_rate=[15.,15.0, 15.0, 15.0],
    data_colors=["C00", "C01", "C02","C03"],
    model_colors=["C00", "C01", "C02","C03"],
    show_background=False,
    source_only=True,
    step=False
)
ax=fig.get_axes()
h,l = ax[0].get_legend_handles_labels()
ax[0].legend(h,l,fontsize = 'small',loc='upper right')
ax[0].set_ylim(1e-4,10)
ax[0].set_xlim(None,2.5e4)
plt.savefig('/data/eschoe/grb_shader/figs/'+f'{grb1._name_chosen_grb}_median_fit_3ml2.pdf')
```

```python
#fig =grb1.plot_median_fit(savefig=True,
#                     dir_fig='/data/eschoe/grb_shader/figs',
#                     min_rate=15)
#plt.savefig('/data/eschoe/grb_shader/figs/grb1_bestfit_ppc.pdf')
```

```python
renamed_parameters = {'ps.spectrum.main.Cutoff_powerlaw_Ep.K': r'$K$',
                      'ps.spectrum.main.Cutoff_powerlaw_Ep.index':r'$\alpha$',
                     'ps.spectrum.main.Cutoff_powerlaw_Ep.xp':r'$E_p$'}

mpl.rc('xtick', labelsize=10) 
mpl.rc('ytick', labelsize=10) 
fig = grb1.corner_plot(renamed_parameters=renamed_parameters,savefig=True,dir_fig='/data/eschoe/grb_shader/figs');
ax = fig[0].get_axes()
for i in range(9):
    ax[i].tick_params(axis='both', which='major', pad=0.01)
plt.tight_layout()
plt.savefig(f'/data/eschoe/grb_shader/figs/{grb1._name_chosen_grb}_0_corner_plot_3ml.pdf')
```

## E_iso

```python
from popsynth.utils.cosmology import Cosmology
```

```python
c = Cosmology()
```

### from latent parameters

```python
flux = grb1.chosen_grb.source_params['peak_flux']
z = grb1.chosen_grb.z
duration = grb1.chosen_grb.duration
E_iso_exp = flux * duration * 4 * np.pi * (c.luminosity_distance(z))**2 /(1+z^2)

print(c.luminosity_distance(0.001)*3.24078e-25)
print(E_iso_exp)
```

```python
flux/((1+z)**2)
```

### from fitted spectrum

```python
results = grb1.bayes_results[0].results
```

```python
#gives flux in 68% cofidence interval
results.get_flux(ene_min=8*u.keV, 
                ene_max = 40*u.MeV,
                use_components=True,
                flux_unit="erg/(cm2 s)")['hi bound']['ps: total'].value
```

```python
from scipy.special import gamma, gammaincc

xp = 860
alpha = -0.76
K = 0.55

ec = xp / (2 + alpha)
a = 10 
b = 1e4
K = 0.55
# Cutoff power law

# get the intergrated flux, unit: keV^2

i1 = gammaincc(2.0 + alpha, a / ec) * gamma(2.0 + alpha)
i2 = gammaincc(2.0 + alpha, b / ec) * gamma(2.0 + alpha)

intflux = -ec * ec * (i2 - i1)
factor = (xp/(2+alpha))**(alpha)

print(K *factor* intflux*1.60218e-9) #erg
```

# repeat for other GRB

```python
chosen_grb_ind2 = 12
grb2_name = det_grbs[chosen_grb_ind2]
grb2 = GRBAnalysis_constant(dir_sim = dir_sim, name_chosen_grb = grb2_name)
```

```python
print(repr(grb2.chosen_grb))
#print('\nLatent spectral parameters\n\
#Ep: {},\nz: {},\nindex: {},\npeak_flux: {} keV/s/cm^2,\nK: {} cnts/s/cm^2/keV'.format(
#grb2.ep_latent,grb2.z_latent,grb2.alpha_latent,grb2.peak_flux_latent,grb2.K_exp))
```

```python
#grb_file = Path(survey_no_selec_const.files_detected_grbs[chosen_grb_ind2])
#destination = grb_file.parent
#grbsave_to_gbm_fits(survey_no_selec_const.files_detected_grbs[chosen_grb_ind2],destination=destination)
```

```python
grb2.fit_cpl_ep(plot_lightcurve=True,
                  plot_count_spectrum=True,
                  savefigs=False,
                  dir_figs=None,
                  n_live_points=1000)
```

```python
grb2.plot_median_fit(min_rate=10,savefig=True,dir_fig='/data/eschoe/grb_shader/figs')
```

```python
renamed_parameters = {'ps.spectrum.main.Cutoff_powerlaw_Ep.K': r'$K$',
                      'ps.spectrum.main.Cutoff_powerlaw_Ep.index':r'$\alpha$',
                     'ps.spectrum.main.Cutoff_powerlaw_Ep.xp':r'$E_p$'}

#dict(K=r'$K$',index=r'$\alpha$',xp=r'$E_p$')
fig = grb2.corner_plot(renamed_parameters=renamed_parameters,savefig=True,dir_fig='/data/eschoe/grb_shader/figs');
```

# one more GRB


## Choose GRB

```python
chosen_grb_ind3=1
grb3_name = det_grbs[chosen_grb_ind3]
grb3 = GRBAnalysis_constant(dir_sim = dir_sim, name_chosen_grb = grb3_name)
```

```python
grb3.chosen_grb.info
```

## Count spectra and light curves

```python
grb3.plot_photon_lc();
plt.savefig(f'/data/eschoe/grb_shader/figs/{grb3._name_chosen_grb}_photon_lc.pdf')
```

```python
fig = grb3.plot_photon_spectrum(alpha=0.9);
ax = fig.get_axes()
ax[0].set_ylim(1e-3,7)
ax[1].set_ylim(1e-3,7)
ax[0].set_xlim(200,40000)
ax[1].set_xlim(200,40000)
#h, l = ax[0].get_legend_handles_labels()
#fig.legend(h,l,loc='lower right',bbox_to_anchor=[0.985, 0.084])
#fig.get_legend().remove()

plt.savefig(f'/data/eschoe/grb_shader/figs/{grb3._name_chosen_grb}_photon_spectra.pdf')
```

## Closest detectors

```python
print(grb3.closest_dets)
```

## Orientation detectors

```python
posthist_file_cosmogrb = get_path_of_data_file('posthist.h5')
pi = PositionInterpolator.from_poshist_hdf5(posthist_file_cosmogrb)
time_adjustment = survey_no_selec_const[grb3_name].grb.time_adjustment
myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)

ra = survey_no_selec_const[grb3_name].grb.ra
dec = survey_no_selec_const[grb3_name].grb.dec
grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')

min_sep_angle = min(np.array(myGBM.get_separation(grb)))
print(min_sep_angle)
f = Fermi(quaternion = pi.quaternion(time_adjustment), sc_pos = pi.sc_pos(time_adjustment)*u.m)
f.add_ray(ray_coordinate=grb)
f.compute_intersections()

print(myGBM.get_separation(grb))

fig = f.plot_fermi(color_dets_different=True, detectors=grb3.closest_dets,plot_det_label=True, with_intersections=False, with_rays=True);
#f.plot_fermi(color_dets_different=True,plot_det_label=True, with_intersections=False, with_rays=True);


```

```python
ax = fig.get_axes()

```

## fit CPL

```python
#grb_file = Path(survey_no_selec_const.files_detected_grbs[0])
#destination = grb_file.parent
#grbsave_to_gbm_fits(survey_no_selec_const.files_detected_grbs[chosen_grb_ind3],destination=destination)
```

```python
grb3.fit_cpl_ep()
```

```python
print(repr(grb3.chosen_grb))
print('\nLatent spectral parameters\n\
Ep: {},\nz: {},\nindex: {},\npeak_flux: {} keV/s/cm^2,\nK: {} cnts/s/cm^2/keV'.format(
grb3.ep_latent,grb3.z_latent,grb3.alpha_latent,grb3.peak_flux_latent,grb3.K_exp))
```

```python
renamed_parameters = {'ps.spectrum.main.Cutoff_powerlaw_Ep.K': r'$K$',
                      'ps.spectrum.main.Cutoff_powerlaw_Ep.index':r'$\alpha$',
                     'ps.spectrum.main.Cutoff_powerlaw_Ep.xp':r'$E_p$'}

mpl.rc('xtick', labelsize=10) 
mpl.rc('ytick', labelsize=10) 
fig = grb3.corner_plot(renamed_parameters=renamed_parameters,savefig=True,dir_fig='/data/eschoe/grb_shader/figs');
ax = fig[0].get_axes()
for i in range(9):
    ax[i].tick_params(axis='both', which='major', pad=0.01)
plt.tight_layout()
plt.savefig(f'/data/eschoe/grb_shader/figs/{grb3._name_chosen_grb}_0_corner_plot_3ml.pdf')
```

```python
fig = display_spectrum_model_counts(
    grb3.bayes_results[0],
    min_rate=[15.,15.0, 15.0, 15.0],
    data_colors=["C00", "C01", "C02","C03"],
    model_colors=["C00", "C01", "C02","C03"],
    show_background=False,
    source_only=True,
    step=False
)
ax=fig.get_axes()
h,l = ax[0].get_legend_handles_labels()
ax[0].legend(h,l,fontsize = 'small',loc='upper right')
ax[0].set_ylim(1e-4,50)
ax[0].set_xlim(None,2.5e4)
plt.savefig('/data/eschoe/grb_shader/figs/'+f'{grb3._name_chosen_grb}_median_fit_3ml2.pdf')
```

```python

```
