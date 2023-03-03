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
from pathlib import Path
from tqdm import tqdm
from threeML.utils.OGIP.response import OGIPResponse
from threeML import *
```

# Hard flux selection


## popsynth selection: hard flux selection (1e-7 erg cm^-2 s^-1)

```python
dir_hard_flux_selec = "/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/hardfluxselec/"
```

```python
pop_hard_flux_selec = dir_hard_flux_selec + "pop_1234.h5"
```

```python
pop = ps.Population.from_file(pop_hard_flux_selec)
print(f'Total number GRBs in Universe: \t{pop.n_objects}')
print(f'Number detected GRBs: \t\t{pop.n_detections}')
print(f'Number non-detected GRBs: \t{pop.n_non_detections}')
```

```python
survey_file_hard_flux_selec = dir_hard_flux_selec + "survey_1234.h5"
survey_hard_flux_selec = Survey.from_file(survey_file_hard_flux_selec)
survey_hard_flux_selec.info()
grb_names_hard_flux_selec = survey_hard_flux_selec.names_grbs
```

## Convert h5 files to fits and rsp files

```python
## generate fits files for all detected GRBs:
#grb_file = Path(survey_hard_flux_selec.grb_save_files[0])
#destination = grb_file.parent
#for det_grb in survey_hard_flux_selec.grb_save_files:
#    cosmogrb.grbsave_to_gbm_fits(det_grb,destination=destination)
```

## Find GRB


Find a GRB that was the first detected by the GBM trigger as well as the hard flux selection

```python
found = False
i = 0
while found == False:
    grb_name = grb_names_hard_flux_selec[i]
    if survey_hard_flux_selec[grb_name].grb.z == 3.374404717911651:
        hard_flux_chosen_grb= grb_name
        print(grb_name)
        print(survey_hard_flux_selec[grb_name].grb.info)
        found = True
    i += 1
```

## Plot detected light curve

```python
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(10,10))
row=0
col = 0
for k,v  in survey_hard_flux_selec['SynthGRB_6'].grb.items():
    ax = axes[row][col]
    
    lightcurve =v['lightcurve']
    
    lightcurve.display_lightcurve(dt=.5, ax=ax,lw=1,color='#25C68C')
    lightcurve.display_source(dt=.5,ax=ax,lw=1,color="#A363DE")
    lightcurve.display_background(dt=.5,ax=ax,lw=1, color="#2C342E")
    ax.set_xlim(-10, 30)
    ax.set_title(k,size=8)
    
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0

axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
```

## Plot detected spectra

```python
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(10,10))
row=0
col = 0
for k,v  in survey_hard_flux_selec['SynthGRB_6'].grb.items():
    ax = axes[row][col]
    
    lightcurve =v['lightcurve']
    
    lightcurve.display_count_spectrum( ax=ax,lw=1,color='#25C68C')
    lightcurve.display_count_spectrum_source(ax=ax,lw=1,color="#A363DE",label='Source')
    lightcurve.display_count_spectrum_background(ax=ax,lw=1, color="#2C342E",label='Background')
    ax.set_title(k,size=8)
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0
axes[0,0].legend()

axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
```

## Analyze GRB and fit spectrum/ light curve with 3ml

```python
grb_folder = dir_hard_flux_selec + '_1234/'
response = OGIPResponse(grb_folder +f'rsp_{hard_flux_chosen_grb}_n3.rsp')
gbm_tte =  TimeSeriesBuilder.from_gbm_tte('n3_tte',tte_file=grb_folder +f'tte_{hard_flux_chosen_grb}_n3.fits',rsp_file=response)
```

```python

```

# GBM Trigger Selection - Constant temporal profile


## popsynth selection: no selection

```python
pop_file_no_selec = "/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/noselec/pop_1234.h5"
pop_no_selec = ps.Population.from_file(pop_file_no_selec)
print(f'Total number GRBs in Universe: \t{pop_no_selec.n_objects}')
print(f'Number detected GRBs: \t\t{pop_no_selec.n_detections}')
print(f'Number non-detected GRBs: \t{pop_no_selec.n_non_detections}')
```

## cosmogrb: processed using GBM trigger

```python
survey_file_no_selec = '/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/noselec/survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec = Survey.from_file(survey_file_no_selec)
survey_no_selec.info()
survey_no_selec.names_detected_grbs
```

## Convert detected GRBSave files to HDF5

```python
## generate fits files for all detected GRBs:
#grb_file = Path(survey_no_selec.files_detected_grbs[0])
#destination = grb_file.parent
#for i in tqdm(range(len(survey_no_selec.files_detected_grbs))):
#    cosmogrb.grbsave_to_gbm_fits(survey_no_selec.files_detected_grbs[i],destination=destination)
```

## Choose a GRB

```python
#choose a GRB:
chosen_grb = 'SynthGRB_12'
```

```python
survey_no_selec[chosen_grb].detector_info.info()
```

```python
print(survey_no_selec[chosen_grb].grb.z)
print(survey_no_selec[chosen_grb].grb._output())
print(survey_no_selec[chosen_grb].grb._output()['peak_flux'])
```

## plot latent light curve

```python
from cosmogrb.instruments.gbm.gbm_grb import GBMGRB_CPL_Constant
```

Simulate the chosen GRB's latent spectra and light curves

```python
gbmgrb = GBMGRB_CPL_Constant(name=chosen_grb, 
                             z = survey_no_selec[chosen_grb].grb.z,
                            T0 = survey_no_selec[chosen_grb].grb.T0,
                            ra = survey_no_selec[chosen_grb].grb.ra,
                            dec = survey_no_selec[chosen_grb].grb.dec,
                            duration = survey_no_selec[chosen_grb].grb.duration,
                            peak_flux = survey_no_selec[chosen_grb].grb._output()['peak_flux'],
                            alpha = survey_no_selec[chosen_grb].grb._output()['alpha'],
                            ep = survey_no_selec[chosen_grb].grb._output()['ep'])

```

```python
list(gbmgrb._lightcurves.values())
```

```python
gbmgrb.info
```

plot energy integrated light curve

```python
time = np.linspace(3, 1e4, 500)
gbmgrb.display_energy_integrated_light_curve(time, color="#A363DE")
```

```python
%matplotlib widget
energy = np.logspace(1, 3, 1000)

gbmgrb.display_energy_dependent_light_curve(time, energy, lw=.25, alpha=.5)
plt.show()
```

## plot detected light curve

```python
%matplotlib widget
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(10,10))
row=0
col = 0
for k,v  in survey_no_selec['SynthGRB_12'].grb.items():
    ax = axes[row][col]
    
    lightcurve =v['lightcurve']
    
    lightcurve.display_lightcurve(dt=.5, ax=ax,lw=1,color='#25C68C')
    lightcurve.display_source(dt=.5,ax=ax,lw=1,color="#A363DE")
    lightcurve.display_background(dt=.5,ax=ax,lw=1, color="#2C342E")
    print(lightcurve.time_adjustment)
    ax.set_xlim(-10, 30)
    ax.set_title(k,size=8)
    
    
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0

axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
plt.tight_layout()
```

## Plot detected spectra

```python
%matplotlib widget
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(10,10))
row=0
col = 0

lightcurves = []

for k,v  in survey_no_selec['SynthGRB_12'].grb.items():
    ax = axes[row][col]
    
    lightcurve = v['lightcurve']
    
    lightcurve.display_count_spectrum(tmin=-1,tmax=0,ax=ax,lw=1,color='#25C68C')
    lightcurve.display_count_spectrum_source(tmin=-1,tmax=0,ax=ax,lw=1,color="#A363DE",label='Source')
    lightcurve.display_count_spectrum_background(tmin=-1,tmax=0,ax=ax,lw=1, color="#2C342E",label='Background')
    #ax.set_xlim(10, 30)
    ax.set_title(k,size=8)
    
    lightcurves.append(lightcurve)
    
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0

axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
axes[0][0].legend()
plt.tight_layout()
```

## Analyze GRB

```python
grb_folder = str(Path(survey_file_no_selec).parent) + '/_1234/'
print(grb_folder)
response = OGIPResponse(grb_folder +f'rsp_{chosen_grb}_n7.rsp')
gbm_tte =  TimeSeriesBuilder.from_gbm_tte('n3_tte',tte_file=grb_folder +f'tte_{chosen_grb}_n7.fits',rsp_file=response)
```

```python
ts_tte.set_active_time_interval('source_interval')
```

# GBM Trigger Selection - Pulse Profile


## popsynth selection: no selection

```python
dir_sim = "/data/eschoe/grb_shader/sims/221016/c_noselec_pulse_noocc/"
pop_file_no_selec_pulse = dir_sim + "pop_1234.h5"
pop_no_selec_pulse = ps.Population.from_file(pop_file_no_selec_pulse)
print(f'Total number GRBs in Universe: \t{pop_no_selec_pulse.n_objects}')
print(f'Number detected GRBs: \t\t{pop_no_selec_pulse.n_detections}')
print(f'Number non-detected GRBs: \t{pop_no_selec_pulse.n_non_detections}')
```

## cosmogrb: processed using GBM trigger

```python
survey_file_no_selec_pulse = dir_sim + 'survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec_pulse = Survey.from_file(survey_file_no_selec_pulse)
survey_no_selec_pulse.info()
print(survey_no_selec_pulse.names_detected_grbs)
chosen_grb = 'SynthGRB_4'
```

## Plot latent light curve and spectra

```python
from cosmogrb.instruments.gbm.gbm_grb import GBMGRB_CPL
```

```python
survey_no_selec_pulse[chosen_grb].grb._output()
```

```python
gbmgrb2 = GBMGRB_CPL(name=chosen_grb, 
                             z = survey_no_selec_pulse[chosen_grb].grb.z,
                            T0 = survey_no_selec_pulse[chosen_grb].grb.T0,
                            ra = survey_no_selec_pulse[chosen_grb].grb.ra,
                            dec = survey_no_selec_pulse[chosen_grb].grb.dec,
                            duration = survey_no_selec_pulse[chosen_grb].grb.duration,
                            peak_flux = survey_no_selec_pulse[chosen_grb].grb._output()['peak_flux'],
                            alpha = survey_no_selec_pulse[chosen_grb].grb._output()['alpha'],
                            ep_start = survey_no_selec_pulse[chosen_grb].grb._output()['ep_start'],
                            ep_tau = survey_no_selec_pulse[chosen_grb].grb._output()['ep_tau'],
                            trise = survey_no_selec_pulse[chosen_grb].grb._output()['trise'],
                            tdecay = survey_no_selec_pulse[chosen_grb].grb._output()['tdecay'],
                   )


```

```python
gbmgrb2.info
```

```python
from cosmogrb.sampler.cpl_functions import cpl_evolution,norris

z = gbmgrb2.z
f =gbmgrb2.peak_flux
ep_start = gbmgrb2.ep_start
ep_tau = gbmgrb2.ep_tau
alpha=gbmgrb2.alpha
trise = gbmgrb2.trise
tdecay = gbmgrb2.tdecay
erg2keV = 6.24151e8

print('Peak flux in keV/s/cm^2',f*6.24151e8)
print('Duration in s',gbmgrb2.duration)
print('Redshift',z)

time = np.linspace(0.001, 2, 1000)
energy = np.logspace(1, 3, 1000)

fig, ax = plt.subplots()
out = cpl_evolution(energy, time, f, ep_start, ep_tau,alpha,trise,tdecay,
    10,50,z)
energy_integrated_flux = np.zeros(len(time))
norris_arr = np.zeros(len(time))
for i in range(len(time)):
    energy_integrated_flux[i] = np.trapz(energy*(1+z)**2*out[i,:],energy)
    norris_arr[i] = erg2keV * norris(time[i], K=f, t_start=0.0, t_rise=trise, t_decay=tdecay)
ax.plot(time,energy_integrated_flux)
#ax.plot(time,norris_arr)
#ax.plot(time,out[:,50])
```

```python
gbmgrb2.peak_flux*erg2keV
```

```python
time = np.linspace(0.001, 4, 500)
gbmgrb2.display_energy_integrated_light_curve(time, color="#A363DE")
```

```python
energy = np.logspace(-3, 5, 30)
time = np.linspace(0.001, 0.5, 500)
fig = gbmgrb2.display_energy_dependent_light_curve(time, energy, uselog=True,cmap='plasma_r', lw=2, alpha=.8)
```

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

```python
fig,(ax,ax2) = plt.subplots(1,2,figsize=(10,5))
time = np.geomspace(0.001, 0.5, 1000)
energy = np.geomspace(8,800,500)

gbmgrb2.display_energy_integrated_light_curve(time,ax=ax)
time = np.geomspace(0.05, 0.5, 30)
#gbmgrb2.display_energy_dependent_light_curve(time, energy, uselog=True,ax=ax2,cmap='viridis', alpha=.8)

ax2.set_ylim(1e-3,0.6)

time = np.linspace(0.02, 0.5,10)
fig = gbmgrb2.display_time_dependent_spectrum(time, energy, ax=ax2,cmap='viridis', lw=1, alpha=.5)
#plt.savefig('/data/eschoe/grb_shader/figs/latent_time_dependent_spectrum')
```

## plot detected light curve

```python
%matplotlib widget
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(9,9))
row=0
col = 0
for k,v  in survey_no_selec_pulse[chosen_grb].grb.items():
    ax = axes[row][col]
    
    lightcurve =v['lightcurve']
    lightcurve.display_lightcurve(dt=.5, ax=ax,lw=1,color='#25C68C',label='Full')
    lightcurve.display_source(dt=.5,ax=ax,lw=1,color="#A363DE",label='Source')
    lightcurve.display_background(dt=.5,ax=ax,lw=1, color="#2C342E",label='Background')
    ax.set_xlim(-10, 30)
    ax.set_title(k,size=8)
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0
    
axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
axes[0][0].legend()
plt.tight_layout()
```

## Plot detected spectra

```python
%matplotlib widget
fig, axes = plt.subplots(4,4,sharex=True,sharey=False,figsize=(10,10))
row=0
col = 0

lightcurves = []

for k,v  in survey_no_selec_pulse[chosen_grb].grb.items():
    ax = axes[row][col]
    
    lightcurve = v['lightcurve']
    
    lightcurve.display_count_spectrum(tmin=-1,tmax=4,ax=ax,lw=1,color='#25C68C')
    lightcurve.display_count_spectrum_source(tmin=-1,tmax=4,ax=ax,lw=1,color="#A363DE",label='Source')
    lightcurve.display_count_spectrum_background(tmin=-1,tmax=4,ax=ax,lw=1, color="#2C342E",label='Background')
    #ax.set_xlim(10, 30)
    ax.set_title(k,size=8)
    
    lightcurves.append(lightcurve)
    
    
    if col < 3:
        col+=1
    else:
        row+=1
        col=0

axes[3,2].set_visible(False)  
axes[3,3].set_visible(False)
axes[0][0].legend()
plt.tight_layout()
```

## Analyze GRB

```python
grb_folder_pulse = str(Path(survey_file_no_selec_pulse).parent) + '/_1234/'
response = OGIPResponse(grb_folder_pulse +f'rsp_{chosen_grb}_n8.rsp')
gbm_tte3 =  TimeSeriesBuilder.from_gbm_tte('n8_tte',tte_file=grb_folder_pulse +f'tte_{chosen_grb}_n8.fits',rsp_file=response)
```

```python
gbm_tte3.set_active_time_interval('0-5')
gbm_tte3.set_background_interval('20-30')
```

```python
gbm_tte3.view_lightcurve(-10,10);
```

```python
fluence_plugin = gbm_tte3.to_spectrumlike()
fluence_plugin.set_active_measurements("9-900")
```

```python
fit_function = Cutoff_powerlaw(K=1e-3, xc=1000, index=-0.66)
#Cutoff_powerlaw.
point_source = PointSource("ps", 0, 0, spectral_shape=fit_function)
model = Model(point_source)
```

```python
model.ps.spectrum.main.Cutoff_powerlaw.K.prior = Truncated_gaussian(
    lower_bound=1e-5, upper_bound=100, mu=1e-3, sigma=0.5
)
model.ps.spectrum.main.Cutoff_powerlaw.index.prior = Truncated_gaussian(
    lower_bound=-5, upper_bound=0, mu=-0.66, sigma=0.5
)
model.ps.spectrum.main.Cutoff_powerlaw.piv.prior = Truncated_gaussian(
    lower_bound=1e-5, upper_bound=100, mu=1, sigma=0.5
)

model.ps.spectrum.main.Cutoff_powerlaw.xc.prior = Truncated_gaussian(
    lower_bound=0, upper_bound=1e5, mu=1e3, sigma=0.1
)
```

```python
bayes = BayesianAnalysis(model, DataList(fluence_plugin))
```

```python
bayes.set_sampler("multinest", share_spectrum=True)
```

```python
#fig = display_spectrum_model_counts(bayes, min_rate=5, step=False)
```

```python
bayes.sampler.setup(n_live_points=400)
bayes.sample()
```

```python
bayes.restore_median_fit()
fig = display_spectrum_model_counts(bayes)
```

# Truncated t90

```python
dir_sim = "/data/eschoe/grb_shader/sims/220822/ghirl_c_t90fit_pulse/noselec/"
```
