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
from cosmogrb import GRBSave
from cosmogrb.universe.survey import Survey
from cosmogrb import grbsave_to_gbm_fits

from popsynth import silence_warnings
silence_warnings()

import collections

from tqdm.auto import tqdm

from pathlib import Path

import pandas as pd 

import numpy as np

%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from grb_shader import RestoredMultiverse, LocalVolume
from grb_shader.selections import *
from grb_shader.plotting.minor_symlog_locator import MinorSymLogLocator
from grb_shader.utils.package_data import get_path_of_data_file
from grb_shader.plotting.plotting_functions import array_to_cmap
from grb_shader.grb_analysis import GRBAnalysis_constant
from grb_shader.plotting.plotting_functions import logbins_scaled_histogram,logbins_norm_histogram
```

# Restore simulations

```python
sim_path = '/data/eschoe/grb_shader/sims/221001/catalogselec_const_no_trunc_occ/'

restored_sim = RestoredMultiverse(sim_path)
print(f'Number of simulated universes: {restored_sim.n_universes}')
```

```python
print('Number detected GRBs',int(np.sum(restored_sim._n_detected_grbs)))
print('Number detected GRBs',int(np.sum(restored_sim.n_detected_grbs)))
```

# Convert detected GRBs to fits format

```python
#create fits files for all detected GRBs:
#for i in tqdm(range(len(restored_sim.surveys))):
#    for j in range(len(restored_sim.surveys[i].files_detected_grbs)):
#        grb_file = Path(restored_sim.surveys[i].files_detected_grbs[j])
#        destination = grb_file.parent
#        grbsave_to_gbm_fits(grb_file,destination=destination)
```

#  Fit spectra
Add results to a summarizing csv file

```python
lv = LocalVolume.from_lv_catalog()
```

```python
from csv import DictWriter
dict_names = ['universe_seed', 'GRB_name', 'hit_galaxy_name',
               'z', 'Eiso_latent','Eiso_lv','Eiso_lv_lower','Eiso_lv_upper',
               'Eiso_z','Eiso_z_lower','Eiso_z_upper',
              'Ep_obs_latent','Ep_obs_fit','Ep_obs_fit_lower','Ep_obs_fit_upper',
               'duration_latent','duration_bb']
```

```python
bayesblock_failed = []
for i in tqdm(range(845,len(restored_sim.surveys))):
    survey = restored_sim.surveys[i]
    if len(survey.files_detected_grbs) >= 1:
        lv.read_population(restored_sim.populations[i],unc_angle=0.)
        lv_selected_galaxies_detected = np.array(lv.selected_galaxies)[survey.mask_detected_grbs]
    for j in range(0,len(survey.files_detected_grbs)):
        with open(csv_file, 'a') as f:
            dict_i = {}
            
            grb_file = Path(restored_sim.surveys[i].files_detected_grbs[j])
            print(grb_file)
            
            dict_i['universe_seed'] = str(grb_file.parent.relative_to(sim_dir))
            dict_i['GRB_name'] = str(grb_file.stem).strip('_store')
            dict_i['hit_galaxy_name'] = lv_selected_galaxies_detected[j][0].name
            
            grb = GRBAnalysis_constant(
                dir_sim=sim_dir,
                name_chosen_grb=str(grb_file.stem).strip('_store'),
                pulse_profile=False,
                pop_file = restored_sim.population_files[i],
                survey_file = restored_sim.survey_files[i],
                grb_folder = str(grb_file.parent.relative_to(sim_dir))
            )
            #grb._find_time_intervals(.1)
            #print(grb.duration_latent)
            #print(grb._start_times)
            #print(grb._stop_times)
            #grb.plot_photon_lc()
            
            dict_i['z'] = grb.z_latent
            dict_i['duration_latent'] = grb.duration_latent
            dict_i['Eiso_latent'] = grb.Eiso_latent
            dict_i['Ep_obs_latent'] = grb.ep_latent()/(1+grb.z_latent)
            
            #print(grb._stop_times)
            
            flag = grb.fit_cpl_ep(n_live_points=500,p0=.05)
            #flag=False
            if flag:
            
                dict_i['Eiso_z'] = grb.Eiso_fromspectralfit_at_z()
                dict_i['Eiso_z_lower'] = grb.Eiso_fromspectralfit_at_z(lower=True)
                dict_i['Eiso_z_upper'] = grb.Eiso_fromspectralfit_at_z(upper=True)

                d = lv_selected_galaxies_detected[j][0].distance

                dict_i['Eiso_lv'] = grb.Eiso_fromspectralfit_at_d(d)
                dict_i['Eiso_lv_lower'] = grb.Eiso_fromspectralfit_at_d(d,lower=True)
                dict_i['Eiso_lv_upper'] = grb.Eiso_fromspectralfit_at_d(d,upper=True)

                dict_i['duration_bb'] = np.sum(np.array(grb.duration_bayesian_blocks))

                dict_i['Ep_obs_fit'] = grb.Ep_bestfit()
                dict_i['Ep_obs_fit_lower'] = grb.Ep_bestfit(lower=True)
                dict_i['Ep_obs_fit_upper'] = grb.Ep_bestfit(upper=True)

                dictwriter_object = DictWriter(f, fieldnames=dict_names,delimiter='\t')

                dictwriter_object.writerow(dict_i)
            else:
                bayesblock_failed+=[grb_file]
```

# Select GRBs

```python
sim_dir = Path('/data/eschoe/grb_shader/sims/221001/catalogselec_const_no_trunc_occ/')
csv_file = sim_dir / 'summary.csv'
data = pd.read_csv(csv_file)
```

## SFR Selection

```python
hit_galaxies = restored_sim._galaxynames_hit
hit_and_detected_galaxies = restored_sim._galaxynames_hit_and_detected
#Galaxy names that are hit by GRBs, are detected and analyzable 
# (analyzable = three or more found time intervals by Bayesian blocks)
hit_analyzable_gals = np.unique(data['gal'])
```

```python
#define thresholds for sfr as minimum SFRs of given hit galaxies from above
lim_halpha = lv.galaxies_with_sfr['MESSIER031'].logSFR
lim_uv = lv.galaxies_with_sfr['MESSIER031'].logFUVSFR

hit_galaxies_sfr,hit_galaxies_sfr_excl = select_gals_sfr(hit_galaxies,lim_halpha,lim_uv)
hit_and_detected_galaxies_sfr,_ = select_gals_sfr(hit_and_detected_galaxies,lim_halpha,lim_uv)
hit_analyzable_gals_sfr,_ = select_gals_sfr(hit_analyzable_gals,lim_halpha,lim_uv)
```

```python
print('Number hit unique galaxies: ',len(hit_galaxies))
print('With SFR constrain: ',len(hit_galaxies_sfr))
print()
print('Number hit+detected unique galaxies: ',len(hit_and_detected_galaxies))
print('With SFR constrain: ',len(hit_and_detected_galaxies_sfr))
print()
print('Number hit+detected+analyzable unique galaxies: ',len(hit_analyzable_gals))
print('With SFR constrain: ',len(hit_analyzable_gals_sfr))

```

```python
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)
plt.rc('legend', fontsize=10)

fig, ax = plt.subplots(1,2,figsize=(9,6),gridspec_kw={'width_ratios': [1, 2.5]})
width=0.7
fig = restored_sim.hist_galaxies(n=30,ax=ax[0],label='Hit',width=width,fill=False,edgecolor='C00')
n=30
exclude=hit_galaxies_sfr_excl
hits = restored_sim._galaxynames_hit.most_common(n)
names = [x[0] for x in hits]
counts = []
for x in hits:
    if x[0] not in exclude:
        counts += [x[1]]
    else:
        counts += [0]
x = np.arange(len(names))  # the label locations

ax[0].barh(x, counts, width,color='C00',alpha=0.8)
ax[0].set_axisbelow(True)
ax[0].set_xlim(0,1e4)

hit_gal_names = list(restored_sim._galaxynames_hit.keys())

hit_gal_n = list(restored_sim._galaxynames_hit.values())

hit_gal_dist = np.zeros(len(hit_gal_names))
hit_gal_area = np.zeros(len(hit_gal_names))

for i, name in enumerate(hit_gal_names):
    hit_gal_dist[i] = restored_sim._catalog.galaxies[name].distance
    hit_gal_area[i] = restored_sim._catalog.galaxies[name].area

n,distances,areas = hit_gal_n,hit_gal_dist,hit_gal_area    

uselog=True
cmap='viridis'
_, colors = array_to_cmap(n, cmap=cmap, use_log=uselog)
    
if uselog:
    norm = mpl.colors.LogNorm(vmin=min(n), vmax=max(n))
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
else:
    norm = mpl.colors.Normalize(vmin=min(n), vmax=max(n))

#Add an axis for colorbar on right
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax, label='# Spatial Coincidences')

for i in range(len(distances)):
    if hit_gal_names[i] not in exclude:
        ax[1].scatter(x=float(distances[i]), y=float(areas[i]),color=colors[i],s=30,alpha=0.9,zorder=20)
    else:
        ax[1].scatter(x=float(distances[i]), y=float(areas[i]),facecolors='none', edgecolors=colors[i],s=30,alpha=0.9)

ax[1].set_xlabel('Distance [Mpc]')
ax[1].set_ylabel(r'Angular Area [rad$^2$]')

plt.savefig('/data/eschoe/grb_shader/figs/coincidences_gals_sfr2.png',dpi=300)
```

## GBM Trigger Selection


### Only GBM trigger

```python
print('WITH SagdSph')
n_hit_det=len(restored_sim.n_detected_grbs[restored_sim.n_detected_grbs>0])
print('\tNumber of surveys with at least 1 coinciding detected GRB',n_hit_det)

print('\tNumber of years until prob>5%:',5*14/(n_hit_det/10.), ' yr')

################
print('WITHOUT SagdSph')
print('\tNumber of surveys with at least 1 coinciding detected GRB',n_hit_det)
5*14/(n_coinc_det_sfr_woSagdSph/10.)
```

```python
n_coinc_det_sfr_tot =0
n_coinc_det_sfr_woSagdSph_tot =0
n_coinc_det_woSagdSph_tot = 0
n_coinc_det_tot = 0
n_coinc_woSagdSph_tot = 0
n_coinc_tot = 0

n_coinc_woSagdSph_arr = np.zeros(len(restored_sim._populations))

for j, pop in enumerate(tqdm(restored_sim._populations, desc="counting galaxies")):
    survey = restored_sim._surveys[j]
    restored_sim._catalog.read_population(pop,unc_angle=0.)

    #Number of selected galaxies has to be same as number of simulated GRBs
    assert len(restored_sim._catalog.selected_galaxies) == survey.n_grbs
    
    n_coinc_det_sfr =0
    n_coinc_det_sfr_woSagdSph =0
    n_coinc_det_woSagdSph = 0
    n_coinc_det = 0
    n_coinc_woSagdSph = 0
    n_coinc = 0

    for i,galaxy in enumerate(restored_sim._catalog.selected_galaxies):

        if survey.mask_detected_grbs[i]:
            #print(survey.mask_detected_grbs)
            n_coinc_det+=1
            if galaxy[0].name != 'SagdSph':
                n_coinc_det_woSagdSph += 1
                if galaxy[0].name not in hit_galaxies_sfr_excl:
                    n_coinc_det_sfr_woSagdSph += 1
            elif galaxy[0].name not in hit_galaxies_sfr_excl:
                n_coinc_det_sfr += 1


        if galaxy[0].name != 'SagdSph':
            n_coinc_woSagdSph += 1
    
    n_coinc_woSagdSph_arr[j] = n_coinc_woSagdSph
    #count number of surveys, not number of events 
    #-> donot count twice if one survey has two hits   
    if n_coinc_det > 0:
        n_coinc_det_tot += 1
    if n_coinc_det_woSagdSph > 0:
        n_coinc_det_woSagdSph_tot += 1
    if n_coinc_det_sfr_woSagdSph > 0:
        n_coinc_det_sfr_woSagdSph_tot += 1
    if n_coinc_det_sfr > 0:
        n_coinc_det_sfr_tot += 1
    if n_coinc > 0:
        n_coinc_tot += 1
    if n_coinc_woSagdSph > 0:
        n_coinc_woSagdSph_tot += 1
```

```python
print('ONLY GBM SELECTION')
#n_hit_det=len(restored_sim.n_detected_grbs[restored_sim.n_detected_grbs>0])
#print(n_hit_det)
print('\tWITH SagdSph')
print('\t\tN surveys with det sGRBs: ',n_coinc_det_tot)
print('\t\tNumber of years until prob>5%:',5*14/(n_coinc_det_tot/10.), ' yr')
print()
print('\tWITHOUT SagdSph')
print('\t\tN surveys with det sGRBs: ',n_coinc_det_woSagdSph_tot)
print('\t\tNumber of years until prob>5%:',5*14/(n_coinc_det_woSagdSph_tot/10.), ' yr')
```

```python
print('GBM + SFR')
print('\tWITH SagdSph')
print('\t\tN surveys with det sGRBs, SFR cond: ',n_coinc_det_sfr_tot)
print('\t\tNumber of years until prob>5%:',5*14/(n_coinc_det_sfr_tot/10.), ' yr')
print()
print('\tWITHOUT SagdSph')
print('\t\tN surveys with det sGRBs, SFR cond without SagdSPh: ',n_coinc_det_sfr_woSagdSph_tot)
print('\t\tNumber of years until prob>5%:',5*14/(n_coinc_det_sfr_woSagdSph_tot/10.), ' yr')
```

## Eiso
Consider only analyzable GRBs that could be fit with threeML. Use output of above fitting output

```python
Eiso_lv = data['Eiso_lv'].to_numpy()
Eiso_z = data['Eiso_z'].to_numpy()
```

### Numbers

```python
print('GBM + ANALYZABLE + Eiso selection')
print('\tWITH SagdSph')
univ = np.array(data['seed'][Eiso_lv<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
#print(len(univ))
#print(len(unique_univ))

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph']
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][Eiso_lv[mask]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
#include SFR selection
print('GBM + ANALYZABLE + Eiso + SFR selection')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][Eiso_lv[mask]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][Eiso_lv[mask]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
#combine with duration selection,neglect SagdSph
print('GBM + ANALYZABLE + Eiso + SFR + T90selection')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2)
univ = np.array(data['seed'][mask_comb][Eiso_lv[mask_comb]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

### Eiso histogram

```python
fig, ax = plt.subplots(figsize=(7,4)) 
ax.grid(lw=0.5)
ax.set_axisbelow(True)
#intervals=logbins_norm_histogram(Eiso_lv.to_numpy(),ax=ax,n_bins=100)
ax.hist(np.log10(Eiso_z),bins=30,histtype='step',lw=1.5,label='Host: at latent redshift $z$',color='C01')
n,bins,_=ax.hist(np.log10(Eiso_lv),histtype='step',lw=1.5,bins=40,label='Host: LV galaxy',edgecolor='C00')
ax.hist(np.log10(Eiso_lv[Eiso_lv<5.3e46]),bins=bins,label='Selected',color='C00')
ax.axvline(np.log10(5.3e46),color='C03',ls='--',label='Limit observed MGFs')
ax.set_xlabel('log$_{10}(E_{\mathrm{iso}})$ [erg]')
#ax.set_yscale('log')
ax.set_ylabel('Number')
#ax.set_ylim(1,50)
ax.legend(loc='upper center')
plt.savefig('/data/eschoe/grb_shader/figs/selection_eiso_wSagdSph.png',dpi=300)
```

### E_p-E_iso scatter plot with joint GBM and Konus data

```python
gbm = get_path_of_data_file('grb_short_gbm.csv')
konus = get_path_of_data_file('grb_short_konus.csv')
```

```python
data_gbm = pd.read_csv(gbm,sep='\s+',skiprows=3,names=['name','Eiso','Ep','z'])
data_konus = pd.read_csv(konus,sep=',',skiprows=3,names=['Eiso','Ep'])
```

```python
data_gbm
```

```python
fig, ax = plt.subplots(figsize=(7,4))
plt.rcParams.update({'font.size': 14})
ax.scatter(data_gbm['Eiso'],data_gbm['Ep'],s=8,marker='x',c='black',label='GBM + Konus data',alpha=0.8,zorder=3)
ax.scatter(data_konus['Eiso'],data_konus['Ep'],s=8,marker='x',c='black',label='',alpha=0.8,zorder=3)
ax.scatter(data['Eiso_lv'],data['Ep_obs_fit'],s=8,label='Simulated, host LV galaxy')
ax.scatter(data['Eiso_z'],data['Ep_obs_fit']*(1+data['z']),s=8,label='Simulated, host at latent $z$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.axvline(5.3e46,color='C03',ls='--',label='Limit observed MGFs')
ax.set_xlabel('$E_{iso}$ [erg]')
ax.set_ylabel('$E_{\mathrm{p}}$ [keV]')
ax.legend(fontsize=10,loc='upper center')
plt.savefig('/data/eschoe/grb_shader/figs/selection_ep_eiso_scatter.pdf')#,dpi=300)
```

### Exclude galaxies

```python
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
```

```python
fig, ax = plt.subplots(figsize=(7,4)) 
ax.grid(lw=0.5)
ax.set_axisbelow(True)

ax.hist(np.log10(Eiso_z[mask]),bins=24,histtype='step',lw=1.5,label='Host: at latent redshift $z$',color='C01')
n,bins,_=ax.hist(np.log10(Eiso_lv[mask]),histtype='step',lw=1.5,bins=43,label='Host: LV galaxy',edgecolor='C00')
ax.hist(np.log10(Eiso_lv[mask][Eiso_lv[mask]<5.3e46]),bins=bins,label='Selected',color='C00')

ax.axvline(np.log10(5.3e46),color='C03',ls='--',label='Limit observed MGFs')
ax.set_xlabel('log$_{10}(E_{\mathrm{iso}})$ [erg]')
#ax.set_yscale('log')
ax.set_ylabel('Number')
ax.set_ylim(0,12)
ax.legend(loc='upper center')
plt.savefig('/data/eschoe/grb_shader/figs/selection_eiso_woSagdSph.png',dpi=300)
```

```python
len(Eiso_z.to_numpy()[mask])
```

## Duration selection


### Numbers

```python
dur_bb = data['duration_bb'].to_numpy()
```

```python
print('GBM + ANALYZABLE + T90 selection')
print('\tWITH SagdSph')
univ = np.array(data['seed'][0.9*dur_bb<2.])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
#print(len(univ))
#print(len(unique_univ))

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph']
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][0.9*dur_bb[mask]<2.])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
#include SFR selection
print('GBM + ANALYZABLE + T90 + SFR selection')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][0.9*dur_bb[mask]<2.])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][0.9*dur_bb[mask]<2.])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

### Histogram with all galaxies

```python
fig, ax = plt.subplots(figsize=(7,4))
ax.grid(lw=0.5)
ax.set_axisbelow(True)
ax.hist(np.log10(0.9*dur_bb[0.9*dur_bb>2]),bins=12, histtype='step',edgecolor='C00',label='All',lw=1.5)
ax.hist(np.log10(0.9*dur_bb[0.9*dur_bb<2]),bins=20,color='C00',label='Selected')
ax.axvline(np.log10(2),color='C03',lw=2,ls='--',label='$T_{90}=2$ s')
ax.set_xlabel('log$_{10}(T_{90})$')
ax.set_ylabel('Number')


ax.legend()
plt.savefig('/data/eschoe/grb_shader/figs/selection_t90.png',dpi=300)
```

### Histogram excluding galaxies

```python
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]

fig, ax = plt.subplots(figsize=(7,4))
ax.grid(lw=0.5)
ax.set_axisbelow(True)
ax.hist(np.log10(0.9*dur_bb[mask][0.9*dur_bb[mask]>2]),bins=12, histtype='step',edgecolor='C00',label='All',lw=1.5)
ax.hist(np.log10(0.9*dur_bb[mask][0.9*dur_bb[mask]<2]),bins=20,color='C00',label='Selected')
ax.axvline(np.log10(2),color='C03',lw=2,ls='--',label='$T_{90}=2$ s')
ax.set_xlabel('log$_{10}(T_{90})$')
ax.set_ylabel('Number')

ax.legend()
plt.savefig('/data/eschoe/grb_shader/figs/selection_t90_woSagdSph')
```

## Fluence selection

```python
lv = LocalVolume.from_lv_catalog()
Eiso_lv = data['Eiso_lv'].to_numpy()
gal_names = data['gal'].to_numpy()
fluence = np.zeros_like(Eiso_lv)
for i in range(len(gal_names)):
    fluence[i]= Eiso_lv[i]/(4*np.pi* (lv.galaxies[gal_names[i]].distance*3.086e+24)**2)
```

### Histogram

```python
from popsynth.utils.cosmology import Cosmology
c = Cosmology

for i in range(len(gal_names)):
    fluence_z= Eiso_lv/(4*np.pi* (lv.galaxies[gal_names[i]].distance*3.086e+24)**2)
```

```python
fig, ax = plt.subplots(figsize=(7,4)) 
ax.grid(lw=0.5)
ax.set_axisbelow(True)
#intervals=logbins_norm_histogram(Eiso_lv.to_numpy(),ax=ax,n_bins=100)
#ax.hist(np.log10(fluence),bins=30,histtype='step',lw=1.5,label='Host: at latent redshift $z$',color='C01')
n,bins,_=ax.hist(np.log10(fluence),histtype='step',lw=1.5,bins=41,label='All',edgecolor='C00')
ax.hist(np.log10(fluence[fluence>1e-6]),bins=bins,label='Selected',color='C00')
ax.axvline(np.log10(1e-6),lw=2,color='C03',ls='--',label='Limit IPN')
ax.set_xlabel(r'log$_{10}$(Fluence) [erg cm$^{-2}$]')
#ax.set_yscale('log')
ax.set_ylabel('Number')
#ax.set_ylim(1,50)
ax.legend(loc='upper right')
plt.savefig('/data/eschoe/grb_shader/figs/selection_fluence.png',dpi=300)
```

### Numbers

```python
print('GBM + ANALYZABLE + Fluence selection')
print('\tWITH SagdSph')
univ = np.array(data['seed'][fluence>1e-6])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
#print(len(univ))
#print(len(unique_univ))

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph']
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
univ = np.array(data['seed'][mask][fluence[mask]>1e-6])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
#ALL Selections
print('ALL: GBM + ANALYZABLE + Fluence + Eiso + SFR + T90 selection')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb][Eiso_lv[mask_comb]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb][Eiso_lv[mask_comb]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + T90 selection')
print('\tWITH SagdSph')
_exclude = []#hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask,0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] #+ hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask,0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + Eiso selection')
print('\tWITH SagdSph')
_exclude = []#hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask,Eiso_lv<5.3e46),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] #+ hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask,Eiso_lv<5.3e46),fluence>1e-6)
univ = np.array(data['seed'][mask_comb][Eiso_lv[mask_comb]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + SFR')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(mask,fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(mask,fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + Eiso + T90 selection')
print('\tWITH SagdSph')
_exclude = []#hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb][Eiso_lv[mask_comb]<5.3e46])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] #+ hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + SFR + Eiso')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, Eiso_lv<5.3e46),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

```python
print('GBM + ANALYZABLE + Fluence + SFR + Eiso')
print('\tWITH SagdSph')
_exclude = hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, 0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')

print()
print('\tWITHOUT SagdSph')
_exclude = ['SagdSph'] + hit_galaxies_sfr_excl
mask = [(data['gal'].to_numpy()[i] not in _exclude) for i in range(len(data['gal'].to_numpy()))]
mask_comb = np.logical_and(np.logical_and(mask, 0.9*dur_bb<2),fluence>1e-6)
univ = np.array(data['seed'][mask_comb])
unique_univ = np.unique(univ)
print('\t\tN surveys:',len(unique_univ))
print('\t\tNumber of years until prob>5%:',5*14/(len(unique_univ)/10.), ' yr')
```

# Histogram all hit galaxies

```python
#plt.clf()
fig, ax = plt.subplots(1,2)
#restored_sim.hist_n_sim_grbs(ax=ax,alpha=0.5,label='Total')
restored_sim.hist_galaxies(n=30,ax=ax[0],alpha=0.5,label='Hit',width=0.7)
restored_sim.hist_galaxies_detected(n=30,ax=ax[1],alpha=0.5,label='Hit',width=0.7)
#restored_sim.hist_n_det_grbs(ax=ax,alpha=0.5,label='Hit+detected',width=0.7)
ax[0].set_xlabel('# Spatial Coincidences')
ax[1].set_xlabel('# Spatial Coincidences')

print(len(restored_sim._galaxynames_hit))
print(len(restored_sim._galaxynames_hit_and_detected))
plt.show()
#plt.savefig('/data/eschoe/grb_shader/figs/coincidences_gals.pdf')
```

# Histogram number detections

```python
#plt.clf()
fig, ax = plt.subplots(figsize=(7,4))
ax.grid()
ax.set_axisbelow(True)
#restored_sim.hist_n_sim_grbs(ax=ax,alpha=0.5,label='Total')
restored_sim.hist_n_hit_grbs(ax=ax,alpha=0.8,label='Hit',width=0.7)
restored_sim.hist_n_det_grbs(ax=ax,alpha=0.6,label='Hit+detected',width=0.7)
labels, counts = np.unique(n_coinc_woSagdSph_arr, return_counts=True)
ax.bar(labels,counts,label='Hit, without SagdSph',alpha=0.5)
ax.set_xlabel('# Spatial Coincidences')
ax.set_ylabel('# Surveys')

ax.legend()
plt.show()
plt.savefig('/data/eschoe/grb_shader/figs/coincidences_woSagdSph.png',dpi=300)
```

## WITHOUT SagdSph

```python
fig, ax = plt.subplots()
ax.set_xlabel('# Spatial Coincidences')
ax.set_ylabel('# Surveys')
labels, counts = np.unique(n_coinc_woSagdSph_arr, return_counts=True)
ax.bar(labels,counts)
#ax.set_xlim(0.5,None)
#ax.set_ylim(0,50)
```

```python
n_coinc_det_woSagdSph
```

```python
np.sum(restored_sim.n_detected_grbs)/np.sum(restored_sim.n_hit_grbs)
```

```python
1.-len(restored_sim.n_hit_grbs[restored_sim.n_hit_grbs==0])/np.sum(restored_sim.n_hit_grbs)
```

# N(distance,area)

```python
fig, (ax,ax2) = plt.subplots(1,2,figsize=(10,5))
restored_sim.plot_det_distance_area_n(ax=ax2)
restored_sim.plot_distance_area_n(ax=ax)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Distance [Mpc]')
ax.set_ylabel(r'Angular Area [rad$^2$]')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Distance [Mpc]')
ax2.set_ylabel(r'Angular Area [rad$^2$]')
plt.savefig('/data/eschoe/grb_shader/figs/distance_area_n')
```

```python

```
