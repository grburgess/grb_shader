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

# Distributions used to generate population of short GRBs

```python
from grb_shader import LocalVolume, GRBPop,get_path_of_data_file,RestoredMultiverse
from pathlib import Path
import popsynth as ps
import matplotlib.pyplot as plt
import numpy as np
from grb_shader.samplers import DurationSampler, TRiseSampler, EisoSampler, TriangleT90Sampler_Cor
%matplotlib widget
import scipy.stats as stats
from cosmogrb.universe.survey import Survey
import networkx as nx
#import seaborn as sns; #sns.set_theme()
#custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#sns.set_theme(style="ticks", rc=custom_params,font='serif')
```

In the following the chosen distributions for the short GRBs are plotted for all 3 cases (a-c) of Ghirlanda (2016).

A hard flux selector was chosen with the limit of $10^{-7}$ erg s$^{-1}$ cm$^{-2}$ to show results.


## Case a)

```python
#Hard flux selection of 1e-7 erg cm^-2 s^-1
#without spatial selection
#sim_path_a = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/a/pop"
param_file_a = get_path_of_data_file("ghirlanda2016_a_triangle.yml")

sim_path_a = "/data/eschoe/grb_shader/sims/221010/a_noselec_const_no_trunc_noocc/pop"
```

```python
pop_a = ps.Population.from_file(f"{sim_path_a}_1234.h5")
```

### Number of GRBs simulated (from integral over redshift distribution)

```python
print(f'Total number GRBs in Universe: \t{pop_a.n_objects}')
print(f'Number detected GRBs: \t\t{pop_a.n_detections}')
print(f'Number non-detected GRBs: \t{pop_a.n_non_detections}')
```

### Show graphical model

```python
import networkx as nx
import matplotlib.pyplot as plt

#duration.draw_sample(size=100)

plt.subplots()
G = pop_a.graph

seed = 10
pos = nx.spring_layout(G, seed=seed,k=1)

nodes = nx.draw_networkx_nodes(G, pos, node_color="indigo",label=True,node_size=1000, alpha=0.7)
edges = nx.draw_networkx_edges(
    G,
    pos, label=True,
    arrowstyle="->",
    arrowsize=10,
    width=1,
)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='white',font_size='8')
```

### Redshift distribution: Cole et al. (2001)

```python
plt.subplots()
plt.hist(pop_a.distances,bins=100,alpha=0.7,label='all')
#plt.hist(pop_a.distances[pop_a.selection],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'z')
```

### Luminosity Distribution: broken power law (from Ep-L correlation)

```python
pop_a.luminosity_parameters
```

```python
plt.subplots()
plt.hist(np.log10(pop_a.luminosities),alpha=0.7,bins=100,label='all')
plt.hist(np.log10(pop_a.luminosities[pop_a.selection]),alpha=0.7,bins=100,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(L [erg/s])')
```

### E_peak - broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_a.ep),bins=100,alpha=0.7,label='all')
#plt.hist(np.log10(pop_a.ep[pop_a.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

### Duration - from T = 2*E_iso/L and E_iso from E_iso-E_p correation

```python
plt.subplots()
#print(pop.luminosities_latent)
plt.hist(np.log(0.9*pop_a.duration*(1+pop_a.distances)),bins=50,alpha=0.7,label='all')
plt.hist(np.log(0.9*pop_a.duration[pop_a.selection]*(1+pop_a.distances[pop_a.selection])),bins=50,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

### Flux-redshift diagram

```python
fig = pop_a.display_fluxes(true_color="C00", obs_color="C01", with_arrows=False, s=8)
ax = fig.get_axes()[0]
ax.set_xlabel('z')
```

## Case b)

```python
sim_path_b = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/b2/pop"
param_file_b = get_path_of_data_file("ghirlanda2016_b_triangle.yml")
```

```python
pop_b = ps.Population.from_file(f"{sim_path_b}_1714.h5")
```

### Number of GRBs simulated

```python
print(f'Total number GRBs in Universe: \t{pop_b.n_objects}')
print(f'Number detected GRBs: \t\t{pop_b.n_detections}')
print(f'Number non-detected GRBs: \t{pop_b.n_non_detections}')
```

### Show graphical model

```python
import networkx as nx
import matplotlib.pyplot as plt

#duration.draw_sample(size=100)

plt.subplots()
G = pop_b.graph

seed = 10
pos = nx.spring_layout(G, seed=seed,k=1)

nodes = nx.draw_networkx_nodes(G, pos, node_color="indigo",label=True,node_size=1000, alpha=0.7)
edges = nx.draw_networkx_edges(
    G,
    pos, label=True,
    arrowstyle="->",
    arrowsize=10,
    width=1,
)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='white',font_size='8')
```

### Redshift distribution: Cole et al. (2001)

```python
plt.subplots()
plt.hist(pop_b.distances,bins=100,alpha=0.7,label='all')
plt.hist(pop_b.distances[pop_b.selection],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'z')
```

### Luminosity Distribution: broken power law (from Ep-Liso correlation)

```python
pop_b.luminosity_parameters
```

```python
plt.subplots()
plt.hist(np.log10(pop_b.luminosities),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_b.luminosities[pop_b.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(L [erg/s])')
```

### E_peak - broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_b.ep),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_b.ep[pop_b.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

```python
plt.subplots()
plt.hist(np.log10(pop_b.Eiso),bins=100,alpha=0.7,label='all1')
plt.hist(np.log10(pop_b.Eiso[pop_b.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{iso}$ [erg])')
```

### Duration - from T = 2*E_iso/L and E_iso from E_iso-E_p correation

```python
plt.subplots()
#print(pop.luminosities_latent)
plt.hist(np.log10(0.9*pop_b.duration*(1+pop_b.distances)),bins=100,color='C00',alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_b.duration[pop_b.selection]*(1+pop_b.distances[pop_b.selection])),bins=100,color='C01',alpha=0.7,label='observed')
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
plt.legend()
```

### Flux-redshift diagram

```python
fig = pop_b.display_fluxes(true_color="C00", obs_color="C01", with_arrows=False, s=8)
ax = fig.get_axes()[0]
ax.set_xlabel('z')
```

## Case c)

```python
sim_path_c = "/data/eschoe/grb_shader/sims/ghirlanda_t90fit_hardfluxselec_wospatialselec/3/pop"
param_file_c = get_path_of_data_file("ghirlanda2016_c_constant.yml")
```

```python
pop_c = ps.Population.from_file(f"{sim_path_c}_1234.h5")
```

### Number of GRBs simulated

```python
print(f'Total number GRBs in Universe: \t{pop_c.n_objects}')
print(f'Number detected GRBs: \t\t{pop_c.n_detections}')
print(f'Number non-detected GRBs: \t{pop_c.n_non_detections}')
```

### Show graphical model

```python
import networkx as nx
import matplotlib.pyplot as plt

#duration.draw_sample(size=100)

plt.subplots()
G = pop_c.graph

seed = 10
pos = nx.spring_layout(G, seed=seed,k=1)

nodes = nx.draw_networkx_nodes(G, pos, node_color="indigo",label=True,node_size=1000, alpha=0.7)
edges = nx.draw_networkx_edges(
    G,
    pos, label=True,
    arrowstyle="->",
    arrowsize=10,
    width=1,
)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='white',font_size='8')
```

### Redshift distribution: Cole et al. (2001)

```python
plt.subplots()
plt.hist(pop_c.distances,bins=100,alpha=0.7,label='all')
plt.hist(pop_c.distances[pop_c.selection],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'z')
```

### Luminosity Distribution: broken power law

```python
pop_c.luminosity_parameters
```

```python
plt.subplots()
plt.hist(np.log10(pop_c.luminosities),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c.luminosities[pop_c.selection]),bins=100,alpha=0.7,label='detected')
plt.legend()
plt.xlabel(r'log$_{10}$(L [erg/s])')
```

### E_peak - broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_c.ep),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c.ep[pop_c.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

### Duration - from fitted t90 lognormal distribution in Ghirlanda case c

```python
plt.subplots()
#print(pop.luminosities_latent)

# from rest frame to observer frame 
plt.hist(np.log10(0.9*pop_c.duration),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c.duration[pop_c.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

### Flux-redshift diagram

```python
fig = pop_c.display_fluxes(true_color="C00", obs_color="C01", with_arrows=False, s=8)
ax = fig.get_axes()[0]
ax.set_xlabel('z')
ax.set_ylabel('Flux []')
```

Offene Fragen:
- Warum sieht duration distribution für a) und b) nicht so aus wie im Paper?
    - im Paper: Peak um etwa $T_{\mathrm{obs, p}}=10^{-0.5}$  s, hier aber bei: $T_{\mathrm{obs, p}}=10^{0.5}$ s
- Warum unterscheidet sich die Zahl der GRBs, die simuliert wird so stark (Integral über Redshift distribution)
    - Normierungsfaktor r_0c ist 4x größer als in Fall a


## Case c) with T90 fit distribution - Hard flux selection

```python
sim_path_c2 = "/data/eschoe/grb_shader/sims/ghirlanda_t90fit_hardfluxselec_wospatialselec/3/"
param_file_c2 = get_path_of_data_file("ghirlanda2016_c_t90fit.yml")
```

```python
pop_c2 = ps.Population.from_file(f"{sim_path_c2}pop_1254.h5")
```

### Number simulated GRBs with hard flux selection

```python
print(f'Total number GRBs in Universe: \t{pop_c2.n_objects}')
#print(f'Integral{pop_2.}')
print(f'Number detected GRBs: \t\t{pop_c2.n_detections}')
print(f'Number non-detected GRBs: \t{pop_c2.n_non_detections}')
```

```python
r_c = RestoredMultiverse(sim_path_c2)
```

```python
#With hard flux selection (1e-7 erg cm^-2 s^-1)! - repeat after using cosmogrb
fig, ax = plt.subplots()
r_c.hist_n_sim_grbs(ax=ax,label='All simulated',lw=0,width=1)
r_c.hist_n_det_grbs(ax=ax,label='Detected',lw=0,width=1)
ax.axvline(718,ls='--',color='black',label='GBM SGRBs') #718 when using best fit for t90 distribution
ax.legend()
```

### Duration - fit to GBM data

```python
plt.subplots()
#print(pop.luminosities_latent)

# from rest frame to observer frame 
plt.hist(np.log10(0.9*pop_c2.duration),bins=100,alpha=0.4,label='all',density=True)
plt.hist(np.log10(0.9*pop_c2.duration[pop_c2.selection]),bins=100,alpha=0.4,label='observed',density=True)
t_arr = np.linspace(-2,3,1000)
plt.plot(t_arr, stats.norm.pdf(t_arr,-0.196573, 0.541693),label='SGRBs Median Best Fit',ls='--',color='black')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

## Case c) T90 fit, constant, no t90 truncation - GBM trigger

```python
dir_sim_c_const_notrunc = "/data/eschoe/grb_shader/sims/220902/noselec_const_no_trunc/"
pop_c_const_notrunc = ps.Population.from_file(dir_sim_c_const_notrunc+f"pop_1234.h5")
```

### Number simulated and detected GRBs (GBM trigger)

```python
survey_file_no_selec_const_notrunc = dir_sim_c_const_notrunc + 'survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec_const_notrunc = Survey.from_file(survey_file_no_selec_const_notrunc)
survey_no_selec_const_notrunc.info()

mask_det_const_notrunc = survey_no_selec_const_notrunc.mask_detected_grbs
```

### Duration distribution - T90 fit

```python
plt.subplots()
plt.hist(np.log10(0.9*pop_c_const_notrunc.duration),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c_const_notrunc.duration[mask_det_const_notrunc]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

## Case c) with T90 fit and constant - GBM trigger - t90 truncation

```python
dir_sim_c_const = "/data/eschoe/grb_shader/sims/220902/noselec_const/"
pop_c_const = ps.Population.from_file(dir_sim_c_const+f"pop_1234.h5")
```

### Number of simulated and detected GRBs (GBM trigger)

```python
survey_file_no_selec_const = dir_sim_c_const + 'survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec_const = Survey.from_file(survey_file_no_selec_const)
survey_no_selec_const.info()

mask_det_const = survey_no_selec_const.mask_detected_grbs
```

### Show graphical model

```python
plt.subplots()
G = pop_c_const.graph

seed = 10
pos = nx.spring_layout(G, seed=seed,k=1)

nodes = nx.draw_networkx_nodes(G, pos, node_color="indigo",label=True,node_size=1000, alpha=0.7)
edges = nx.draw_networkx_edges(
    G,
    pos, label=True,
    arrowstyle="->",
    arrowsize=10,
    width=1,
)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='white',font_size='8')
```

### Redshift distribution: Cole et al. 2001

```python
plt.subplots()
plt.hist(pop_c_const.distances,bins=100,alpha=0.7,label='all')
plt.hist(pop_c_const.distances[mask_det_const],bins=100,alpha=0.7,label='detected')
plt.legend()
plt.xlabel(r'z')
```

### Luminosity Distribution: broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_c_const.luminosities),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c_const.luminosities[mask_det_const]),bins=100,alpha=0.7,label='detected')
plt.legend()
plt.xlabel(r'log$_{10}$(L [erg/s])')
```

### E_peak - broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_c_const.ep),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c_const.ep[mask_det_const]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

### Duration distribution - fitted T90

```python
plt.subplots()
plt.hist(np.log10(0.9*pop_c_const.duration),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c_const.duration[mask_det_const]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

## Case c) with T90 fit and constant- GBM trigger

```python
dir_sim_c_pulse = '/data/eschoe/grb_shader/sims/220921/noselec_const_no_trunc/'
sim_path_c_pulse = dir_sim_c_pulse + "pop"
pop_c_pulse = ps.Population.from_file(f"{sim_path_c_pulse}_1234.h5")
```

### Number of simulated and detected GRBs (GBM trigger)

```python
survey_file_no_selec_pulse = dir_sim_c_pulse + 'survey_1234.h5'
#selected a trigger threshold of 4.5 $\sigma$ to mimic the true GBM trigger

survey_no_selec_pulse = Survey.from_file(survey_file_no_selec_pulse)
survey_no_selec_pulse.info()

mask_det = survey_no_selec_pulse.mask_detected_grbs
```

### Show graphical model

```python
plt.subplots()
G = pop_c_pulse.graph

seed = 10
pos = nx.spring_layout(G, seed=seed,k=1)

nodes = nx.draw_networkx_nodes(G, pos, node_color="indigo",label=True,node_size=1000, alpha=0.7)
edges = nx.draw_networkx_edges(
    G,
    pos, label=True,
    arrowstyle="->",
    arrowsize=10,
    width=1,
)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='white',font_size='8')
```

### Redshift distribution: Cole et al. 2001

```python
plt.subplots()
plt.hist(pop_c_pulse.distances,bins=100,alpha=0.7,label='all')
plt.hist(pop_c_pulse.distances[mask_det],bins=100,alpha=0.7,label='detected')
plt.legend()
plt.xlabel(r'z')
```

### Luminosity distribution: broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_c_pulse.luminosities),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c_pulse.luminosities[mask_det]),bins=100,alpha=0.7,label='detected')
plt.legend()
plt.xlabel(r'log$_{10}$(L [erg/s])')
```

### E_peak - broken power law

```python
plt.subplots()
plt.hist(np.log10(pop_c_pulse.ep/(pop_c_pulse.distances)),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(pop_c_pulse.ep[mask_det]/(pop_c_pulse.distances[mask_det])),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

### Duration - from fitted T90 distribution

```python
plt.subplots()
plt.hist(np.log10(0.9*pop_c_pulse.duration),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c_pulse.duration[mask_det]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

### t_rise Distribution - N(1,1) [0.01,5.0]

```python
plt.subplots()
plt.hist(pop_c_pulse.trise,bins=100,alpha=0.7,label='all')
plt.hist(pop_c_pulse.trise[mask_det],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r't$_{\mathrm{rise}}$ [s]')
```

### t_decay distribution - from t_rise and t90

```python
plt.subplots()
plt.hist(pop_c_pulse.tdecay[pop_c_pulse.tdecay < 10],bins=100,alpha=0.7,label='all')
plt.hist(pop_c_pulse.tdecay[pop_c_pulse.tdecay < 10][mask_det[pop_c_pulse.tdecay < 10]],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r't$_{\mathrm{decay}}$ [s]')
plt.xlim(0,10)

print(np.max(pop_c_pulse.tdecay))
```

### tau distribution - N(2,0.5) [1.5,2.5]

```python
plt.subplots()
plt.hist(pop_c_pulse.tau,bins=100,alpha=0.7,label='all')
plt.hist(pop_c_pulse.tau[mask_det],bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'$\tau$ [s]')
```
