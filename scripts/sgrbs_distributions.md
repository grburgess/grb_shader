---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Distributions used to generate population of short GRBs

```python
from grb_shader import LocalVolume, play_god, GRBPop, RestoredSimulation,get_path_of_data_file
from pathlib import Path
import popsynth as ps
import matplotlib.pyplot as plt
import numpy as np
from grb_shader.samplers import CatalogSelector, DurationSampler, TDecaySampler, EisoSampler, TriangleT90Sampler_Cor
%matplotlib widget
```

In the following the chosen distributions for the short GRBs are plotted for all 3 cases (a-c) of Ghirlanda (2016).

A hard flux selector was chosen with the limit of $10^{-7}$ erg s$^{-1}$ cm$^{-2}$ to show results.


## Case a)

```python
#Hard flux selection of 1e-7 erg cm^-2 s^-1
#without spatial selection
sim_path_a = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/a/pop"
param_file_a = get_path_of_data_file("ghirlanda2016_a_triangle.yml")
```

```python
pop_a = ps.Population.from_file(f"{sim_path_a}_1874.h5")
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
plt.hist(pop_a.distances[pop_a.selection],bins=100,alpha=0.7,label='observed')
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
plt.hist(np.log10(pop_a.ep[pop_a.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{peak}$ [keV])')
```

### Duration - from T = 2*E_iso/L and E_iso from E_iso-E_p correation

```python
plt.subplots()
#print(pop.luminosities_latent)
plt.hist(np.log10(0.9*pop_a.duration*(1+pop_a.distances)),bins=50,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_a.duration[pop_a.selection]*(1+pop_a.distances[pop_a.selection])),bins=50,alpha=0.7,label='observed')
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
sim_path_b = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/b/pop"
sim_path_b2 = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/b2/pop"
param_file_b = get_path_of_data_file("ghirlanda2016_b_triangle.yml")
```

```python
pop_b = ps.Population.from_file(f"{sim_path_b}_1714.h5")
pop_b2 = ps.Population.from_file(f"{sim_path_b2}_1714.h5")
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
plt.hist(np.log10(pop_b2.Eiso[pop_b2.selection]),bins=100,alpha=0.7,label='all2')
plt.hist(np.log10(pop_b2.Eiso),bins=100,alpha=0.7,label='all2')
plt.legend()
plt.xlabel(r'log$_{10}$(E$_\mathrm{iso}$ [erg])')
```

### Duration - from T = 2*E_iso/L and E_iso from E_iso-E_p correation

```python
plt.subplots()
#print(pop.luminosities_latent)
plt.hist(np.log10(0.9*pop_b.duration*(1+pop_b.distances)),bins=100,color='C00',alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_b.duration[pop_b.selection]*(1+pop_b.distances[pop_b.selection])),bins=100,color='C01',alpha=0.7,label='observed')
plt.hist(np.log10(0.9*pop_b2.duration*(1+pop_b2.distances)),bins=100,color='C02',alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_b2.duration[pop_b2.selection]*(1+pop_b2.distances[pop_b2.selection])),bins=100,color='C03',alpha=0.7,label='observed')
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
sim_path_c = "/data/eschoe/grb_shader/sims/ghirlanda_triangle_hardfluxselec_wospatialselec/c2/pop"
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
plt.hist(np.log10(pop_c.luminosities[pop_c.selection]),bins=100,alpha=0.7,label='all')
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
plt.hist(np.log10(0.9*pop_c.duration*(1+pop_c.distances)),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c.duration[pop_c.selection]*(1+pop_c.distances[pop_c.selection])),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

### Flux-redshift diagram

```python
fig = pop_c.display_fluxes(true_color="C00", obs_color="C01", with_arrows=False, s=8)
ax = fig.get_axes()[0]
ax.set_xlabel('z')
```

Offene Fragen:
- Warum sieht duration distribution nicht so aus wie im Paper?
    - im Paper: Peak um etwa $T_{\mathrm{obs}}=10^{-0.5}$  s, hier aber bei: $T_{\mathrm{obs}}=10^{0.5}$ s
- Warum unterscheidet sich die Zahl der GRBs, die simuliert wird so stark (Integral über Redshift distribution)
    - Normierungsfaktor r_0c ist 4x größer als in Fall a (Warum aber?)


## Case c) with T90 fit distribution

```python
sim_path_c2 = "/data/eschoe/grb_shader/sims/ghirlanda_t90fit_hardfluxselec_wospatialselec/pop"
param_file_c2 = get_path_of_data_file("ghirlanda2016_c_t90fit.yml")
```

```python
pop_c2 = ps.Population.from_file(f"{sim_path_c2}_1234.h5")

```

### Duration - fit to GBM data

```python
plt.subplots()
#print(pop.luminosities_latent)

# from rest frame to observer frame 
plt.hist(np.log10(0.9*pop_c2.duration),bins=100,alpha=0.7,label='all')
plt.hist(np.log10(0.9*pop_c2.duration[pop_c2.selection]),bins=100,alpha=0.7,label='observed')
plt.legend()
plt.xlabel(r'log$_{10}$($t_{90}$) [s]')
```

```python
from popsynth.aux_samplers import BrokenPowerLawAuxSampler
```

```python
s =BrokenPowerLawAuxSampler('test')
s.xmin=0.1
s.xmax=1.0e+5
s.alpha=0.55
s.beta= -2.5
s.xbreak=2100
```

```python
s.true_sampler(size=20000)
```

```python
Ep = s.true_values

m_y = 0.69
q_y =0.068
```

```python
L = np.power(10,(1./m_y * ( np.log10(Ep/670.) - q_y))) * 1e52 #erg/s
```

```python
plt.subplots()
plt.hist(np.log10(L),bins=50)
```

```python

```
