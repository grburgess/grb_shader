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
from grb_shader import LocalVolume, GRBPop,get_path_of_data_file,RestoredMultiverse
from grb_shader.restored_simulation import RestoredUniverse
from pathlib import Path
import popsynth as ps
import matplotlib.pyplot as plt
import numpy as np
from grb_shader.samplers import DurationSampler, TDecaySampler, EisoSampler, TriangleT90Sampler_Cor
%matplotlib widget
import scipy.stats as stats
from cosmogrb.universe.survey import Survey
import networkx as nx
from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.distributions.bpl_distribution import BPLDistribution,bpl
from astromodels import *
```

```python
from popsynth.distributions.cosmological_distribution import SFRDistribution
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

r0=[11.2/14,57/14]
a=1
rise=2.0
decay=2.0
peak=2.8
fig, ax=plt.subplots(figsize=(5,4))
for i in range(len(r0)):
    r0_i = r0[i]
    sfr = SFRDistribution()
    sfr.r0=r0_i
    sfr.a=a
    sfr.rise=rise
    sfr.decay=decay
    sfr.peak=peak
    x = np.linspace(0,6,1000)
    f =sfr.dNdV(x) #* sfr.differential_volume(x) / sfr.time_adjustment(x)
    ax.plot(x,f)

ax.set_yscale('log')
ax.set_xlabel('z')
ax.set_ylabel(r'$\dot{\rho}^\prime$ [Gpc$^{-3}$ yr$^{-1}$]')
#plt.savefig('/data/eschoe/grb_shader/figs/redshift_distribution2.png',dpi=300)
```

# with class


## Case c - constant

```python
dir_sim_c = '/data/eschoe/grb_shader/sims/220921/noselec_const_no_trunc_occ/'
survey_file = dir_sim_c #+ 'survey_1234.h5'
```

```python
c1 = RestoredUniverse(survey_file)
```

```python
print(c1.n_sim_grbs)
print(c1.survey.n_detected)
```

```python
c1.hist_parameters(normalized_hist=True)
plt.savefig('/data/eschoe/grb_shader/figs/distributions_noselec_const_no_trunc_occ.png',dpi=300)
```

## Case a - constant- only pop

```python
dir_sim_c = '/data/eschoe/grb_shader/sims/221010/a_noselec_const_no_trunc_noocc/'
#survey_file = dir_sim_c + 'survey_1234.h5'
```

```python
c2 = RestoredUniverse(dir_sim_c)
c2.pop.fluxes[6]
```

```python
c2.hist_parameters(
    normalized_hist=True,
    sfr_r0=1.68,
    sfr_a=1,
    sfr_rise=1.8,
    sfr_decay=1.7,
    sfr_peak=2.7,
    ep_bpl_min=0.1,
    ep_bpl_alpha= -0.8,
    ep_bpl_break= 1400,
    ep_bpl_beta=-2.6,
    ep_bpl_max=1.e+5,
    lum_min=1.e+47,
    lum_alpha=-0.88,
    lum_break= 2.1e+52,
    lum_beta=-2.2,
    lum_max=1.e+55,
    alpha_mu=-0.6,
    alpha_sigma=0.2,
    alpha_lower_bound=-1.5,
    alpha_upper_bound=0,
    t90_mu = -0.196573,
    t90_sigma=0.541693
                  )
```

## Case c - pulse

```python
dir_sim_c_pulse = '/data/eschoe/grb_shader/sims/221016/c_noselec_pulse_noocc_inc/'
```

```python
u3 = RestoredUniverse(dir_sim_c_pulse)
```

```python
u3.survey.info()
```

```python
u3.hist_parameters(normalized_hist=True)
```

```python

```
