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

```python
from cosmogrb.universe.survey import Survey
from popsynth import silence_warnings
silence_warnings()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
%matplotlib widget
from matplotlib.ticker import MaxNLocator
from grb_shader import RestoredMultiverse
from tqdm.auto import tqdm
```

```python
sim_path = '/data/eschoe/grb_shader/sims/220722/ghirl_c_t90fit/catalogselec/'

restored_sim = RestoredMultiverse(sim_path)
```

```python
fig, ax = plt.subplots(1,2)

restored_sim.hist_galaxies_detected(ax=ax[1],alpha=0.5,n=15)
restored_sim.hist_galaxies(ax=ax[0],alpha=0.5,n=15)
ax[0].set_title('Hit')
ax[1].set_title('Hit+detected')
plt.show()
```

```python
fig, ax = plt.subplots()

#restored_sim.hist_n_sim_grbs(ax=ax,alpha=0.5,label='Total')
restored_sim.hist_n_hit_grbs(ax=ax,alpha=0.5,label='Hit',width=0.7)
restored_sim.hist_n_det_grbs(ax=ax,alpha=0.5,label='Hit+detected',width=0.7)
ax.legend()
plt.show()
```
