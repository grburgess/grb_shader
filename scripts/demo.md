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
from grb_shader import LocalVolume

import popsynth as ps
ps.update_logging_level("INFO")
ps.silence_progress_bars()
ps.silence_warnings()

import matplotlib.pyplot as plt
import numpy as np

%matplotlib widget
```

```python
#generate dictionary with names of all local volume galaxies and the object Galaxy with corresponding properties
lv = LocalVolume.from_lv_catalog()
dir_sim = "/data/eschoe/grb_shader/sims/220902/noselec_const_no_trunc/"
pop = ps.Population.from_file(dir_sim + 'pop_1234.h5')
#lv.display()
```

```python
lv.read_population(pop)
lv.selected_galaxies
#lv.show_selected_galaxies()
```

```python
lv.show_selected_galaxies()
```

# Case a

```python
sim_path_a = "/data/eschoe/grb_shader/sims/ghirlanda_a_triangle/pop"
r_a = RestoredSimulation(sim_path_a)
```

```python
#histogram which galaxy was hit by a SGRB how many times
exclude_list = []
#exclude=["SagdSph", "MESSIER031"]
r_a.hist_galaxies(1000, exclude=[])
```

```python
lv.read_population(r_a.populations[8])
lv.show_selected_galaxies()
```

```python
#lv.display()
```

# Case b

```python
sim_path_b = "/data/eschoe/grb_shader/sims/ghirlanda_b_triangle/pop"
r_b = RestoredSimulation(sim_path_b)
```

```python
#histogram which galaxy was hit by a SGRB how many times
exclude_list = []
#exclude=["SagdSph", "MESSIER031"]
r_b.hist_galaxies(1000, exclude=[])
```

```python
lv.read_population(r_b.populations[46])
lv.show_selected_galaxies()
```

# Case c

```python
sim_path_c = "/data/eschoe/grb_shader/sims/ghirlanda_t90fit_hardfluxselec_wospatialselec/3/pop"
r_c = RestoredSimulation(sim_path_c)
```

```python
#histogram which galaxy was hit by a SGRB how many times
exclude_list = []
#exclude=["SagdSph", "MESSIER031"]
r_c.hist_galaxies(20, exclude=[]);
```

```python
lv.read_population(r_c.populations[0])
lv.show_selected_galaxies()
```

```python

```

```python

```
