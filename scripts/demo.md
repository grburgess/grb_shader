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
from grb_shader import LocalVolume, play_god, GRBPop, RestoredSimulation,get_path_of_data_file

import popsynth
popsynth.update_logging_level("INFO")
popsynth.silence_progress_bars()
popsynth.silence_warnings()

%matplotlib notebook
```

```python
import gbmgeometry
```

```python
#generate dictionary with names of all local volume galaxies and the object Galaxy with corresponding properties
lv = LocalVolume.from_lv_catalog()
import numpy as np
#lv.display()
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
sim_path_c = "/data/eschoe/grb_shader/sims/ghirlanda_c_triangle/pop"
r_c = RestoredSimulation(sim_path_c)
```

```python
#histogram which galaxy was hit by a SGRB how many times
exclude_list = []
#exclude=["SagdSph", "MESSIER031"]
r_c.hist_galaxies(20, exclude=[])
```

```python
lv.read_population(r_c.populations[0])
lv.show_selected_galaxies()
```

```python

```

```python

```
