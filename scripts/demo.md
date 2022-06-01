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
from grb_shader import LocalVolume, play_god, GRBPop, RestoredSimulation,get_ghirlanda_model

import popsynth
popsynth.update_logging_level("INFO")
popsynth.silence_progress_bars()
popsynth.silence_warnings()

%matplotlib notebook
```

```python
#generate dictionary with names of all local volume galaxies and the object Galaxy with corresponding properties
lv = LocalVolume.from_lv_catalog()
#lv.display()
```

```python
sim_path = "sims/pop_"

#generate population of SGRBs
play_god(param_file=get_ghirlanda_model(),
         n_sims=500,
         n_cpus=8, 
         base_file_name=sim_path)
```

```python
sim_path = "sims/pop_"
r = RestoredSimulation(sim_path)
```

```python
#histogram which galaxy was hit by a SGRB how many times
r.hist_galaxies(17, exclude=["SagdSph", "MESSIER031"])
```

```python
lv.read_population(r.populations[30])
lv.show_selected_galaxies()

```

```python
#lv.display()
```

```python

```
