---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from grb_shader import LocalVolume, play_god, GRBPop, RestoredSimulation, get_ghirlanda_model

import popsynth
#popsynth.update_logging_level("INFO")
popsynth.silence_progress()
popsynth.silence_warnings()


%matplotlib notebook
```

```python
lv = LocalVolume.from_lv_catalog()
#lv.display()
```

```python
sim_path = "sims/pop_"

play_god(param_file=get_ghirlanda_model(),
         n_sims=500,
         n_cpus=8, 
         base_file_name=sim_path)
```

```python
r = RestoredSimulation(sim_path)
```

```python
r.hist_galaxies(17, exclude=["SagdSph", "MESSIER031"])
```

```python
lv.read_population(r.populations[1])
lv.show_selected_galaxies()

```

```python
lv.display()
```
