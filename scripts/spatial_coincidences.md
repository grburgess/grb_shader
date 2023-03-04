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

# Imports

```python
from grb_shader import GodMultiverse, RestoredUniverse, LocalVolume
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
%matplotlib widget
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Circle, Ellipse,Patch
from matplotlib.lines import Line2D
```

```python
param_file = 'ghirlanda2016_c_t90fit_r0_inc.yml'
pops_dir = '/data/eschoe/grb_shader/sims/only_coinc/221103/'
constant_temporal_profile = True
g = GodMultiverse(n_universes=5)
```

```python
#g.go_pops(param_file,pops_dir,constant_temporal_profile,catalog_selec=True)
```

```python
u1 = RestoredUniverse(pops_dir,pop_base_file_name='pop_1234')
u2 = RestoredUniverse(pops_dir,pop_base_file_name='pop_1244')
u3 = RestoredUniverse(pops_dir,pop_base_file_name='pop_1254')
u4 = RestoredUniverse(pops_dir,pop_base_file_name='pop_1264')
u5 = RestoredUniverse(pops_dir,pop_base_file_name='pop_1274')
```

```python
u5.pop.n_objects
```

```python
u1.pop.n_objects
```

```python

```

```python
lv = LocalVolume.from_lv_catalog()
```

```python
lv.read_population(u2.pop,unc_angle=0.)
```

# All galaxies

```python
fig, ax=lv.show_all_galaxies_area()
ax.grid(zorder=-100)
plt.figtext(x=0.5,y=0.22,ha='center',s='Right Ascension [deg]',fontsize='small')
plt.figtext(x=0.005,y=0.58,va='center',s='Declination [deg]',rotation='vertical',fontsize='small')
#ax.set_axisbelow()
plt.savefig('/data/eschoe/grb_shader/figs/lv_galaxies_loc.png',dpi=300)
```

```python
ax
```

# Read population

```python
lv.read_population(u1.pop,unc_angle=0.)
```

```python
#lv.read_population(u2.pop)
```

```python
#lv.read_population(u3.pop)
```

```python
#lv.read_population(u4.pop)
```

## Chosen galaxy

```python
lv.read_population(u5.pop,unc_angle=0.)
```

```python

fig, ax = lv.show_selected_galaxies(projection='astro degrees globe',center='-35d +10d',grb_color='C03')

fig.set_size_inches((8, 4))
ax.grid()

ax.scatter(u5.pop.ra,u5.pop.dec,
           transform=ax.get_transform("icrs"),
           color='k',
           edgecolor='k',
           s=1,
          alpha=0.2)

ax_zoom_rect = plt.axes(
    [0.01, 0.6, 0.3, 0.3],
    projection='astro degrees zoom',
    center='23.5d +30.65d',
    radius='1 deg')

ax_zoom_rect.scatter(
    u5.pop.ra,u5.pop.dec,
    transform=ax_zoom_rect.get_transform("icrs"),
    color='k',
    edgecolor='k',
    s=1,
    alpha=0.2)

ax_zoom_rect.coords['ra'].set_ticks(number=3)
ax_zoom_rect.coords['dec'].set_ticks(number=4)
ax_zoom_rect.coords['dec'].set_axislabel('Dec [deg]',minpad=0.6)
ax_zoom_rect.coords['dec'].set_axislabel_position('l')
ax_zoom_rect.coords['ra'].set_axislabel('RA [deg]',minpad=0.6)
ax_zoom_rect.coords['ra'].set_axislabel_position('b')

ax.mark_inset_axes(ax_zoom_rect)
ax.connect_inset_axes(ax_zoom_rect, 'upper right')
ax.connect_inset_axes(ax_zoom_rect, 'lower right')

lv.show_selected_galaxies(projection='astro degrees zoom',ax=ax_zoom_rect,grb_color='C03')

ax_zoom_rect2 = plt.axes(
    [0.71, 0.5, 0.3, 0.3],
    projection='astro degrees zoom',
    center='284.5d -30.5d',
    radius='5 deg')

#ax_zoom_rect2.set_xlabel('RA [dec]',labelpad=1,ha='center', va = 'top')
#ax_zoom_rect2.set_ylabel('Dec [dec]',labelpad=1,ha='left', va = 'center')


ax_zoom_rect2.scatter(
    u5.pop.ra,u5.pop.dec,
    transform=ax_zoom_rect2.get_transform("icrs"),
    color='k',
    edgecolor='k',
    s=1,
    alpha=0.2)

ax_zoom_rect2.coords['dec'].set_ticklabel_position('r')
ax_zoom_rect2.coords['dec'].set_axislabel('Dec [deg]',minpad=0.6)
ax_zoom_rect2.coords['dec'].set_axislabel_position('r')
ax_zoom_rect2.coords['ra'].set_ticklabel_position('t')
ax_zoom_rect2.coords['ra'].set_ticks(number=4)
ax_zoom_rect2.coords['ra'].set_axislabel('RA [deg]',minpad=0.6)
ax_zoom_rect2.coords['ra'].set_axislabel_position('t')
ax_zoom_rect2.coords['dec'].set_ticks(number=4)


lv.show_selected_galaxies(projection='astro degrees zoom',ax=ax_zoom_rect2,grb_color='C03')

ax.mark_inset_axes(ax_zoom_rect2)
ax.connect_inset_axes(ax_zoom_rect2, 'upper left')
ax.connect_inset_axes(ax_zoom_rect2, 'lower right')

#ax.scatter()

for axis in [ax, ax_zoom_rect, ax_zoom_rect2]:
    axis.grid(which='major')
    axis.set_facecolor('none')
    for key in ['ra', 'dec']:
        axis.coords[key].set_auto_axislabel(False)
        axis.coords.grid()

legend_elements = [Patch(facecolor='C00', edgecolor='C00',alpha=0.6,
                         label='Messier 33'),
                   Patch(facecolor='C01', edgecolor='C01',alpha=0.6,
                         label='SagdSph'),
                   Line2D([0], [0], marker='o', color='w', label='Not Coinciding GRBs',
                          markerfacecolor='grey', markersize=3,alpha=0.8),
                  Line2D([0], [0], marker='o', color='w', label='Coinciding GRBs',
                          markerfacecolor='C03', markersize=3,alpha=0.9)]



#ax.legend(handles=legend_elements,loc='lower right', bbox_to_anchor=(1, -0.2))

ax.legend(handles=legend_elements,loc='lower left', bbox_to_anchor=(-0.5, 0))
#plt.autoscale()
plt.savefig('/data/eschoe/grb_shader/figs/coinc_gals_with_all_grbs.png',dpi=300)
```

# Repeat with error around GRB


## From simulations

```python
sim_path = "/data/eschoe/grb_shader/sims/only_coinc/221110/"
```

```python
lv = LocalVolume.from_lv_catalog()
lv.sample_angles()
```

```python
unc = np.round(np.geomspace(1./60.,3,50),4)
pop_files = [f'pop_{1234+10*i}' for i in range(30)]
```

```python
area = np.sum(lv.areas)
print(area/(4*np.pi))
```

```python
u1 = RestoredUniverse(sim_path+str(unc[0]),pop_base_file_name=pop_files[0])
```

```python
#fig,ax = plt.subplots()
fractions_det = np.zeros(shape=(len(unc),len(pop_files)))
n_det = np.zeros(shape=(len(unc),len(pop_files)))
N_tot = np.zeros(shape=(len(unc),len(pop_files)))

for j in range(len(pop_files)):
    for i in range(len(unc)):
        u1 = RestoredUniverse(sim_path+str(unc[i]),pop_base_file_name=pop_files[j])
        N = len(u1.pop.selection)
        n=len(u1.pop.selection[u1.pop.selection == True])
        fractions_det[i][j] = n/N
        n_det[i][j] = n
        N_tot[i][j] = N


#ax.scatter(unc,fractions_det,s=6,alpha=0.5)
```

```python
np.average(N_tot)
```

```python
fig,ax = plt.subplots()
ax.grid()
for i in range(len(pop_files)):
    ax.scatter(unc,fractions_det[:,i],s=7,alpha=0.2,c='C00',edgecolors=None,zorder=3)
ax.set_xlabel(r'$\delta_{\mathrm{err}}$ [deg]')
ax.set_ylabel(r'$n_{\mathrm{coinc}}/N$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.axhline(area/(4*np.pi))


plt.savefig('/data/eschoe/grb_shader/figs/frac_coinc_err_xylog')
```

```python
fig,ax = plt.subplots(figsize=(7,4))
plt.rcParams.update({'font.size': 13})
ax.grid()
for i in range(len(pop_files)):
    sc = ax.scatter(unc,n_det[:,i],s=13,alpha=0.3,color='C00',zorder=3)
    sc.set_edgecolors('None')
ax.set_xlabel(r'$\delta_{\mathrm{err}}$ [deg]')
ax.set_ylabel(r'$n_{\mathrm{coinc}}$')
ax.set_yscale('log')
ax.set_xscale('log')

#ax.plot(unc,area/(4*np.pi)*np.average(N_tot)+2.5e2*unc**1.5,color='C01',label=r'$\propto {\delta_{\mathrm{err}}}^{1.5}$')
#ax.plot(unc,area/(4*np.pi)*np.average(N_tot)+2.5e2*unc**2,color='C02',label=r'$\propto {\delta_{\mathrm{err}}}^{2}$')
#ax.plot(unc,2.5e2*unc**1,color='C03',label=r'$\propto {\delta_{\mathrm{err}}}^{1.0}$')
ax.axhline(area/(4*np.pi)*np.average(N_tot),ls='--',color='C01',label=r'$n_{\mathrm{coinc, \, exp}}$')
ax.legend(fontsize=11)
plt.savefig('/data/eschoe/grb_shader/figs/n_coinc_err_xylog_fit.pdf',dpi=300)
```

```python

```

```python


fig, ax= lv.show_all_galaxies()
fig.set_size_inches((6, 5))
u1 = RestoredUniverse(sim_path+str(1.0396),pop_base_file_name=pop_files[0])

ax.grid()

ax_zoom = plt.axes(
    [0.08, 0.62, 0.3, 0.3],
    projection='astro degrees zoom',
    center='284d -31d',
    radius='4 deg')

ax.mark_inset_axes(ax_zoom)
ax.connect_inset_axes(ax_zoom, 'lower left')
ax.connect_inset_axes(ax_zoom, 'upper right')
ax_zoom.coords['ra'].set_auto_axislabel(False)
ax_zoom.coords['dec'].set_auto_axislabel(False)
ax_zoom.coords['ra'].set_ticklabel_position('t')
ax_zoom.coords['ra'].set_ticks(number=3)
ax_zoom.coords['dec'].set_ticks(number=4)
ax_zoom.grid()

ax_zoom2 = plt.axes(
    [0.62, 0.62, 0.3, 0.3],
    projection='astro degrees zoom',
    center='161d +13d',
    radius='4 deg')

ax.mark_inset_axes(ax_zoom2)
ax.connect_inset_axes(ax_zoom2, 'upper left')
ax.connect_inset_axes(ax_zoom2, 'lower right')
ax_zoom2.coords['ra'].set_auto_axislabel(False)
ax_zoom2.coords['dec'].set_auto_axislabel(False)
ax_zoom2.coords['ra'].set_ticklabel_position('t')
ax_zoom2.coords['dec'].set_ticklabel_position('r')
ax_zoom2.coords['ra'].set_ticks(number=3)
ax_zoom2.coords['dec'].set_ticks(number=4)
ax_zoom2.grid()

#plt.tight_layout(pad=12)

lv.show_all_galaxies(ax=ax_zoom)
lv.show_all_galaxies(ax=ax_zoom2)
print('Done')
for i in range(len(u1.pop.ra)):
    if u1.pop.ra[i] + 3 < 360:
        if u1.pop.ra[i] - 3 > 0:
            if u1.pop.dec[i]- 3 > - 90:
                if u1.pop.dec[i] + 3 < 90:
                    if u1.pop.selection[i]:
                        for axis in [ax,ax_zoom,ax_zoom2]:
                            c = Circle(
                                            (u1.pop.ra[i], u1.pop.dec[i]),
                                            radius=1.0396,
                                            alpha=0.6,
                                            color='C01',
                                            zorder=5,
                                            transform=axis.get_transform("icrs")
                                        )
                            axis.add_patch(c)
                    else:
                        for axis in [ax,ax_zoom,ax_zoom2]:
                            c = Circle(
                                            (u1.pop.ra[i], u1.pop.dec[i]),
                                            radius=1.0396,
                                            alpha=0.2,
                                            color='grey',
                                            zorder=3,
                                            transform=axis.get_transform("icrs")
                                        )
                            axis.add_patch(c)

legend_elements = [Patch(facecolor='C00', edgecolor='C00',alpha=0.6,
                         label='LV Galaxies'),
                   Patch(facecolor='grey', edgecolor='grey',alpha=0.2,
                         label='Not Coinciding GRBs'),
                   Patch(facecolor='C01', edgecolor='C01',alpha=0.6,
                         label='Coinciding GRBs')]

ax.legend(handles=legend_elements,loc='lower right', bbox_to_anchor=(1, -0.2))
plt.figtext(x=0.5,y=0.19,ha='center',s='Right Ascension [deg]',fontsize='small')
plt.figtext(x=0.005,y=0.5,va='center',s='Declination [deg]',rotation='vertical',fontsize='small')
plt.savefig('/data/eschoe/grb_shader/figs/1deg_error_simulation.png',dpi=300)
plt.show()
```
