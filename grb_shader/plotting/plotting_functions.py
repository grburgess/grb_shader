import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def logbins_scaled_histogram(quant,intervals=None,n_bins=50,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if intervals is None:
        intervals = np.logspace(np.log10(min(quant)),np.log10(max(quant)),n_bins)
    #compute histogram
    counts, bins = np.histogram(quant,bins=intervals)

    ax.stairs(counts/np.diff(intervals),intervals,fill=True,alpha=0.7)
    
    return intervals

def logbins_norm_histogram(quant,intervals=None,n_bins=50,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if intervals is None:
        intervals = np.logspace(np.log10(min(quant)),np.log10(max(quant)),n_bins)

    ax.hist(quant,bins=intervals,fill=True,alpha=0.7,density=True)
    
    return intervals
    
def log10normal(x,mu,sigma,K=1):
    return K/(np.sqrt(2*np.pi)*sigma*x*np.log(10)) * np.exp(-1./2. * ((np.log10(x)-mu)/sigma)**2)

def array_to_cmap(values, cmap, use_log=False):
    """
    Generates a color map and color list that is normalized
    to the values in an array. Allows for adding a 3rd dimension
    onto a plot

    :param values: a list a values to map into a cmap
    :param cmap: the mpl colormap to use
    :param use_log: if the mapping should be done in log space

    """

    if use_log:

        norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))

    else:

        norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))

    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgb_colors = [cmap.to_rgba(v) for v in values]

    return cmap, rgb_colors