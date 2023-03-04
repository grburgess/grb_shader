---
jupyter:
  jupytext:
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns; #sns.set_theme()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params,font='serif')
from cmdstanpy import CmdStanModel
import cmdstanpy
import arviz as av
import scipy.stats as stats
fig_path = '/data/eschoe/grb_shader/fit_t90_distribution/'
```

```python
%matplotlib widget

t90_data = pd.read_csv('t90_data.txt',header=0,delimiter='\t',engine='python')
ax =sns.rugplot(data=np.log10(t90_data), x="t90")
##ax.set_xscale('log')
sns.histplot(data=np.log10(t90_data), x="t90",bins=100)
ax.set_xlabel(r'log$_{10}$($T_{90}$) [s]')
plt.tight_layout()
```

```python
stan_model = CmdStanModel(stan_file="stan/model_mixture3_parallel.stan",cpp_options={"STAN_THREADS": True})
#print(stan_model.code())
```

```python
#print(stan_model.code())
```

```python
t90_data = pd.read_csv('t90_data.txt',header=0,delimiter='\t',engine='python')
#convert from data frame to numpy arrays
t90 = t90_data['t90'].to_numpy()
t90_err = t90_data['t90_error'].to_numpy()
print(t90_err)

# prepare data
input_data = {}
input_data["N"] = len(t90)
input_data["t90"] = t90#np.log10(t90)
#approximated error of log10(x)
input_data["t90_err"] = t90_err
#np.abs(t90_err/(t90*np.log(10)))
```

```python
fit = stan_model.sample(data=input_data, chains=4, seed=1223, iter_warmup=1000, iter_sampling=1500,show_progress=True,threads_per_chain=5)
```

```python
fit = cmdstanpy.from_csv('/data/eschoe/grb_shader/fit_t90_distribution/sims/20220613_mixed3_parallel')
```

```python
print(fit.diagnose())
```

```python
fit.summary()[:7]
```

```python
fit.save_csvfiles(dir='/data/eschoe/grb_shader/fit_t90_distribution/sims/20220613_mixed3_parallel/')
```

# Posterior distribution t90

```python
stan_var_dict = fit.stan_variables()
mu = stan_var_dict['mu']
sigma = stan_var_dict['sigma']
theta = stan_var_dict['theta']

t = 0 
print(double_gaussian(t, theta, mu, sigma)[0])
print(double_gaussian2(t, theta[0], mu[0], sigma[0]))
```

```python
def double_gaussian(t, theta, mu, sigma):
    return (theta*stats.norm.pdf(t,mu[:,0], sigma[:,0])+(1-theta)*stats.norm.pdf(t,mu[:,1],sigma[:,1]))

def from_conf_int2upp_low(probs):
    upp_low = np.zeros(shape=(len(probs), 2))
    for i in range(len(probs)):
        lower = (100 - probs[i])/2 #%
        upper = (100 - probs[i])/2 + probs[i] #%
        upp_low[i] = np.array([lower/100., upper/100.])
    return upp_low
```

```python
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(np.log10(t90_data['t90']), bins=60,density=True,histtype='step',linewidth='1.5')

probs= np.sort(np.array([68,95,99.9])) #%
#probs= np.sort(np.array([68])) #%
upp_low = from_conf_int2upp_low(probs)
cm = plt.get_cmap('Blues')
colors = np.linspace(0.3,0.6,len(probs))

ax.set_xlabel(r'log$_{10}$(T$_{90}$)')
log10_t_obs = np.linspace(-2,3,100)

#averages:
#a = 0.200
#mu_s = 0.12
#sigma_s = 0.51
#mu_l =1.400
#sigma_l = 0.45

#print(np.shape(new_quantities.stan_variable('theta_prior')))

t_arr = np.linspace(-2,3,1000)
for k in reversed(range(0, len(probs))):
    f_lower = np.zeros_like(t_arr)
    f_upper = np.zeros_like(t_arr)
    for j in range(len(t_arr)):
        f_arr = double_gaussian(t_arr[j],theta,mu,sigma)
        f_lower[j] = np.quantile(f_arr,upp_low[k][0])
        f_upper[j] = np.quantile(f_arr,upp_low[k][1])
    
    ax.fill_between(x = t_arr,y1 =(f_lower),y2=(f_upper),color=cm(colors[len(probs)-k-1]),alpha=0.5,linewidth=0)


#set labels
legend_elements = []
legend_elements += [Patch(facecolor='w', edgecolor=cm(0.8),label='Data',linewidth='1.5')]
for j in reversed(range(len(probs))):
    legend_elements += [Patch(facecolor=cm(colors[len(probs)-j-1]),label=f'{j+1}$\sigma$',linewidth='1.5')]

ax.set_xlabel(r'log$_{10}$(T$_{90}$)')
ax.set_ylabel(r'Number GRBs')
ax.legend(handles=legend_elements)
plt.tight_layout()

plt.savefig(fig_path + 'posterior_f_plot.png',dpi=300)

#ax.set_ylim(0,1)

#ax.plot(log10_t_obs, (a*stats.norm.pdf(log10_t_obs,mu_s, sigma_s)+(1-a)*stats.norm.pdf(log10_t_obs,mu_l,sigma_l)))
```

# Trace plots

```python
av.plot_trace(fit, var_names=["theta", "mu","sigma"]);
#av.plot_trace(fit, var_names=["log10t90_true"[898]]);
plt.tight_layout()
```

# Marginal plots

```python
fig, ax = plt.subplots(5,5,figsize=(12,9))
av.plot_pair(fit,ax=ax,var_names=["mu","sigma",'theta'],marginals=True,kind="scatter",
                  scatter_kwargs={"marker":".","markeredgecolor":"k",'markeredgewidth':0.1,"alpha":0.05},
             point_estimate='mode',point_estimate_kwargs={'ls':'--','lw':1},
            point_estimate_marker_kwargs={'marker':'s','s':15})
#ax = av.plot_pair(fit, ax = ax,var_names=["mu","sigma",'theta'], marginals=True,kind="kde",backend_kwargs={"cmap":plt.cm.viridis})


#ax = fig.get_axes
ax[4][0].set_xlabel(r'$\mu_1$')
ax[4][1].set_xlabel(r'$\mu_2$')
ax[4][2].set_xlabel(r'$\sigma_1$')
ax[4][3].set_xlabel(r'$\sigma_2$')
ax[4][4].set_xlabel(r'$\theta$')

ax[0][0].set_ylabel(r'$\mu_1$')
ax[1][0].set_ylabel(r'$\mu_2$')
ax[2][0].set_ylabel(r'$\sigma_1$')
ax[3][0].set_ylabel(r'$\sigma_2$')
ax[4][0].set_ylabel(r'$\theta$')

#ax[1][0].set_ylabel(r'$\mu_2$')
#ax.set_ylabel(r'$\mu_2$')
plt.tight_layout()
plt.savefig(fig_path + 'pair_plot.png',dpi=300)
plt.show()
```

```python
print(np.shape(ax))
```

# Posterior predictive test

```python
#stan_model.compile()
new_quantities = stan_model.generate_quantities(input_data,fit)
```

```python
print(min(new_quantities.stan_variable('t90_post_pred')[100]))
```

```python
Nreps = 200

fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(t90_data['t90']), bins=60,density=True,histtype='step',color='C00')

for i in range(Nreps):
    u = np.random.randint(0,4000)
    t90_post_pred = new_quantities.stan_variable('t90_post_pred')[u]
    t90_post_pred_pos = t90_post_pred[t90_post_pred > 0.]
    #y_true = new_quantities.stan_variable('log10t90_post_pred')[u]
    #print(y_post_pred)
    
    ax.hist(np.log10(t90_post_pred_pos),alpha=0.1,density=True,bins=bins,color='C01',histtype='step',range=(-3,3))



n, bins, patches = ax.hist(np.log10(t90_data['t90']), bins=60,density=True,histtype='step',color='C00',label='Data')

ax.hist([0],bins=[1],color='C01',histtype='step',label='PPC')

ax.set_xlabel(r'log$_{10}$(T$_{90}$)')
ax.legend()
```

## plot posterior predictive histogram with quantiles

```python
#Size of posterior sample in total
n_posterior = fit.chains * fit.num_draws_sampling
#Number of bins
n_bins = 50
#iterations over posterior predictive samples
iters = range(0,n_posterior,5)
#number of iterations 
n_iter = len(iters)
#list of confidence intervals that are plotted as boxes
probs= np.sort(np.array([68,95,99.9])) #%

cm = plt.get_cmap('Blues')
colors = np.linspace(0.3,0.7,len(probs)+3)

def from_conf_int2upp_low(probs):
    upp_low = np.zeros(shape=(len(probs), 2))
    for i in range(len(probs)):
        lower = (100 - probs[i])/2 #%
        upper = (100 - probs[i])/2 + probs[i] #%
        upp_low[i] = np.array([lower/100., upper/100.])
    return upp_low

def double_gauss(t,a,mu1,sigma1,mu2,sigma2):
    return (a*stats.norm.pdf(t,mu1, sigma1)+(1-a)*stats.norm.pdf(t,mu2,sigma2))


fig, ax = plt.subplots(figsize=(8,5))
n, bins, patches = ax.hist(np.log10(t90_data['t90']), bins=n_bins,histtype='step',color=cm(0.8),zorder=2,linewidth=1.5,density=True)


bin_width= bins[1] - bins[0]

counts = np.zeros(shape=(n_iter,n_bins))
counter = 0
#create histograms for set of generated posterior predictive samples
for i in iters:
    y_post_pred = new_quantities.stan_variable('t90_post_pred')[i]
    hist, bin_edges = np.histogram(np.log10(y_post_pred[y_post_pred > 0.]),bins=bins,density=True)
    counts[counter] = hist
    counter += 1

upp_low = from_conf_int2upp_low(probs)

#mu = fit.stan_variable('mu')
#sigma = fit.stan_variable('sigma')
#theta = fit.stan_variable('theta')

"""for j in reversed(range(len(probs))):
    mu1_low = np.quantile(mu[:,0],upp_low[j][0])
    mu1_upp = np.quantile(mu[:,0],upp_low[j][1])
    mu2_low = np.quantile(mu[:,1],upp_low[j][0])
    mu2_upp = np.quantile(mu[:,1],upp_low[j][1])
    sigma1_low = np.quantile(sigma[:,0],upp_low[j][0])
    sigma1_upp = np.quantile(sigma[:,0],upp_low[j][1])
    sigma2_low = np.quantile(sigma[:,1],upp_low[j][0])
    sigma2_upp = np.quantile(sigma[:,1],upp_low[j][1])
    theta_low = np.quantile(theta,upp_low[j][0])
    theta_upp = np.quantile(theta,upp_low[j][1])
    
    t = np.linspace(-3,3,1000)
    #ax.plot(t, double_gauss(t,theta_low,mu1_low,sigma1_low,mu2_low,sigma2_low))
    #print(theta_low,mu1_low,sigma1_low,mu2_low,sigma2_low)
    
    ax.fill_between(x = t,y1 =double_gauss(t,theta_low,mu1_low,sigma1_low,mu2_low,sigma2_low),y2=(double_gauss(t,theta_upp,mu1_upp,sigma1_upp,mu2_upp,sigma2_upp)),color=cm(colors[len(probs)-j-1]))"""


quant_counts = np.zeros(shape=(n_bins,len(probs),2))
#compute the quantiles to compute the confidence intervals
for i in range(n_bins):
    for j in reversed(range(len(probs))):
        cred_low = np.quantile(counts[:,i],upp_low[j][0])
        cred_upp = np.quantile(counts[:,i],upp_low[j][1])
        quant_counts[i][j] = np.array([cred_low, cred_upp])
        ax.fill_between(x = (bins[i],bins[i+1]),y1 =(cred_low),y2=(cred_upp),color=cm(colors[len(probs)-j-1]))

#set labels
legend_elements = []
legend_elements += [Patch(facecolor='w', edgecolor=cm(0.8),label='Data',linewidth='1.5')]
for j in reversed(range(len(probs))):
    legend_elements += [Patch(facecolor=cm(colors[len(probs)-j-1]),label=f'PPC: {j+1}$\sigma$',linewidth='1.5')]

ax.set_xlabel(r'log$_{10}$(T$_{90}$)')
ax.set_ylabel(r'Number GRBs')
ax.legend(handles=legend_elements)
plt.tight_layout()
#plt.show()

plt.savefig(fig_path + 'PPC_plot.png',dpi=300)
```

```python
np.shape(fit.stan_variable('mu')[:,0])
```

# QQ Plot

```python
#Size of posterior sample in total
n_posterior = fit.chains * fit.num_draws_sampling
#Number of bins
n_bins = 60
#iterations over posterior predictive samples
iters = range(0,n_posterior,5)
#number of iterations 
n_iter = len(iters)
#list of confidence intervals that are plotted as boxes
probs= np.sort(np.array([68,95,99])) #%

cm = plt.get_cmap('Blues')
colors = np.linspace(0.3,0.7,len(probs)+2)

def from_conf_int2upp_low(probs):
    upp_low = np.zeros(shape=(len(probs), 2))
    for i in range(len(probs)):
        lower = (100 - probs[i])/2 #%
        upper = (100 - probs[i])/2 + probs[i] #%
        upp_low[i] = np.array([lower/100., upper/100.])
    return upp_low


fig, ax = plt.subplots(figsize=(8,5))
n, bins, patches = ax.hist(np.log10(t90_data['t90']), bins=n_bins,histtype='step',color='C00',zorder=2,linewidth=1.5,cumulative=True)

counts = np.zeros(shape=(n_iter,n_bins))
counter = 0
#create histograms for set of generated posterior predictive samples
for i in iters:
    y_post_pred = new_quantities.stan_variable('t90_post_pred')[i]
    hist, bin_edges,patches = ax.hist(np.log10(y_post_pred),bins=bins,cumulative=True)
    counts[counter] = hist
    counter += 1

upp_low = from_conf_int2upp_low(probs)

#quant_counts = np.zeros(shape=(n_bins,len(probs),2))
#compute the quantiles to compute the confidence intervals
plt.close(fig)
fig, ax = plt.subplots(figsize=(8,5))
for j in reversed(range(len(probs))):
    cred_low = np.zeros(n_bins)
    cred_upp = np.zeros(n_bins)
    for i in range(n_bins):
        cred_low[i] = np.quantile(counts[:,i],upp_low[j][0])
        cred_upp[i] = np.quantile(counts[:,i],upp_low[j][1])
    ax.fill_between(x = n, y1 =(cred_low),y2=(cred_upp),color=cm(colors[len(probs)-j-1]))

    
##set labels
legend_elements = []
for j in reversed(range(len(probs))):
    legend_elements += [Patch(facecolor=cm(colors[len(probs)-j-1]),label=f'PPC: {j+1}$\sigma$',linewidth='1.5')]

pl, = ax.plot(np.arange(min(n),max(n)),np.arange(min(n),max(n)),label='Expected',ls='--',color=cm(0.8))
legend_elements += [pl]
    
ax.set_xlabel(r'Cum. Obs. Number GRBs')
ax.set_ylabel(r'Cum. Model Number GRBs')
ax.legend(handles=legend_elements)


```

# Rank plot

```python
fig, ax = plt.subplots(2,1,figsize=(7,7))
ax = av.plot_rank(fit, var_names=['mu'],ax=ax)
plt.tight_layout()
np.shape(fit.draws()[0])
```

```python
#prior predictive test
new_quantities = stan_model.generate_quantities(data=input_data, mcmc_sample=fit)
```

# Prior predictive check

```python
print(np.shape(new_quantities.stan_variable('theta_prior')))


Nreps = 100

fig, ax = plt.subplots()
ax.hist(np.log10(t90_data['t90']), bins=42,density=True)

for i in range(Nreps):
    #u = np.random.randint(0,4000)
    a = new_quantities.stan_variable('theta_prior')[i]
    mu_s = new_quantities.stan_variable('mu_prior')[:,0][i]
    mu_l = new_quantities.stan_variable('mu_prior')[:,1][i]
    sigma_s = new_quantities.stan_variable('sigma_prior')[:,0][i]
    sigma_l = new_quantities.stan_variable('sigma_prior')[:,1][i]
    ax.plot(log10_t_obs, (a*stats.norm.pdf(log10_t_obs,mu_s, sigma_s)+(1-a)*stats.norm.pdf(log10_t_obs,mu_l,sigma_l)),alpha=0.1,color='C01')
    
ax.set_ylim(0,1)
```

```python
input_data["N"]*5
```

```python

```
