from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincc
from astropy.coordinates import SkyCoord

import popsynth as ps
from popsynth.utils.cosmology import Cosmology
from cosmogrb.universe.survey import Survey
from cosmogrb.utils.package_utils import get_path_of_data_file
from gbmgeometry import GBM, PositionInterpolator
from threeML import *
from threeML.utils.OGIP.response import OGIPResponse
from astromodels.functions import Cutoff_powerlaw_Ep

from .catalog import LocalVolume

class GRBAnalysis_constant(object):

    def __init__(
        self,
        dir_sim,
        name_chosen_grb,
        pulse_profile = False,
        pop_file = 'pop_1234.h5',
        survey_file = 'survey_1234.h5',
        grb_folder = '_1234'
        ):

        self._name_chosen_grb = name_chosen_grb
        
        self._dir_sim = Path(dir_sim)

        self._pop_file = self._dir_sim / pop_file

        self._pop = ps.Population.from_file(self._pop_file)

        self._survey_file = self._dir_sim / survey_file

        self._survey = Survey.from_file( self._survey_file)

        self._dir_grbs = self._dir_sim / grb_folder

        self._pulse_profile = pulse_profile

        self._find_closest_dets()

        self._tmin = -0.5 #T0=0 - no offset simulated
        #choose time interval for fitting from maximum triggered_time_scales
        #tmax = max(survey_no_selec_pulse[chosen_grb].detector_info.extra_info['triggered_time_scales'])
        self._tmax = self.duration_latent + 0.5

        self._sampled = False
        
        #start and stop times of the intervals found by binning method in _find_time_intervals
        self._start_times = None
        self._stop_times = None

        #load local volume galaxies catalog
        self._catalog = LocalVolume.from_lv_catalog()

        #self._galaxy = 

    
    @property
    def tmin(self):
        return self._tmin
    
    @property
    def tmax(self):
        return self._tmax

    @property
    def pop(self):
        return self._pop

    @property
    def survey(self):
        return self._survey

    @property
    def chosen_grb(self):
        return self.survey[self._name_chosen_grb].grb
    
    @property
    def z_latent(self):
        return self.chosen_grb.z

    @property
    def ra(self):
        return self.chosen_grb.ra

    @property
    def dec(self):
        return self.chosen_grb.dec

    @property
    def alpha_latent(self):
        return self.chosen_grb.source_params['alpha']

    @property
    def peak_flux_latent(self):
        return self.chosen_grb.source_params['peak_flux']
    
    @property
    def Eiso_latent(self):
        
        c = Cosmology()
        
        Eiso = self.peak_flux_latent/((1+self.z_latent)**2) * self.duration_latent * 4 * np.pi * (c.luminosity_distance(self.z_latent))**2
        
        return Eiso

    @property
    def duration_latent(self):
        return self.chosen_grb.duration

    @property
    def bayes_results(self):
        return self._bayes_results

    def norris(self,x, K, t_start, t_rise, t_decay):
        if x > t_start:
            return (
                K
                * np.exp(2 * np.sqrt(t_rise / t_decay))
                * np.exp(-t_rise / (x - t_start) - (x - t_start) / t_decay)
            )
        else:
            return 0.0

    def _compute_ep_latent_pulse(self,t):
        return self.chosen_grb.source_params['ep_start'] / (1 + t / self.chosen_grb.source_params['ep_tau'])

    def _compute_K_exp_pulse(self,t):
        # normalization factor of astromodels Ep_cutoff_powerlaw
        ep = self._compute_ep_latent_pulse(t)

        K = self.norris(
            t, K=self.chosen_grb.source_params['peak_flux'], t_start=0.0, t_rise=self.chosen_grb.source_params['trise'], t_decay=self.chosen_grb.source_params['tdecay']
        )
                
        a = 10 * (1 + self.z_latent)
        b = 1e4 * (1 + self.z_latent)

        #cutoff energy
        ec = ep / (2 + self.alpha_latent)

        #for conversion of popsynth flux to keV/cm^2/s
        erg2keV = 6.24151e8

        #integral of E*f_cpl
        i1 = gammaincc(2.0 + self.alpha_latent, a / ec) * gamma(2.0 + self.alpha_latent)
        i2 = gammaincc(2.0 + self.alpha_latent, b / ec) * gamma(2.0 + self.alpha_latent)

        intflux = -ec * ec * (i2 - i1)
        norm = K * erg2keV / (intflux)

        #from comparison of cpl in cosmogrb and ep_cpl definition in astromodels
        factor = ((2+self.alpha_latent)*(1+self.z_latent)/ep)**self.alpha_latent

        self._K_exp = norm*factor

    def _compute_K_exp_const(self):
        # normalization factor of astromodels Ep_cutoff_powerlaw
                
        a = 10 * (1 + self.z_latent)
        b = 1e4 * (1 + self.z_latent)

        #cutoff energy
        ec = self.ep_latent() / (2 + self.alpha_latent)

        #for conversion of popsynth flux to keV/cm^2/s
        erg2keV = 6.24151e8

        #integral of E*f_cpl
        i1 = gammaincc(2.0 + self.alpha_latent, a / ec) * gamma(2.0 + self.alpha_latent)
        i2 = gammaincc(2.0 + self.alpha_latent, b / ec) * gamma(2.0 + self.alpha_latent)

        intflux = -ec * ec * (i2 - i1)
        norm = self.peak_flux_latent * erg2keV / (intflux)

        #from comparison of cpl in cosmogrb and ep_cpl definition in astromodels
        factor = ((2+self.alpha_latent)*(1+self.z_latent)/self.ep_latent())**self.alpha_latent

        self._K_exp = norm*factor

    def K_exp(self,t=None):
        if self._pulse_profile:
            if t is None:
                raise Exception('Specify time for pulse profile')
            else:
                self._compute_K_exp_pulse(t)
                return self._K_exp
        else:
            self._compute_K_exp_const()
            return self._K_exp

    def ep_latent(self,t=None):
        if self._pulse_profile:
            if t is None:
                raise Exception('Specify time for pulse profile')
            else:
                return self._compute_ep_latent_pulse(t)
        else:
            return self.chosen_grb.source_params['ep']

    def _find_closest_dets(self):
        # find three closest NaI and one closest BGO detector

        # read used position history file from cosmogrb and generate position interpolator
        posthist_file_cosmogrb = get_path_of_data_file('posthist.h5')
        pi = PositionInterpolator.from_poshist_hdf5(posthist_file_cosmogrb)

        # get sampled orbit time and corresponding position of spacecraft in orbit and orientation of GV
        time_adjustment = self.chosen_grb.time_adjustment
        myGBM = GBM(pi.quaternion(time_adjustment),sc_pos=pi.sc_pos(time_adjustment)*u.m)

        ra = self.chosen_grb.ra
        dec = self.chosen_grb.dec
        grb = SkyCoord(ra=ra,dec=dec,frame='icrs', unit='deg')

        # find out if GRB is occulted by sun
        self._is_occulted = pi.is_earth_occulted(ra=ra,dec=dec,t=time_adjustment)
        if self._is_occulted:
            print('GRB is occulred by Earth')
        #print(f'Is occulted by sun: {is_occulted}')

        #get separation angles between grb and detectors
        sep_angles = myGBM.get_separation(grb)
        #angles to NaI detectors
        sep_angles_n = dict(sep_angles[:-2])
        #angles to BGO detectors
        sep_angles_b = dict(sep_angles[12:])
        #compute 3 closest NaI detectors
        closest_n = list(dict(sorted(sep_angles_n.items(), key=lambda x: x[1])))[:3]
        #compute 1 closest BGO detector
        closest_b = list(dict(sorted(sep_angles_b.items(), key=lambda x: x[1])))[0]
        #list all 4 detectors
        self._closest_dets = closest_n + [closest_b]
    
    @property
    def closest_dets(self):
        return self._closest_dets

    @property 
    def is_occulted_earth(self):
        return self._is_occulted

    def plot_photon_lc(self,savefig=False,dir_figs=None,axes=None,dt=None,**kwargs):

        if axes is None:

            fig, axes = plt.subplots(5,3,sharex=True,sharey=True,figsize=(7,7))

        else:

            try:
                # Single plot
                fig = axes.get_figure()
            except AttributeError:
                # Subplots
                try:
                    # 1D grid
                    fig = axes[0].get_figure()
                except AttributeError:
                    # 2D grid
                    fig = axes[0][0].get_figure()


        print(f'Chosen time interval: {self._tmin} - {self._tmax} s')

        row=0
        col = 0

        if dt is None:
            dt = (self.duration_latent + 1.)/50.

        for k,v  in self.chosen_grb.items():
            ax = axes[row][col]

            lightcurve =v['lightcurve']
            lightcurve.display_background(dt=dt,tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1, color="C00",label='Background',**kwargs)
            lightcurve.display_lightcurve(dt=dt, tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color='C01',label='Source+Background',**kwargs)
            lightcurve.display_source(dt=dt,ax=ax,lw=1,color="C02",label='Source')
            
            ax.set_xlim(self._tmin, self._tmax)
            ax.set_title(k,size=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            if col < 2:
                col+=1
            else:
                row+=1
                col=0
                
            if col == 1:
                ax.set_ylabel('Rate [cnts/s]',size=8)
            if row == 4:
                ax.set_xlabel('Time [s]',size=8)
            
            ax.xaxis.set_tick_params(labelsize=8,size=5)
            ax.yaxis.set_tick_params(labelsize=8,size=5)  

        axes[3,2].xaxis.set_tick_params(size=5,labelbottom=True)  
            
        axes[4,2].set_visible(False)
        
        h, l = axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower right',bbox_to_anchor=[0.985, 0.084])

        plt.tight_layout()

        if savefig:

            if dir_figs is None:

                dir_figs = self._dir_sim / '3ml_fits'
            
            plt.savefig(str(dir_figs / f'{self._name_chosen_grb}_photon_lightcurve.pdf'))

        return fig

    def plot_photon_spectrum(self,savefig=False,dir_figs=None,axes=None,**kwargs):

        if axes is None:

            fig, axes = plt.subplots(5,3,sharex=False,sharey=False,figsize=(7,7))

        else:

            try:
                # Single plot
                fig = axes.get_figure()
            except AttributeError:
                # Subplots
                try:
                    # 1D grid
                    fig = axes[0].get_figure()
                except AttributeError:
                    # 2D grid
                    fig = axes[0][0].get_figure()

        print(f'Chosen time interval: {self._tmin} - {self._tmax} s')

        row=0
        col = 0

        lightcurves = []

        for k,v  in self.chosen_grb.items():
            ax = axes[row][col]
            
            lightcurve = v['lightcurve']
        
            lightcurve.display_count_spectrum_background(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1, color="C00",label='Background',**kwargs)
            lightcurve.display_count_spectrum(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color='C01',label='Source+Background',**kwargs)
            lightcurve.display_count_spectrum_source(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color="C02",label='Source',**kwargs)
            
            #ax.set_xlim(10, 30)
            #ax.set_xlim(8,40000)

            pha = lightcurve._bin_spectrum(tmin=self._tmin, tmax=self._tmax, times=lightcurve.times, pha=lightcurve.pha)
            ax.set_ylim(bottom=1e-1,top=max(pha)*2)

            ax.set_title(k,size=8)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            lightcurves.append(lightcurve)
            
            if col < 2:
                col+=1
            else:
                row+=1
                col=0
                
            if col == 1:
                ax.set_ylabel('Rate [cnts/s/keV]',size=8)
            if row == 4:
                ax.set_xlabel('Energy [keV]',size=8)
                
            ax.xaxis.set_tick_params(labelsize=8,size=5)
            ax.yaxis.set_tick_params(labelsize=8,size=5)  

        h, l = axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower right',bbox_to_anchor=[0.985, 0.084])

        axes[3,2].xaxis.set_tick_params(size=5,labelbottom=True)  

        axes[4,2].set_visible(False)  

        plt.tight_layout()

        if savefig:

            if dir_figs is None:

                dir_figs = self._dir_sim / '3ml_fits'
            
            plt.savefig(str(dir_figs / f'{self._name_chosen_grb}_photon_spectrum.pdf'))

        return fig
    
    @property
    def duration_bayesian_blocks(self):
        if len(self._start_times)>2:
            duration = []
            for i in range(1, len(self._start_times)-1):
                duration += [self._stop_times[i] - self._start_times[i]]
            return duration
        else:
            raise Exception('No source found')
    
    def energy_integrated_fluxes(self,upper=False,lower=False):
        
        fluxes = np.zeros(len(self.bayes_results))
        
        for i in range(len(self.bayes_results)):
            if upper:
                fluxes[i] = self.bayes_results[i].results.get_flux(
                    ene_min=8*u.keV, 
                    ene_max = 40*u.MeV,
                    use_components=True,
                    flux_unit="erg/(cm2 s)")['hi bound']['ps: total'].value
            elif lower:
                fluxes[i] = self.bayes_results[i].results.get_flux(
                    ene_min=8*u.keV, 
                    ene_max = 40*u.MeV,
                    use_components=True,
                    flux_unit="erg/(cm2 s)")['low bound']['ps: total'].value
            else:
                fluxes[i] = self.bayes_results[i].results.get_flux(
                    ene_min=8*u.keV, 
                    ene_max = 40*u.MeV,
                    use_components=True,
                    flux_unit="erg/(cm2 s)")['flux']['ps: total'].value
                
        return fluxes
    
    def Eiso_fromspectralfit_at_z(self,z=None,upper=False,lower=False):
        c = Cosmology()
        
        if z is None:
            
            z = self.z_latent
        
        if upper==True:
            #68% interval upper
            Eiso = np.sum(self.energy_integrated_fluxes(upper=True) * self.duration_bayesian_blocks) * 4 * np.pi * (c.luminosity_distance(z))**2
        elif lower == True:
            #68% interval lower
            Eiso = np.sum(self.energy_integrated_fluxes(lower=True) * self.duration_bayesian_blocks) * 4 * np.pi * (c.luminosity_distance(z))**2
        else:
            #68% interval median
            Eiso = np.sum(self.energy_integrated_fluxes() * self.duration_bayesian_blocks) * 4 * np.pi * (c.luminosity_distance(z))**2
        
        return Eiso
    
    def Eiso_fromspectralfit_at_d(self,d,upper=False,lower=False):
        
        Mpc2cm = 3.086e+24
        d_cm = d*Mpc2cm
        
        if upper==True:
            #68% interval upper
            Eiso = np.sum(self.energy_integrated_fluxes(upper=True) * self.duration_bayesian_blocks) * 4 * np.pi * (d_cm)**2
        elif lower == True:
            #68% interval lower
            Eiso = np.sum(self.energy_integrated_fluxes(lower=True) * self.duration_bayesian_blocks) * 4 * np.pi * (d_cm)**2
        else:
            #median
            Eiso = np.sum(self.energy_integrated_fluxes() * self.duration_bayesian_blocks) * 4 * np.pi * (d_cm)**2
        
        return Eiso
    
    def Ep_bestfit(self,upper=False,lower=False,index=0):
        results = self.bayes_results[index].results
        if upper:
            Ep = results.get_equal_tailed_interval('ps.spectrum.main.Cutoff_powerlaw_Ep.xp')[-1]
        elif lower:
            Ep = results.get_equal_tailed_interval('ps.spectrum.main.Cutoff_powerlaw_Ep.xp')[0]
        else:
            Ep = results.get_data_frame()['value']['ps.spectrum.main.Cutoff_powerlaw_Ep.xp']
        return Ep

    def _find_time_intervals(self,p0):

        # choose time intervals from closest detector with highest count rate
        det = self._closest_dets[0]

        #read response file
        response = OGIPResponse(self._dir_grbs / f'rsp_{self._name_chosen_grb}_{det}.rsp')

        #read temporally unbinned simulated data
        self._tte_time_binning =  TimeSeriesBuilder.from_gbm_tte(
                name=f'{det}_tte',
                tte_file=self._dir_grbs / f'tte_{self._name_chosen_grb}_{det}.fits',
                rsp_file=response,
                poly_order = 0
                )

        #use bayesian block method to find time bins
        self._tte_time_binning.create_time_bins(start=-10.,stop=30,method='bayesblocks',p0=p0)

        start_times = self._tte_time_binning.bins.start_times
        stop_times = self._tte_time_binning.bins.stop_times
        
        self._start_times = []
        self._stop_times = []
        for i in range(len(start_times)):
            if start_times[i] != stop_times[i]:
                self._start_times += [start_times[i]]
                self._stop_times += [stop_times[i]]
            else:
                print('Found twice same time')
            

    def _fit_cpl_ep(self, plot_lightcurve, plot_count_spectrum, savefigs, dir_figs, n_live_points, p0):
        
        self._find_time_intervals(p0)
        
        if len(self._start_times) > 2:
        
            # define first and last time bin as background bins 
            # fit in each other time bin, the CPL spectrum

            self._bayes_results = []

            for i in range(1, len(self._start_times)-1):

                fluence_plugins = []

                #fit data of all closest detectors together
                for j in range(len(self._closest_dets)):
                    
                    # read response file
                    response = OGIPResponse(self._dir_grbs / f'rsp_{self._name_chosen_grb}_{self._closest_dets[j]}.rsp')

                    #read temporally unbinned simulated data
                    tte =  TimeSeriesBuilder.from_gbm_tte(
                    name=f'{self._closest_dets[j]}_tte',
                    tte_file=self._dir_grbs / f'tte_{self._name_chosen_grb}_{self._closest_dets[j]}.fits',
                    rsp_file=response,
                    poly_order = 0
                    )

                    tte.read_bins(self._tte_time_binning)

                    #set time interval of source
                    tte.set_active_time_interval(f'{np.round(self._start_times[i],4)}-{np.round(self._stop_times[i],4)}')
                    #set first and last time bin as background
                    tte.set_background_interval(f'{self._start_times[0]}-{self._stop_times[0]}',f'{self._start_times[-1]}-{self._stop_times[-1]}')

                    fluence_plugin = tte.to_spectrumlike()

                    #choose active energy measurement region
                    if self._closest_dets[j][0] == 'n':

                        fluence_plugin.set_active_measurements("9-900")

                    elif self._closest_dets[j][0] == 'b':

                        fluence_plugin.set_active_measurements("250-30000")
                    
                    #make sure there is at least one count in each bin
                    fluence_plugin.rebin_on_background(1)
                    
                    fluence_plugins.append(fluence_plugin)

                    if dir_figs is None:

                        dir_figs = self._dir_sim / '3ml_fits'
                    else:
                        dir_figs = Path(dir_figs)

                    if plot_lightcurve:

                        tte.view_lightcurve(use_binner=True)

                        if savefigs:

                            plt.savefig(dir_figs / f'{self._name_chosen_grb}_{self._closest_dets[j]}_{i-1}_lightcurve_3ml.pdf')

                    if plot_count_spectrum:

                        fluence_plugin.view_count_spectrum()

                        if savefigs:

                            plt.savefig(dir_figs / f'{self._name_chosen_grb}_{self._closest_dets[j]}_{i-1}_count_spectrum_3ml.pdf')
                    
                fit_function = Cutoff_powerlaw_Ep()

                    #Cutoff_powerlaw.

                point_source = PointSource("ps", self.ra, self.dec, spectral_shape=fit_function)

                model = Model(point_source)

                #set priors
                model.ps.spectrum.main.Cutoff_powerlaw_Ep.K.prior = Log_normal(
                    mu=-1.5, sigma=1.5
                )
                model.ps.spectrum.main.Cutoff_powerlaw_Ep.index.prior =Truncated_gaussian(
                    lower_bound=-3, upper_bound=0.5, mu=-0.5, sigma=0.3
                )
                model.ps.spectrum.main.Cutoff_powerlaw_Ep.xp.prior = Truncated_gaussian(
                    lower_bound=10, upper_bound=1e4, mu=1000, sigma=600
                )

                data_list = DataList(*fluence_plugins)

                self._bayes = BayesianAnalysis(model, data_list)

                self._bayes.set_sampler("multinest", share_spectrum=True)

                self._bayes.sampler.setup(n_live_points=n_live_points)

                self._bayes.sample(quiet=True)

                self._sampling_results_file = f"3ml_sampling_results_{i-1}.fits"

                self._sampled = True

                self._bayes_results.append(self._bayes)

                self._bayes.results.write_to(self._dir_sim/ self._sampling_results_file, overwrite=True)
            
            return True
        
        else:
            print(f'Only {len(self._start_times)} found by Bayesian Blocks')
            
            return False    

    def fit_cpl_ep(self,plot_lightcurve=True,plot_count_spectrum=True,savefigs=False, dir_figs=None,n_live_points=1000,p0=.05):
        
        if self._sampled:
            print('It was sampled before.')
        else:
            flag = self._fit_cpl_ep(plot_lightcurve,plot_count_spectrum,savefigs, dir_figs,n_live_points,p0)
        return flag
    
    def plot_median_fit(self, savefig=False, dir_fig=None,**kwargs):

        if self._sampled == False:

            raise Exception('Sample first')

        else:
            figs = []
            for i in range(len(self._bayes_results)):
                self._bayes_results[i].restore_median_fit()

                fig = display_spectrum_model_counts(
                    self._bayes_results[i],
                    data_colors=["C00", "C01", "C02","C03"],
                    model_colors=["C00", "C01", "C02","C03"],
                    show_background=False,
                    source_only=True,
                    step=False,
                    **kwargs)

                ax = fig.get_axes()

                ax[0].set_ylim(1e-4,10)
                h,l = ax[0].get_legend_handles_labels()
                ax[0].legend(h,l,fontsize = 'small',loc='upper right')
                ax[0].set_xlim(None,2.5e4)

                plt.tight_layout()

                figs.append(fig)

                if dir_fig is None:

                    dir_figs = self._dir_sim / '3ml_fits'
                
                else:
                    
                    dir_figs = Path(dir_fig)

                if savefig:

                    plt.savefig(dir_figs / f'{self._name_chosen_grb}_{i}_median_fit_3ml.pdf')

    def corner_plot(self,savefig=False, dir_fig=None, dir_bayes_results=None,renamed_parameters=None,**kwargs):

        if self._sampled == False:

            if dir_bayes_results is None:

                raise Exception("Specify path of Bayesian fit results or execute fit with .fit_cpl_ep()")

            else:

                results_reloaded = load_analysis_results(dir_bayes_results)

        if self._sampled:
            figs = []
            for i in range(len(self._bayes_results)):

                fig,ax = plt.subplots(3,3,figsize=(6,6))

                if self._pulse_profile:
                    t=0.5*(self._stop_times[i]-self._start_times[i])
                else:
                    t=None

                truths = [self.K_exp(t), self.alpha_latent, self.ep_latent(t)/(1+self.z_latent)]

                self._bayes_results[i].results.corner_plot(fig=fig,truths=truths,truth_color='C01',renamed_parameters=renamed_parameters,**kwargs)
                
                ax = fig.get_axes()
                for i in range(9):
                    ax[i].tick_params(axis='both', which='major', pad=0.01)
                    
                plt.tight_layout()

                figs.append(fig)

                if dir_fig is None:

                    dir_figs = self._dir_sim / '3ml_fits'

                else:
                    
                    dir_figs = Path(dir_fig)


                if savefig:

                    plt.savefig(dir_figs / f'{self._name_chosen_grb}_{i}_corner_plot_3ml.pdf')

            return figs
