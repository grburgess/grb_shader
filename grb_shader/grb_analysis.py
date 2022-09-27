from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincc
from astropy.coordinates import SkyCoord

import popsynth as ps
from cosmogrb.universe.survey import Survey
from cosmogrb.utils.package_utils import get_path_of_data_file
from gbmgeometry import GBM, PositionInterpolator
from threeML import *
from threeML.utils.OGIP.response import OGIPResponse
from astromodels.functions import Cutoff_powerlaw_Ep

class GRBAnalysis_constant(object):

    def __init__(
        self,
        dir_sim,
        name_chosen_grb,
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

        self._compute_K_exp()

        self._find_closest_dets()

        self._tmin = -0.5 #T0=0 - no offset simulated
        #choose time interval for fitting from maximum triggered_time_scales
        #tmax = max(survey_no_selec_pulse[chosen_grb].detector_info.extra_info['triggered_time_scales'])
        self._tmax = self.duration_latent + 0.5

        self._sampled = False
    
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
    def ep_latent(self):
        return self.chosen_grb.source_params['ep']
    
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
    def duration_latent(self):
        return self.chosen_grb.duration

    def _compute_K_exp(self):
        # normalization factor of astromodels Ep_cutoff_powerlaw
                
        a = 10 * (1 + self.z_latent)
        b = 1e4 * (1 + self.z_latent)

        #cutoff energy
        ec = self.ep_latent / (2 + self.alpha_latent)

        #for conversion of popsynth flux to keV/cm^2/s
        erg2keV = 6.24151e8

        #integral of E*f_cpl
        i1 = gammaincc(2.0 + self.alpha_latent, a / ec) * gamma(2.0 + self.alpha_latent)
        i2 = gammaincc(2.0 + self.alpha_latent, b / ec) * gamma(2.0 + self.alpha_latent)

        intflux = -ec * ec * (i2 - i1)
        norm = self.peak_flux_latent * erg2keV / (intflux)

        #from comparison of cpl in cosmogrb and ep_cpl definition in astromodels
        factor = ((2+self.alpha_latent)*(1+self.z_latent)/self.ep_latent)**self.alpha_latent

        self._K_exp = norm*factor

    @property
    def K_exp(self):
        return self._K_exp

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

    def plot_photon_lc(self,savefig=False,dir_figs=None,axes=None):

        if axes is None:

            fig, axes = plt.subplots(5,3,sharex=True,sharey=True,figsize=(7,7))

        else:

            fig = axes.get_figure()

        print(f'Chosen time interval: {self._tmin} - {self._tmax} s')

        row=0
        col = 0

        for k,v  in self.chosen_grb.items():
            ax = axes[row][col]

            lightcurve =v['lightcurve']
            lightcurve.display_lightcurve(dt=.1, tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color='#25C68C',label='Source+Background')
            #lightcurve.display_source(dt=.5,ax=ax,lw=1,color="#A363DE",label='Source')
            lightcurve.display_background(dt=.1,tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1, color="#2C342E",label='Background')
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

        plt.tight_layout()

        if savefig:

            if dir_figs is None:

                dir_figs = self._dir_sim / '3ml_fits'
            
            plt.savefig(str(dir_figs / f'{self._name_chosen_grb}_photon_lightcurve.png'),dpi=300)

        return fig

    def plot_photon_spectrum(self,savefig=False,dir_figs=None,axes=None):

        if axes is None:

            fig, axes = plt.subplots(5,3,sharex=True,sharey=True,figsize=(7,7))

        else:

            fig = axes.get_figure()

        print(f'Chosen time interval: {self._tmin} - {self._tmax} s')

        row=0
        col = 0

        lightcurves = []

        for k,v  in self.chosen_grb.items():
            ax = axes[row][col]
            
            lightcurve = v['lightcurve']
            
            lightcurve.display_count_spectrum(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color='#25C68C')
            lightcurve.display_count_spectrum_source(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1,color="#A363DE",label='Source')
            lightcurve.display_count_spectrum_background(tmin=self._tmin,tmax=self._tmax,ax=ax,lw=1, color="#2C342E",label='Background')
            #ax.set_xlim(10, 30)
            ax.set_xlim(8,40000)

            pha = lightcurve._bin_spectrum(tmin=self._tmin, tmax=self._tmax, times=lightcurve.times, pha=lightcurve.pha)
            ax.set_ylim(bottom=1e-3,top=max(pha)*3)

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

        axes[4,2].set_visible(False)  
        #axes[0][0].legend()
        plt.tight_layout()

        if savefig:

            if dir_figs is None:

                dir_figs = self._dir_sim / '3ml_fits'
            
            plt.savefig(str(dir_figs / f'{self._name_chosen_grb}_photon_spectrum.png'),dpi=300)

        return fig

    def _fit_cpl_ep(self, plot_lightcurve, plot_count_spectrum, savefigs, dir_figs, n_live_points):

        """Fit Cutoff powerlaw with Ep parametrization (as defined in astromodels)
            using Multinest 

        :param plot_lightcurve: Choose if total light curve should be plotted 
            including time selections and polynomial background fit
        :type plot_lightcurve: bool
        :param plot_count_spectrum: Choose if count spectrum should be plotted 
        :type plot_count_spectrum: bool
        :param savefigs: choose if figures should be saved
        :type savefigs: bool
        :param dir_figs: path of saved figures
        :type dir_figs: str
        :param n_live_points: Multinest number of live_points
        :type n_live_points: int
        """

        fluence_plugins = []

        for i in range(len(self._closest_dets)):

            response = OGIPResponse(self._dir_grbs / f'rsp_{self._name_chosen_grb}_{self._closest_dets[i]}.rsp')

            tte =  TimeSeriesBuilder.from_gbm_tte(
                name=f'{self._closest_dets[i]}_tte',
                tte_file=self._dir_grbs / f'tte_{self._name_chosen_grb}_{self._closest_dets[i]}.fits',
                rsp_file=response,
                poly_order = 0
                )

            #use bayesian block method to find time bins
            tte.create_time_bins(start=-10.,stop=40,method='bayesblocks',p0=.001, use_background=False)

            # for constant light curve, exactly three time bins should be found 
            # select first and last interval for polynomial background fit
            # second time bin corresponds to source 
            if len(tte.bins) == 3:
                tte.set_active_time_interval(f'{tte.bins.start_times[1]}-{tte.bins.stop_times[1]}')
                tte.set_background_interval(f'{tte.bins.start_times[0]}-{tte.bins.stop_times[0]}',f'{tte.bins.start_times[2]}-{tte.bins.stop_times[2]}')

            fluence_plugin = tte.to_spectrumlike()

            if self._closest_dets[i][0] == 'n':

                fluence_plugin.set_active_measurements("9-900")

            elif self._closest_dets[i][0] == 'b':

                fluence_plugin.set_active_measurements("250-30000")
            
            fluence_plugin.rebin_on_background(1)
            
            fluence_plugins.append(fluence_plugin)

            if dir_figs is None:

                dir_figs = self._dir_sim / '3ml_fits'

            if plot_lightcurve:

                tte.view_lightcurve()

                if savefigs:

                    plt.savefig(dir_figs / f'{self._name_chosen_grb}_{self._closest_dets[i]}_lightcurve_3ml.png',dpi=300)

            if plot_count_spectrum:

                fluence_plugin.view_count_spectrum()

                if savefigs:

                    plt.savefig(dir_figs / f'{self._name_chosen_grb}_{self._closest_dets[i]}_count_spectrum_3ml.png',dpi=300)
            
        fit_function = Cutoff_powerlaw_Ep()

            #Cutoff_powerlaw.

        point_source = PointSource("ps", self.ra, self.dec, spectral_shape=fit_function)

        model = Model(point_source)

        #set priors
        model.ps.spectrum.main.Cutoff_powerlaw_Ep.K.prior = Log_normal(
            mu=-4, sigma=1
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

        self._bayes.sampler.setup(n_live_points=1000)

        self._bayes.sample()

        self._sampling_results_file = "3ml_sampling_results.fits"

        self._sampled = True

        self._bayes.results.write_to(self._dir_sim/ self._sampling_results_file, overwrite=True)

    def fit_cpl_ep(self,plot_lightcurve=True,plot_count_spectrum=True,savefigs=False, dir_figs=None,n_live_points=1000):
        if self._sampled:
            print('It was sampled before.')
        else:
            self._fit_cpl_ep(plot_lightcurve,plot_count_spectrum,savefigs, dir_figs,n_live_points)
    
    def plot_median_fit(self, savefig=False, dir_fig=None):

        if self._sampled == False:

            raise Exception('Sample first')

        else:
            self._bayes.restore_median_fit()

            fig = display_spectrum_model_counts(self._bayes,min_rate=5,step=False)

            ax = fig.get_axes()

            ax[0].set_ylim(1e-4,10)

            plt.tight_layout()

            if dir_fig is None:

                    dir_fig = self._dir_sim / '3ml_fits'

            if savefig:

                plt.savefig(dir_fig / f'{self._name_chosen_grb}_median_fit_3ml.png',dpi=300)

    def corner_plot(self,savefig=False, dir_fig=None, dir_bayes_results=None):

        if self._sampled == False:

            if dir_bayes_results is None:

                raise Exception("Specify path of Bayesian fit results or execute fit with .fit_cpl_ep()")

            else:

                results_reloaded = load_analysis_results(dir_bayes_results)

        fig,ax = plt.subplots(3,3,figsize=(6,6))

        truths = [self.K_exp, self.alpha_latent, self.ep_latent/(1+self.z_latent)]

        if self._sampled:

            self._bayes.results.corner_plot(fig=fig,truths=truths,truth_color='C01')

        else:

            results_reloaded.corner_plot(fig=fig,truths=truths,truth_color='C01')

        plt.tight_layout()

        if dir_fig is None:

                dir_fig = self._dir_sim / '3ml_fits'

        if savefig:

            plt.savefig(dir_fig / f'{self._name_chosen_grb}_corner_plot_3ml.png',dpi=300)

        return fig