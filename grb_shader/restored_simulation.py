import collections
from pathlib import Path
from typing import List
from xml.sax.handler import property_declaration_handler
import matplotlib.pyplot as plt
import numpy as np
import yaml

from astromodels import Truncated_gaussian
from popsynth import Population
from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.distributions.bpl_distribution import BPLDistribution
from cosmogrb.universe.survey import Survey

from tqdm.auto import tqdm

from .catalog import LocalVolume
from .utils.logging import setup_log
from .plotting.plotting_functions import logbins_scaled_histogram,log10normal,logbins_norm_histogram

logger = setup_log(__name__)


class RestoredUniverse(object):

    def __init__(
        self,
        path_survey_file: str,
        survey_base_file_name: str = 'survey_1234',
        pop_base_file_name: str = 'pop_1234',
        path_pops_files: str = None,
    ):
        self._survey_file = Path(path_survey_file)/f"{survey_base_file_name}.h5"
        self._survey_path = self._survey_file.parent

        if self._survey_file.exists():
            logger.info('loading survey')

            self._survey = Survey.from_file(self._survey_file)
        else:
            self._survey = None
            logger.info('no survey file found')

        if path_pops_files is None:
            self._pops_path = self._survey_path
        else:
            self._pops_path = path_pops_files

        self._population_file = self._pops_path/f"{pop_base_file_name}.h5"

        if self._population_file.exists() == False:
            logger.error(f'population file does not exist in {self._pops_path}, check path')

        logger.info('loading population')

        self._pop = Population.from_file(self._population_file)
        self._n_sim_grbs = self._pop.n_objects
        self._n_detected_grbs = self._pop.n_detections


    @property
    def n_sim_grbs(self):
        # total number of GRBs in the population (from integrated spatial distribution)
        return self._n_sim_grbs

    @property
    def fractions_det(self):
        #fraction of detected GRBs 
        return self.n_detected_grbs/self.n_sim_grbs

    @property
    def pop(self):
        return self._pop

    @property
    def survey(self):
        return self._survey
    
    @property
    def mask_det(self):
        if self._survey is None:
            return None
        else:
            return self.survey.mask_detected_grbs

    @property
    def pop_file(self):
        return self._population_file

    def hist_redshift(
        self,
        n_bins,
        ax=None,
        r0=36,
        a=1,
        rise=2.0,
        decay=2.0,
        peak=2.8,
        normalized_hist=False,
        plot_sfr_pdf=True,
        det=True):

        if ax is None:
            fig, ax = plt.subplots()

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        counts, bins = np.histogram(self.pop.distances,bins=n_bins)
        int_hist= sum(counts * np.diff(bins))

        x = np.linspace(min(self.pop.distances),max(self.pop.distances),1000)

        if plot_sfr_pdf:
            sfr = SFRDistribution()
            sfr.r0=r0
            sfr.a=a
            sfr.rise=rise
            sfr.decay=decay
            sfr.peak=peak

            f =sfr.dNdV(x)* sfr.differential_volume(x) / sfr.time_adjustment(x)

            if normalized_hist:

                f_norm = f /(np.trapz(f, x))
                ax.plot(x,f_norm,label='Normalized pdf',color='C00')
            else:
                f_norm = f*int_hist/(np.trapz(f, x))
                ax.plot(x,f_norm,label='Unnormalized pdf',color='C00')

        if normalized_hist:
            ax.set_ylabel(r'Normalized n$_\mathrm{GRBs}$')
            ax.hist(self.pop.distances,bins=n_bins,alpha=0.7,label='All',density=True)
            if det:
                ax.hist(self.pop.distances[self.mask_det],bins=n_bins,alpha=0.7,label='Detected',density=True)
        else:
            ax.set_ylabel(r' n$_\mathrm{GRBs}$')
            ax.hist(self.pop.distances,bins=n_bins,alpha=0.7,label='All')
            if det:
                ax.hist(self.pop.distances[self.mask_det],bins=n_bins,alpha=0.7,label='Detected')

        ax.legend(loc='upper right')
        ax.set_xlabel(r'z')

    def hist_ep(
        self,
        n_bins,
        ax=None,
        normalized_hist=False,
        plot_bpl_distr=True,
        det=True,
        Lmin=0.1,
        alpha=0.55,
        Lbreak= 2100,
        beta=-2.5,
        Lmax=1.e+5,
        obs=False
        ):

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        if ax is None:
            fig, ax = plt.subplots()

        if obs:
            #transform to observer frame
            ep = self.pop.ep/(1+self.pop.distances)
            ep_det = self.pop.ep[self.mask_det]/(1+self.pop.distances[self.mask_det])
        else:
            #in burst frame
            ep = self.pop.ep
            ep_det = self.pop.ep[self.mask_det]
        

        bins2 = np.geomspace(min(ep),max(ep),n_bins)
        counts, bins = np.histogram(ep,bins=bins2)
        int_hist= np.sum(counts)

        if normalized_hist:
            ax.set_ylabel(r'Normalized n$_\mathrm{GRBs}$')
            intervals= logbins_norm_histogram(ep,n_bins=n_bins,ax=ax)
        else:
            ax.set_ylabel(r'Scaled n$_\mathrm{GRBs}$')
            intervals=logbins_scaled_histogram(ep,n_bins=n_bins,ax=ax)

        if det:
            if normalized_hist:
                logbins_norm_histogram(ep_det,ax=ax,intervals=intervals)
            else:
                logbins_scaled_histogram(ep_det,ax=ax,intervals=intervals)

        if plot_bpl_distr:
            bpl = BPLDistribution()
            bpl.Lmin = Lmin
            bpl.alpha = alpha
            bpl.Lbreak = Lbreak
            bpl.beta = beta
            bpl.Lmax = Lmax
            if normalized_hist:
                ax.plot(intervals,bpl.phi(intervals),color='C00')
            else:
                ax.plot(intervals,bpl.phi(intervals)*int_hist,color='C00')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'E$_\mathrm{p,burst}$ [keV]')

    def hist_luminosity(
        self,
        n_bins,
        ax=None,
        normalized_hist=False,
        plot_bpl_distr=True,
        det=True,
        Lmin=1.e+50,
        alpha=0.32,
        Lbreak= 0.79e+52,
        beta=-1.8,
        Lmax=1.e+55
        ):

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        if ax is None:
            fig, ax = plt.subplots()

        intervals = np.geomspace(min(self.pop.luminosities),max(self.pop.luminosities),n_bins)
        counts, bins = np.histogram(self.pop.luminosities,bins=intervals)
        int_hist= np.sum(counts)

        if normalized_hist:
            ax.set_ylabel(r'Normalized n$_\mathrm{GRBs}$')
            intervals= logbins_norm_histogram(self.pop.luminosities,n_bins=n_bins,ax=ax)
        else:
            ax.set_ylabel(r'Scaled n$_\mathrm{GRBs}$')
            intervals=logbins_scaled_histogram(self.pop.luminosities,n_bins=n_bins,ax=ax)

        if det:
            if normalized_hist:
                logbins_norm_histogram(self.pop.luminosities[self.mask_det],ax=ax,intervals=intervals)
            else:
                logbins_scaled_histogram(self.pop.luminosities[self.mask_det],ax=ax,intervals=intervals)

        if plot_bpl_distr:
            bpl = BPLDistribution()
            bpl.Lmin = Lmin
            bpl.alpha = alpha
            bpl.Lbreak = Lbreak
            bpl.beta = beta
            bpl.Lmax = Lmax
            if normalized_hist:
                ax.plot(intervals,bpl.phi(intervals),color='C00')
            else:
                ax.plot(intervals,bpl.phi(intervals)*int_hist,color='C00')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'L [erg/s]')

    def hist_t90(
        self,
        n_bins=50,
        ax=None,
        normalized_hist=False,
        det=True,
        plot_logn=True,
        mu = -0.196573,
        sigma=0.541693
    ):

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        if ax is None:
            fig, ax = plt.subplots()

        bins2 = np.geomspace(min(self.pop.duration),max(self.pop.duration),n_bins)
        counts, bins = np.histogram(self.pop.duration,bins=bins2)
        int_hist= np.sum(counts)

        if normalized_hist:
            ax.set_ylabel(r'Normalized n$_\mathrm{GRBs}$')
            intervals= logbins_norm_histogram(self.pop.duration,n_bins=n_bins,ax=ax)
        else:
            ax.set_ylabel(r'Scaled n$_\mathrm{GRBs}$')
            intervals=logbins_scaled_histogram(self.pop.duration,n_bins=n_bins,ax=ax)

        if det:
            if normalized_hist:
                logbins_norm_histogram(self.pop.duration[self.mask_det],ax=ax,intervals=intervals)
            else:
                logbins_scaled_histogram(self.pop.duration[self.mask_det],ax=ax,intervals=intervals)
        
        if plot_logn:
            if normalized_hist:
                ax.plot(bins2,0.9*log10normal(bins2,mu,sigma),color='C00')
            else:
                ax.plot(bins2,0.9*int_hist*log10normal(bins2,mu,sigma),color='C00')

        ax.set_xlabel(r'$T_{90}$ [s]')
        ax.set_xscale('log')
        ax.set_xlim(5e-3,1e1)

    def hist_alpha(
        self,
        n_bins,
        ax=None,
        normalized_hist=False,
        det=True,
        plot_truncnormal=True,
        mu=-0.6,
        sigma=0.2,
        lower_bound=-1.5,
        upper_bound=0
        ):

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        if ax is None:
            fig, ax = plt.subplots()

        counts, bins = np.histogram(self.pop.alpha,bins=n_bins)
        int_hist= np.trapz(counts,(bins[1:]+bins[:-1])/2)
        
        if plot_truncnormal:
            x = np.linspace(min(self.pop.alpha),max(self.pop.alpha),n_bins)

            trunc_gauss = Truncated_gaussian()
            trunc_gauss.mu=mu
            trunc_gauss.sigma=sigma
            trunc_gauss.lower_bound=lower_bound
            trunc_gauss.upper_bound=upper_bound

            if normalized_hist:
                ax.plot(x,trunc_gauss.evaluate_at(x),color='C00')
            else:
                ax.plot(x,trunc_gauss.evaluate_at(x)*int_hist,color='C00')

        if normalized_hist:
            ax.set_ylabel(r'Normalized n$_\mathrm{GRBs}$')
            ax.hist(self.pop.alpha,bins=n_bins,alpha=0.7,label='all',density=True)
        else:
            ax.set_ylabel(r'n$_\mathrm{GRBs}$')
            ax.hist(self.pop.alpha,bins=n_bins,alpha=0.7,label='all')

        if det:
            if normalized_hist:
                ax.hist(self.pop.alpha[self.mask_det],bins=n_bins,alpha=0.7,label='observed',density=True)
            else:
                ax.hist(self.pop.alpha[self.mask_det],bins=n_bins,alpha=0.7,label='observed')

        ax.set_xlabel(r'$\alpha$')
        ax.set_xlim(-1.25,0)

    def plot_flux_redshift_diagram(self,ax=None,det=True):
        if ax is None:
            fig, ax = plt.subplots()

        if self.survey is None:
            logger.info('no survey defined')
            det = False

        ax.plot(self.pop.distances,self.pop.fluxes.latent,'.',alpha=0.3)
        if det == True:
            ax.plot(self.pop.distances[self.mask_det],self.pop.fluxes.latent[self.mask_det],'.',alpha=0.3)
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'F [erg/cm$^2$/s]')
        ax.set_yscale('log')

    def plot_flux_t90_diagram(self,ax=None,det=True):
        if self.survey is None:
            logger.info('no survey defined')
            det = False

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.pop.duration,self.pop.fluxes.latent,'.',alpha=0.3)
        if det == True:
            ax.plot(self.pop.duration[self.mask_det],self.pop.fluxes.latent[self.mask_det],'.',alpha=0.3)
        ax.set_xlabel(r'$T_{90}$ [s]')
        ax.set_ylabel(r'F [erg/cm$^2$/s]')
        ax.set_yscale('log')
        ax.set_xscale('log')

    def hist_parameters(
        self,
        n_bins=50,
        ax=None,
        normalized_hist=False,
        det=True,
        plot_pdf=True,
        sfr_r0=36,
        sfr_a=1,
        sfr_rise=2.0,
        sfr_decay=2.0,
        sfr_peak=2.8,
        ep_bpl_min=0.1,
        ep_bpl_alpha=0.55,
        ep_bpl_break= 2100,
        ep_bpl_beta=-2.5,
        ep_bpl_max=1.e+5,
        lum_min=1.e+50,
        lum_alpha=0.32,
        lum_break= 0.79e+52,
        lum_beta=-1.8,
        lum_max=1.e+55,
        alpha_mu=-0.6,
        alpha_sigma=0.2,
        alpha_lower_bound=-1.5,
        alpha_upper_bound=0,
        t90_mu = -0.196573,
        t90_sigma=0.541693
        ):
        
        if ax is None:
            fig, ax = plt.subplots(4,2,figsize=(8,8))

        self.hist_redshift(
            ax=ax[0][0],
            n_bins=n_bins,
            r0=sfr_r0,
            a=sfr_a,
            rise=sfr_rise,
            decay=sfr_decay,
            peak=sfr_peak,
            normalized_hist=normalized_hist,
            plot_sfr_pdf=plot_pdf,
            det=det
            )

        self.hist_luminosity(
            n_bins,
            ax=ax[0][1],
            normalized_hist=normalized_hist,
            plot_bpl_distr=plot_pdf,
            det=det,
            Lmin=lum_min,
            alpha=lum_alpha,
            Lbreak=lum_break,
            beta=lum_beta,
            Lmax=lum_max
        )

        self.hist_ep(
            n_bins=n_bins,
            ax=ax[1][0],
            normalized_hist=normalized_hist,
            plot_bpl_distr=plot_pdf,
            det=det,
            Lmin=ep_bpl_min,
            alpha=ep_bpl_alpha,
            Lbreak= ep_bpl_break,
            beta=ep_bpl_beta,
            Lmax=ep_bpl_max,
            obs=False
        )

        self.hist_ep(
            n_bins=n_bins,
            ax=ax[1][1],
            normalized_hist=normalized_hist,
            plot_bpl_distr=False,
            det=det,
            Lmin=ep_bpl_min,
            alpha=ep_bpl_alpha,
            Lbreak= ep_bpl_break,
            beta=ep_bpl_beta,
            Lmax=ep_bpl_max,
            obs=True
        )

        self.hist_alpha(
            n_bins,
            ax=ax[2][0],
            normalized_hist=normalized_hist,
            det=det,
            plot_truncnormal=plot_pdf,
            mu=alpha_mu,
            sigma=alpha_sigma,
            lower_bound=alpha_lower_bound,
            upper_bound=alpha_upper_bound
        )

        self.hist_t90(
            n_bins=n_bins,
            ax=ax[2][1],
            normalized_hist=normalized_hist,
            det=det,
            plot_logn=plot_pdf,
            mu = t90_mu,
            sigma=t90_sigma
        )

        self.plot_flux_redshift_diagram(ax=ax[3][0])

        self.plot_flux_t90_diagram(ax=ax[3][1])
