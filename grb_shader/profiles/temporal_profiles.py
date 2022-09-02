#ToDo: Move all TemporalProfile Samplers here
from .base_profile import BaseProfile
import popsynth as ps
from grb_shader.samplers import DurationSampler, TDecaySampler, TriangleT90Sampler_Cor, EisoSampler

from ..utils.logging import setup_log

logger = setup_log(__name__)

class PulseProfile_Lognormal(BaseProfile):

    def _construct(
        self, 
        t90_mu, 
        t90_tau,
        t_rise_mu, 
        t_rise_tau, 
        t_rise_lower, 
        t_rise_upper,
        tau_mu,
        tau_lower,
        tau_upper,
        tau_tau
        ):
        """
        Samplers for pulse profile parameters (Norris, 2005)

        Sample T_90 from LogNormal(``t90_mu``, ``t90_tau``)
        Sample rise time t_rise from truncated Normal()

        :param t90_mu: mean of t90 
        :type t90_mu: float
        :param t90_tau: tau of t90
        :type t90_tau: float
        :param t_rise_mu: mean of t_rise
        :type t_rise_mu: float
        :param t_rise_tau: tau of t90 
        :type t_rise_tau: float
        :param t_rise_lower: lower limit for t_rise
        :type t_rise_lower: float
        :param t_rise_upper: upper limit for t_rise
        :type t_rise_upper: float
        """

        logger.debug('Use PulseProfile_Lognormal')

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False)

        trise.lower = t_rise_lower
        trise.upper = t_rise_upper
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        #TODO: include other possible t90 samplings (as from corr. or log10Normal)
        t90 = ps.aux_samplers.LogNormalAuxSampler(name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        #for time evolution of Ep: Ep(t)= Ep/(1+t/tau)
        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = tau_lower
        tau.upper = tau_upper
        tau.mu = tau_mu
        tau.tau = tau_tau

        self._quantities = [duration, tdecay, tau]

class PulseProfile_Lognormal_Trunc(BaseProfile):

    def _construct(
        self, 
        t90_mu, 
        t90_tau,
        t90_lower,
        t90_upper,
        t_rise_mu, 
        t_rise_tau, 
        t_rise_lower, 
        t_rise_upper,
        tau_mu,
        tau_lower,
        tau_upper,
        tau_tau
        ):
        """
        Samplers for pulse profile parameters (Norris, 2005)

        Sample T_90 from truncated LogNormal(``t90_mu``, ``t90_tau``)
        Sample rise time t_rise from truncated Normal()

        :param t90_mu: mean of t90 
        :type t90_mu: float
        :param t90_tau: tau of t90
        :type t90_tau: float
        :param t90_lower: lower limit for t90
        :type t90_lower: float
        :param t90_upper: upper limit for t90
        :type t90_upper: float
        :param t_rise_mu: mean of t_rise
        :type t_rise_mu: float
        :param t_rise_tau: tau of t90 
        :type t_rise_tau: float
        :param t_rise_lower: lower limit for t_rise
        :type t_rise_lower: float
        :param t_rise_upper: upper limit for t_rise
        :type t_rise_upper: float
        """

        logger.debug('Use PulseProfile_Lognormal')

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False)

        trise.lower = t_rise_lower
        trise.upper = t_rise_upper
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        t90 = ps.aux_samplers.TruncatedLogNormalAuxSampler(name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau
        t90.lower = t90_lower
        t90.upper = t90_upper

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        #for time evolution of Ep: Ep(t)= Ep/(1+t/tau)
        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = tau_lower
        tau.upper = tau_upper
        tau.mu = tau_mu
        tau.tau = tau_tau

        self._quantities = [duration, tdecay, tau]

class PulseProfile_Log10normal(BaseProfile):

    def _construct(
        self, 
        t90_mu, 
        t90_tau,
        t_rise_mu, 
        t_rise_tau, 
        t_rise_lower, 
        t_rise_upper,
        tau_mu,
        tau_lower,
        tau_upper,
        tau_tau
        ):
        """
        Samplers for pulse profile parameters (Norris, 2005)

        Sample T_90 from LogNormal(``t90_mu``, ``t90_tau``)
        Sample rise time t_rise from truncated Normal()

        :param t90_mu: mean of t90 
        :type t90_mu: float
        :param t90_tau: tau of t90
        :type t90_tau: float
        :param t_rise_mu: mean of t_rise
        :type t_rise_mu: float
        :param t_rise_tau: tau of t90 
        :type t_rise_tau: float
        :param t_rise_lower: lower limit for t_rise
        :type t_rise_lower: float
        :param t_rise_upper: upper limit for t_rise
        :type t_rise_upper: float
        """

        logger.debug('Use PulseProfile_Log10normal')

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False)

        trise.lower = t_rise_lower
        trise.upper = t_rise_upper
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        #TODO: include other possible t90 samplings (as from corr. or log10Normal)
        t90 = ps.aux_samplers.Log10NormalAuxSampler(name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        #for time evolution of Ep: Ep(t)= Ep/(1+t/tau)
        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = tau_lower
        tau.upper = tau_upper
        tau.mu = tau_mu
        tau.tau = tau_tau

        self._quantities = [duration, tdecay, tau]

class PulseProfile_Log10normal_Trunc(BaseProfile):

    def _construct(
        self, 
        t90_mu, 
        t90_tau,
        t90_lower,
        t90_upper,
        t_rise_mu, 
        t_rise_tau, 
        t_rise_lower, 
        t_rise_upper,
        tau_mu,
        tau_lower,
        tau_upper,
        tau_tau
        ):
        """
        Samplers for pulse profile parameters (Norris, 2005)

        Sample T_90 from truncated Log10Normal(``t90_mu``, ``t90_tau``)
        Sample rise time t_rise from truncated Normal()

        :param t90_mu: mean of t90 
        :type t90_mu: float
        :param t90_tau: tau of t90
        :type t90_tau: float
        :param t90_lower: lower limit for t90
        :type t90_lower: float
        :param t90_upper: upper limit for t90
        :type t90_upper: float
        :param t_rise_mu: mean of t_rise
        :type t_rise_mu: float
        :param t_rise_tau: tau of t90 
        :type t_rise_tau: float
        :param t_rise_lower: lower limit for t_rise
        :type t_rise_lower: float
        :param t_rise_upper: upper limit for t_rise
        :type t_rise_upper: float
        """

        logger.debug('Use PulseProfile_Lognormal')

        trise = ps.aux_samplers.TruncatedNormalAuxSampler(name="trise", observed=False)

        trise.lower = t_rise_lower
        trise.upper = t_rise_upper
        trise.mu = t_rise_mu
        trise.tau = t_rise_tau

        t90 = ps.aux_samplers.TruncatedLog10NormalAuxSampler(name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau
        t90.lower = t90_lower
        t90.upper = t90_upper

        tdecay = TDecaySampler()
        duration = DurationSampler()
        tdecay.set_secondary_sampler(t90)
        tdecay.set_secondary_sampler(trise)

        duration.set_secondary_sampler(t90)

        #for time evolution of Ep: Ep(t)= Ep/(1+t/tau)
        tau = ps.aux_samplers.TruncatedNormalAuxSampler(
            name="tau", observed=False)

        tau.lower = tau_lower
        tau.upper = tau_upper
        tau.mu = tau_mu
        tau.tau = tau_tau

        self._quantities = [duration, tdecay, tau]

class ConstantProfile_Lognormal(BaseProfile):

    def _construct(self, t90_mu, t90_tau):
        """
        Samplers for constant temporal profile parameters
        t90 is sampled from full Log Normal distribution
        """
        logger.debug('Use ConstantProfile_Lognormal')

        t90 = ps.aux_samplers.LogNormalAuxSampler(
            name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class ConstantProfile_Lognormal_Trunc(BaseProfile):

    def _construct(self, t90_mu, t90_tau, t90_lower, t90_upper):
        """
        Samplers for constant temporal profile parameters
        t90 is sampled from truncated Log Normal distribution
        """
        logger.debug('Use ConstantProfile_Lognormal')

        t90 = ps.aux_samplers.TruncatedLogNormalAuxSampler(
            name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau
        t90.lower = t90_lower
        t90.upper = t90_upper

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class ConstantProfile_Log10normal(BaseProfile):

    def _construct(self, t90_mu, t90_tau):
        """
        Samplers for constant temporal profile parameters
        t90 is sampled from full Log10 Normal distribution
        """
        logger.debug('Use ConstantProfile_Log10normal')

        t90 = ps.aux_samplers.Log10NormalAuxSampler(
            name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class ConstantProfile_Log10normal_Trunc(BaseProfile):

    def _construct(self, t90_mu, t90_tau, t90_lower, t90_upper):
        """
        Samplers for constant temporal profile parameters
        t90 is sampled from truncated Log10 Normal distribution
        """
        logger.debug('Use ConstantProfile_Lognormal')

        t90 = ps.aux_samplers.TruncatedLog10NormalAuxSampler(
            name="t90", observed=False)

        t90.mu = t90_mu
        t90.tau = t90_tau
        t90.lower = t90_lower
        t90.upper = t90_upper

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]

class TriangleProfile_Cor(BaseProfile):
    """
    Assume Ep-Eiso correlation and triangle shape of light curve
    """

    def _construct(self, ep_profile, q_a, m_a, tau):
        logger.debug('Use TriangleProfile_Cor')

        eiso = EisoSampler()
        eiso.set_secondary_sampler(ep_profile)

        eiso.q_a = q_a
        eiso.m_a = m_a
        eiso.tau = tau

        t90 = TriangleT90Sampler_Cor()

        t90.set_secondary_sampler(eiso)

        duration = DurationSampler()

        duration.set_secondary_sampler(t90)

        self._quantities = [duration]