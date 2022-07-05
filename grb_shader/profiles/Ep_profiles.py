from .base_profile import BaseProfile
import popsynth as ps

class Log10normalEp(BaseProfile):

    def _construct(self, mu, tau):
        """
        Sample Ep from Log10Normal(mu, tau)
        """
        ep = ps.aux_samplers.Log10NormalAuxSampler(
            name="ep", observed=False)

        ep.mu = mu
        ep.tau = tau

        self._quantities = [ep]

class LognormalEp(BaseProfile):

    def _construct(self, mu, tau):
        """
        Sample Ep from LogNormal(mu, tau)
        """
        ep = ps.aux_samplers.LogNormalAuxSampler(
            name="ep", observed=False)

        ep.mu = mu
        ep.tau = tau

        self._quantities = [ep]

class BplEp(BaseProfile):

    def _construct(self, Epmin, alpha, Epbreak, beta, Epmax):
        """
        Sample Ep from Broken power law distribution
        """
        ep = ps.aux_samplers.BrokenPowerLawAuxSampler(
            name="ep", observed=False)

        ep.xmin = Epmin
        ep.alpha = alpha
        ep.xbreak = Epbreak
        ep.beta = beta
        ep.xmax = Epmax

        self._quantities = [ep]