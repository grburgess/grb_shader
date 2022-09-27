from typing import List, Optional, Union

import numpy as np
import scipy.stats as stats
import popsynth as ps


class TDecaySampler(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "TDecaySampler"
    def __init__(self):
        """
        samples the decay of the pulse
        """

        #call super class's __init__ method
        super(TDecaySampler, self).__init__(name="tdecay", observed=False)

    def true_sampler(self, size):

        t90 = 10 ** self._secondary_samplers["t90"].true_values
        trise = self._secondary_samplers["trise"].true_values

        self._true_values = (
            1.0 / 50.0 * (10 * t90 + trise + np.sqrt(trise)
                          * np.sqrt(20 * t90 + trise))
        )


class DurationSampler(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "DurationSampler"
    def __init__(self):
        "samples how long the pulse lasts"

        super(DurationSampler, self).__init__(name="duration", observed=False)

    def true_sampler(self, size):

        t90 = self._secondary_samplers["t90"].true_values

        # add that other 10 %

        self._true_values = 1.1 * t90

class TriangleT90Sampler_Cor(ps.AuxiliarySampler):
    """
    Assume Ep-Eiso correlation
    Assume that pulse has a triangle shape
    T_90 = 2*E_iso/L
    (Case a of Ghirlanda et al., 2016)
    """

    _auxiliary_sampler_name = "TriangleDurationSampler"
    def __init__(self):
        "samples how long the pulse last"

        super(TriangleT90Sampler_Cor, self).__init__(
            name="t90", 
            observed=False,
            uses_luminosity=True
            )

    def true_sampler(self, size):

        eiso = self._secondary_samplers["Eiso"].obs_values

        duration = 2*(eiso)/(self._luminosity)

        self._true_values = duration

class EisoSampler(ps.AuxiliarySampler):
    
    _auxiliary_sampler_name = "EisoSampler"

    #Best fit values for Ep-Eiso correlation 
    #Default: Ghirlanda et al., 2016, case a values
    q_a = ps.auxiliary_sampler.AuxiliaryParameter(default=0.033)
    m_a = ps.auxiliary_sampler.AuxiliaryParameter(default=0.91,vmin=0)
    #tau of lognormal distribution from which observed value for E_iso is computed
    tau = ps.auxiliary_sampler.AuxiliaryParameter(default=0.2,vmin=0)

    def __init__(self):
        """Sample E_iso"""

        super(EisoSampler, self).__init__(name="Eiso", observed=True)

    def true_sampler(self, size):

        ep = self._secondary_samplers["ep"].true_values #keV

        #from Ep-Eiso correlation
        eiso = np.power(10,(1./self.m_a * ( np.log10(ep/670.) - self.q_a))) * 1e51 #erg

        self._true_values = eiso

    def observation_sampler(self, size):
        #lognormal distribution whose central value is given by the true value
        self._obs_values = np.exp(stats.norm.rvs(loc=self._true_values/1.e52, scale=self.tau, size=size)
        )*1.e52
