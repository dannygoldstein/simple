import abc
import numpy as np
from scipy.integrate import quad

__whatami__ = 'Mass profiles for simple supernova atmospheres.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

class MassProfile(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, v):
        """The fraction of the atmosphere's total mass that is enclosed by the
        spherical shell in velocity space of radius `v`. Convert the
        output of this function to a cumulative interior mass by
        multipling by the mass of the ejecta.

        """
        pass

    def accuracy_criterion(self, v, frac):
        """Returns true if truncating the profile at `v` captures at least
        `frac` of the mass."""
        return self(v) >= frac


class Exponential(MassProfile):
    """An exponential density profile."""

    def __init__(self, ke):
        self.ke = ke  # erg
        self.ve = 2455 * (self.ke / 1e51)**0.5  # km / s

    def __call__(self, v):
        return 0.5 * (2.0 - np.exp(-v / self.ve) *
                      (2.0 + (v / self.ve) *
                       (2.0 + v / self.ve)))


class BrokenPowerLaw(MassProfile):
    """A broken power law mass profile with a shallow inner region and a
    steep outer region."""

    def __init__(self, alpha, beta, vt):
        if alpha <= -3:
            raise ValueError("alpha must be greater than -3.")
        if beta >= -3:
            raise ValueError("beta must be less than -3.")
        self.alpha = alpha
        self.beta = beta
        self.vt = vt

    def __call__(self, v):
        v = np.asarray(v)
        vs = v.shape
        v = np.atleast_1d(v)
        res = np.zeros_like(v)

        # constants 
        a3 = self.alpha + 3
        b3 = self.beta + 3
        vt3 = self.vt**3
        vta = self.vt**self.alpha
        vtb3 = self.vt**b3
        t1 = vt3 / a3
        vtmb = self.vt**-self.beta
        _normfac = vt3 / a3 - vt3 / b3  # unnormalized M(oo)
        
        for i, vel in enumerate(v.ravel()):
            ix = np.unravel_index(i, v.shape)
            if vel <= self.vt:
                res[ix] = vel**a3 / a3 / vta
            else:
                t2 = vtmb / b3 * (vel**b3 - vtb3)
                res[ix] = t1 + t2
        return res.reshape(vs) / _normfac
