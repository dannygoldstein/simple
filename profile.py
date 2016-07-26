import abc
import numpy as np
from scipy.integrate import quad

__whatami__ = 'Density profiles for simple supernova atmospheres.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

def _safequad(func, a, b):
    y, eps = quad(func, a, b)
    if eps / np.abs(y) > 1e-5:
        raise RuntimeError('quad convergence uncertain.')
    return y

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

class Exponential(DensityProfile):
    """An exponential density profile."""

    def __init__(self, ke):
        self.ke = ke  # erg
        self.ve = 2455 * (self.ke / 1e51)**0.5  # km / s

    def __call__(self, v):
        return 0.5 * (2.0 - np.exp(-v / self.ve) *
                      (2.0 + (v / self.ve) *
                       (2.0 + v / self.ve)))

class BrokenPowerLaw(DensityProfile):
    """A broken power law."""

    def __init__(self, alpha, beta, vt):
        self.alpha = alpha
        self.beta = beta
        self.vt = vt
        self._normfac = _safequad(self._integrand, 0, np.inf)

    def _integrand(self, v):
        return v**2 * self._dprof(v)

    def _dprof(self, v):
        if v >= self.vt:
            return (v / self.vt)**beta
        else:
            return (v / self.vt)**alpha

    def __call__(self, v):
        v = np.asarray(v)
        vs = v.shape
        v = np.atleast_1d(v)
        res = np.zeros_like(v)
        for i, vel in enumerate(v.ravel()):
            ix = np.unravel_indices(i, v.shape)
            res[ix] = _safequad(self._integrand, 0, vel)
        return res.reshape(vs) / self._normfac
