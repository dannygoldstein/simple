import abc
import scipy
import numpy as np
from scipy.integrate import odeint
import scipy.sparse as sparse
import scipy.sparse.linalg

__whatami__ = 'Mixing for simple supernova atmospheres.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'


def _diffuse1d(phi, D, x, t):
    """Let a 1D concentration `phi` defined over spatial grid `x` diffuse
    over temporal grid `t` subject to (potentially) spatially varying
    diffusion coefficient `D` by integrating the 1D diffusionx equation
    using a Crank Nicolson solver. Return the final concentration.

    Parameters
    ---------- 

    phi : array
        Concentration profile.
    D : array or scalar
        (Potentially) spatially-varying diffusion coefficient. 
    x : array
        Linear spatial grid for `phi` and `D`. 
    t : array
        Linear temporal grid over which to integrate the diffusion
        equation.
    """

    # always spherical
    # make sure t starts at 0

    x_scld = x / x.max() # "R"
    phi_scld = phi / phi.max()
    t_scld = t / t.max()

    K = D / x.max()**2 * t.max()

    dx = x_scld[1] - x_scld[0]
    dt = t_scld[1] - t_scld[0]

    NX = x.size
    NT = t.size

    # create A1
    A1 = np.diag(np.ones(NX) * -2)
    A1[0, 1] = 2
    A1[-1, -2] = 2
    A2 = np.diag(np.zeros(NX))
    for i in range(1, NX-1):
        A1[i, [i-1,i+1]] = 1
        A2[i, [i-1,i+1]] = (-1/x_scld[i], 1/x_scld[i]) # grid factor 1/x 

    A1 = scipy.sparse.csr_matrix(A1)
    A2 = scipy.sparse.csr_matrix(A2)


    # u will store all of the time step solutions for now it just has
    # one element, the initial condition

    u = [phi_scld]


    t1 = (K / dx**2) * A1
    t2 = K / dx * A2
    F = t1 + t2

    I = scipy.sparse.identity(NX)

    for step in t:
        A = (I - dt * F)
        b = (I + dt * F).dot(u[-1])
        sol = sparse.linalg.spsolve(A, b)
        u.append(sol)

    return np.array(u) * phi.max()

class Mixer(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, atm):
        pass

class DiffusionMixer(Mixer):

    def __init__(self, mixing_length, nt=2000):
        if mixing_length <= 0:
            raise ValueError("mixing length must be greater than 0.")
        self.mixing_length = mixing_length
        self.nt = nt

    def __call__(self, atm):

        # convert the mass fractions to concentrations
        wei = np.asarray([ele.weight for ele in atm.spec])
        phi = atm.rho_Msun_km3[:, None] / wei[None, :] * atm.comp

        # recast the diffusion paramters somewhat
        mixing_length_km = self.mixing_length * atm.texp
        x_avg_km = atm.vgrid_avg * atm.texp
        t = np.linspace(0, mixing_length_km**2, self.nt)

        # do diffusion and store the results
        newphiso = np.zeros((atm.nzones, atm.nspec))
        newrhos = np.zeros((atm.nzones, atm.nspec))

        for i in range(atm.nspec):
            this_phi = phi.T[i]
            new_phi = _diffuse1d(this_phi, 1., x_avg_km, t)[-1]
            new_rho = wei[i] * new_phi
            newphis[:, i] = new_phi
            newrhos[:, i] = new_rho

        # recompute atmospheric properties
        rho_Msun_km3 = newrhos.sum(1)
        comp = newrhos / rho_Msun_km3[:, None]

        return MixedAtmosphere(atm.spec, comp, rho_Msun_km3,
                               atm.nzones, atm.texp, atm.v_outer)
