
import scipy
from scipy.integrate import odeint

def mass_fraction_to_concentration(spec_mf, rho_tot, spec_a):
    """Convert an array representing the mass fraction of a species
    (dimensionless) to an array representing the concentration of the
    species (mol / cm3).
    
    Arguments
    ---------
    
    spec_mf : ndarray
        The mass fraction of the species in each zone.
        
    rho_tot : ndarray (same shape as spec_mf)
        The total density of each zone (not just the density of this
        species).
        
    spec_a  : ndarray (same shape as spec_mf)
        The molar mass (in grams) of the species in each zone. 
    """
    
    # density of this species 
    density = spec_mf * rho_tot # g / cm3

    # concentration of this species
    concentration = density / spec_a # mol / cm3

    return concentration

def concentration_to_mass_fraction(spec_phi, rho_tot, spec_a):
    """Convert an array representing the concentration of the species
    (mol / cm3) to an array representing the mass fraction of a
    species (dimensionless).
    
    Arguments
    ---------
    
    spec_phi : ndarray
        The concentration (mol / cm3) of the species in each zone.
        
    rho_tot : ndarray (same shape as spec_mf)
        The total density of each zone (not just the density of this
        species).
        
    spec_a  : ndarray (same shape as spec_mf)
        The molar mass (in grams) of the species in each zone. 
    """
    
    density = spec_a * spec_phi # g / cm3
    mf =  density / rho_tot # dimensionless

    return mf

def diffuse1d(phi, D, x, t):
    """Let a 1D concentration `phi` defined over spatial grid `x`
    diffuse over temporal grid `t` subject to (potentially) spatially
    varying diffusion coefficient `D` by integrating the 1D diffusion
    equation using lsoda from ODEPACK. Return the final concentration.
    
    Arguments
    ---------
    
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

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    # define diffeq
    
    def phidot(phi, t):
        negF = D * scipy.gradient(phi, dx)
        negF[[0, -1]] = 0. # zero-flux boundary condition
        lhs = scipy.gradient(negF, dx)
        return lhs
    
    # check stability
    
    stable = 2 * D * dt / (dx**2)
    try:
        stable = max(stable)
    except TypeError:
        pass
    if stable > 1: 
        raise Exception('Diffusion equation unstable. Decrease D * dt / dx**2')
    
    # integrate and return result at final t
        
    result = odeint(phidot, phi, t)
    return result[-1]
