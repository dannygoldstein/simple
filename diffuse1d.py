
import scipy
import numpy as np
from scipy.integrate import odeint
import scipy.sparse as sparse
import scipy.sparse.linalg
from dedalus import public as de
import time

def _scale01(x):
    xmin = x.min()
    xmax = x.max()
    
    dif = xmax - xmin
    A = 1 / dif
    B = -A * xmin
    
    scld = A * x + B
    return (A,B,scld)

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

def diffuse1d(phi, D, x, t, coordsys='cartesian'):
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

    Ax,Bx,x_scld = _scale01(x)
    At,Bt,t_scld = _scale01(t)
    Ap,Bp,p_scld = _scale01(phi)
    
    newD = D / At
    D = newD - (Bt * D) / (At * (At * t.max() + Bt))
    
    dt = t_scld[1] - t_scld[0]
    dx = x_scld[1] - x_scld[0]
    
    # define diffeq
    
    if coordsys == 'cartesian':
    
        def phidot(phi, t):
            negF = D * scipy.gradient(phi, dx)
            negF[[0, -1]] = 0. # zero-flux boundary condition
            lhs = scipy.gradient(negF, dx)
            return lhs
        
    elif coordsys == 'spherical':
        
        def phidot(phi, t):
            dphidr = scipy.gradient(phi, dx)
            dphidr[[0,-1]] = 0. # zero-flux boundary condition
            darg = x**2 * dphidr
            outer_deriv = scipy.gradient(darg, dx)
            result = D / x * outer_deriv
            return result            
    
    # check stability
    
    stable = 2 * D * dt / (dx**2)
    try:
        stable = max(stable)
    except TypeError:
        pass
    if stable > 1: 
        raise Exception('Diffusion equation unstable. Decrease D * dt / dx**2')
    
    # integrate and return result at final t
        
    result = (odeint(phidot, p_scld, t_scld) - Bp) / Ap
    return result[-1]

def diffuse1d_crank(phi, D, x, t):

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
    
    

def diffuse1d_ded(phi, D, x, t, coordsys='cartesian'):
    
    nx = len(x) 

    Ax,Bx,x_scld = _scale01(x)
    At,Bt,t_scld = _scale01(t)
    Ap,Bp,p_scld = _scale01(phi)
    
    newD = D / At
    D = newD - (Bt * D) / (At * (At * t.max() + Bt))
    
    x_basis = de.Chebyshev('x', nx, interval=(0,1))
    domain = de.Domain([x_basis], np.float64)
    problem = de.IVP(domain, variables=['u','ux'])
    
    problem.parameters['D'] = D

    if coordsys == 'spherical':
        problem.add_equation('dt(u) - (D / x) * ((2 * x * ux) + (x**2 * dx(ux))) = 0')
    elif coordsys == 'cartesian':
        problem.add_equation('dt(u) - (D * dx(ux)) = 0')
    else:
        raise Exception("Invalid coordsys: %s" % coordsys)

    problem.add_equation('dx(u) - ux = 0')
    problem.add_bc('left(ux) = 0') 
    problem.add_bc('right(ux) = 0')

    solver = problem.build_solver(de.timesteppers.RK443)
    
    x = domain.grid(0)
    u = solver.state['u']
    ux = solver.state['ux']

    u['g'] = p_scld
    u.differentiate('x', out=ux)

    solver.stop_sim_time = np.inf
    solver.stop_wall_time = np.inf
    solver.stop_iteration = len(t)

    dt = t_scld[1] - t_scld[0]

    u_list = [np.copy(u['g'])]
    t_list = [solver.sim_time]

    # main loop
    
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)
        if solver.iteration % 100 == 0:
            print('Completed iteration %d' % solver.iteration)
    end_time = time.time()
    print("Runtime: %.3f sec" % (end_time-start_time))
    
    answer = (u_list[-1] - Bp) / Ap
    return answer


    
