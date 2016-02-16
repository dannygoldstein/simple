
import numpy as np
import diffuse1d

MSUN_G = 1.99e33
KM_CM = 1e5

#==============================================================================#
# Abundances
#==============================================================================#

ELE = ['54Fe', '56Ni', '28Si', '32S', '40Ca', '36Ar', '12C', '16O']
Z   = np.array([26, 28, 14, 16, 20, 18, 6, 18])
A   = np.array([54, 56, 28, 32, 40, 36, 12, 16])
WEI = np.array([53.939,55.94,27.976,31.972,39.962,35.967,12.,15.994])
IME = np.array([0., 0., 0.53 ,  0.32,  0.062,  0.083,    0.,    0.])
FER = np.array([1., 0.,   0. ,    0.,     0.,     0.,    0.,    0.])
NIC = np.array([0., 1.,   0. ,    0.,     0.,     0.,    0.,    0.])
CAR = np.array([0., 0.,   0. ,    0.,     0.,     0.,    .5,    .5])
IME = IME / IME.sum() # renormalize 

IME_INDS = [2,3,4,5]
CO_INDS = [-2, -1]
FE_INDS = 0
NI_INDS = 1

#==============================================================================#
# Gridding
#==============================================================================#

def velocity(v_outer, num_zones, kind='average'):
    """Zone radial velocities (km/s).

    Parameters:
    -----------
    kind, str: 'inner', 'outer', and 'average'
    """

    if kind == 'outer':
        return (v_outer / num_zones) * \
            np.arange(1, num_zones + 1)
    else:
        boundary_vels = np.concatenate((np.asarray([0.]),
                                           velocity(v_outer, 
                                                    num_zones, 
                                                    kind='outer')))
        if kind == 'average':
            return (boundary_vels[:-1] + boundary_vels[1:]) / 2.
        elif kind == 'inner':
            return boundary_vels[:-1]
        else:
            raise ValueError('kind must be one of inner, outer, average.')

# mass msol
# energy erg
# length km
# velocity km / s 

#==============================================================================#
# Utilities
#==============================================================================#

def compute_comp(fe_frac, ni_frac, ime_frac, co_frac):
    return FER * fe_frac + NIC * ni_frac + IME * ime_frac + CAR * co_frac

def simple_atmosphere(iron_zone_mass, nickel_zone_mass, ime_zone_mass,
                      co_zone_mass, specific_ke, mixing_length, nzones=100,
                      v_outer=4.e4):

    """Specific KE in erg / Msol"""
    
    ejecta_mass = iron_zone_mass + nickel_zone_mass + ime_zone_mass + \
        co_zone_mass
    
    # initialize composition array
    
    nspec = len(ELE)
    comp = np.zeros((nzones, nspec))
    
    ke = specific_ke * ejecta_mass # erg 
    ve = 2455 * (ke / 1e51)**0.5 # km / s
    vgrid_outer = velocity(v_outer, nzones, kind='outer') # km / s
    vgrid_inner = velocity(v_outer, nzones, kind='inner') # km / s
    vgrid_avg   = velocity(v_outer, nzones, kind='average') # km / s
    
    interior_mass = 0.5 * (2.0 - np.exp(-vgrid_outer / ve) * \
                               (2.0 + (vgrid_outer / ve) * \
                                    (2.0 + vgrid_outer / ve))) \
                                    * ejecta_mass
    
    shell_mass = np.concatenate(([interior_mass[0]], interior_mass[1:] - interior_mass[:-1]))
    fe_edge_shell = interior_mass.searchsorted(iron_zone_mass)
    nickel_radius = iron_zone_mass + nickel_zone_mass
    ni_edge_shell = interior_mass.searchsorted(nickel_radius)
    ime_radius = ime_zone_mass + nickel_radius
    ime_edge_shell = interior_mass.searchsorted(ime_radius)
    last_shell = nzones - 1
    
    for i in range(nzones)[::-1]:
        sm = shell_mass[i] 
        if i <= fe_edge_shell:
            if i == ni_edge_shell and i == ime_edge_shell and i == fe_edge_shell:
                # rare case
                ni_frac = nickel_zone_mass / sm
                ime_frac = ime_zone_mass / sm
                
                # check for CO
                co_adv_mass = shell_mass[i+1:].sum()
                co_frac = (co_zone_mass - co_adv_mass) / sm
                
                # iron is what's left 
                fe_frac = 1 - ni_frac - ime_frac - co_frac

                comp[i] = compute_comp(fe_frac, ni_frac, ime_frac, co_frac)

            elif i == ni_edge_shell and i == fe_edge_shell:
                ni_frac = nickel_zone_mass / sm
                
                # check for IME
                ime_adv_mass = shell_mass[i+1:ime_edge_shell].sum()
                ime_adv_mass += shell_mass[ime_edge_shell] * comp[ime_edge_shell][IME_INDS].sum()
                ime_frac = (ime_zone_mass - ime_adv_mass) / sm
                fe_frac = 1 - ime_frac - ni_frac
                comp[i] = compute_comp(fe_frac, ni_frac, ime_frac, 0.)
                
            elif i == fe_edge_shell:
                nickel_adv_mass = shell_mass[i+1:ni_edge_shell].sum()
                nickel_adv_mass += shell_mass[ni_edge_shell] * comp[ni_edge_shell][NI_INDS]
                ni_frac = (nickel_zone_mass - nickel_adv_mass) / sm
                fe_frac = 1 - ni_frac
                comp[i] = compute_comp(fe_frac, ni_frac, 0., 0.)
            else:
                comp[i] = FER
        elif i <= ni_edge_shell:
            if i == ime_edge_shell and i == ni_edge_shell:
                # nickel ime and co
                ime_frac = ime_zone_mass / sm
                
                # check for CO
                co_adv_mass = shell_mass[i+1:].sum()
                co_frac = (co_zone_mass - co_adv_mass) / sm
                ni_frac = 1 - co_frac - ime_frac
                comp[i] = compute_comp(0., ni_frac, ime_frac, co_frac)
                
            elif i == ni_edge_shell:
                ime_adv_mass = shell_mass[i+1:ime_edge_shell].sum()
                ime_adv_mass += shell_mass[ime_edge_shell] * comp[ime_edge_shell][IME_INDS].sum()
                ime_frac = (ime_zone_mass - ime_adv_mass) / sm
                ni_frac = 1 - ime_frac
                comp[i] = compute_comp(0., ni_frac, ime_frac, 0.)
            else:
                comp[i] = NIC
                
        elif i <= ime_edge_shell:
            if i == ime_edge_shell:
                remaining_co = co_zone_mass - shell_mass[i+1:].sum()
                co_frac = remaining_co / sm
                ime_frac = 1 - co_frac 
                comp[i] = compute_comp(0., 0., ime_frac, co_frac)
            else:
                comp[i] = IME
        else:
            comp[i] = CAR

#==============================================================================#
# Diffusion
#==============================================================================#

    zone_size = vgrid_outer - vgrid_inner
    vol_cm3 = 4 * np.pi / 3 * (vgrid_outer**3 - vgrid_inner**3) * (KM_CM**3)
    rho_g_cm3 = shell_mass * MSUN_G / vol_cm3
    
    """
    for i in range(len(WEI)):
        mf = comp.T[i] / zone_size 

        # set D = 1
        D = 1. 
        t = np.linspace(0,mixing_length**2,200)
        newmf = diffuse1d.diffuse1d(mf, D, vgrid_avg, t)
        
        comp[:, i] = newmf * zone_size
    """
    return comp, rho_g_cm3, vgrid_avg
        
        
        
        
