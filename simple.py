
import numpy as np
import diffuse1d
import random

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

# mass msol
# energy erg
# length km
# velocity km / s 

#==============================================================================#
# Utilities
#==============================================================================#

def compute_comp(fe_frac, ni_frac, ime_frac, co_frac):
    return FER * fe_frac + NIC * ni_frac + IME * ime_frac + CAR * co_frac

class SimpleAtmosphere(object):

    def velocity(self, kind='average'):
        """Zone radial velocities (km/s).

        Parameters:
        -----------
        kind, str: 'inner', 'outer', and 'average'
        """

        if kind == 'outer':
            return (self.v_outer / self.nzones) * \
                np.arange(1, self.nzones + 1)
        else:
            boundary_vels = np.concatenate((np.asarray([0.]),
                                               self.velocity(kind='outer')))
            if kind == 'average':
                return (boundary_vels[:-1] + boundary_vels[1:]) / 2.
            elif kind == 'inner':
                return boundary_vels[:-1]
            else:
                raise ValueError('kind must be one of inner, outer, average.')

    
    def __init__(self, iron_mass, nickel_mass, ime_mass,
                 co_mass, specific_ke, mixing_length, nzones=100,
                 v_outer=4.e4, texp=86400., nt=2000):

        """Specific KE in erg / Msol"""
        
        self.iron_mass = iron_mass
        self.nickel_mass = nickel_mass
        self.ime_mass = ime_mass
        self.co_mass = co_mass
        self.specific_ke = specific_ke
        self.mixing_length = mixing_length
        self.nzones = nzones
        self.v_outer = v_outer
        self.texp = texp
        self.nt = nt


        self.ejecta_mass = self.iron_mass + self.nickel_mass + self.ime_mass + \
            self.co_mass

        # initialize composition array

        self.nspec = len(ELE)
        self.comp = np.zeros((self.nzones, self.nspec))

        self.ke = self.specific_ke * self.ejecta_mass # erg 
        self.ve = 2455 * (self.ke / 1e51)**0.5 # km / s
        self.vgrid_outer = self.velocity(kind='outer') # km / s
        self.vgrid_inner = self.velocity(kind='inner') # km / s
        self.vgrid_avg   = self.velocity(kind='average') # km / s

        self.interior_mass = 0.5 * (2.0 - np.exp(-self.vgrid_outer / self.ve) * \
                                   (2.0 + (self.vgrid_outer / self.ve) * \
                                        (2.0 + self.vgrid_outer / self.ve))) \
                                        * self.ejecta_mass

        self.shell_mass = np.concatenate(([self.interior_mass[0]], self.interior_mass[1:] - self.interior_mass[:-1]))
        self.fe_edge_shell = self.interior_mass.searchsorted(self.iron_mass)
        self.nickel_radius = self.iron_mass + self.nickel_mass
        self.ni_edge_shell = self.interior_mass.searchsorted(self.nickel_radius)
        self.ime_radius = self.ime_mass + self.nickel_radius
        self.ime_edge_shell = self.interior_mass.searchsorted(self.ime_radius)
        last_shell = self.nzones - 1

        for i in range(self.nzones)[::-1]:
            sm = self.shell_mass[i] 
            if i <= self.fe_edge_shell:
                if i == self.ni_edge_shell and i == self.ime_edge_shell and i == self.fe_edge_shell:
                    # rare case
                    ni_frac = self.nickel_mass / sm
                    ime_frac = self.ime_mass / sm

                    # check for CO
                    co_adv_mass = self.shell_mass[i+1:].sum()
                    co_frac = (self.co_mass - co_adv_mass) / sm

                    # iron is what's left 
                    fe_frac = 1 - ni_frac - ime_frac - co_frac

                    self.comp[i] = compute_comp(fe_frac, ni_frac, ime_frac, co_frac)

                elif i == self.ni_edge_shell and i == self.fe_edge_shell:
                    ni_frac = self.nickel_mass / sm

                    # check for IME
                    ime_adv_mass = self.shell_mass[i+1:self.ime_edge_shell].sum()
                    ime_adv_mass += self.shell_mass[self.ime_edge_shell] * self.comp[self.ime_edge_shell][IME_INDS].sum()
                    ime_frac = (self.ime_mass - ime_adv_mass) / sm
                    fe_frac = 1 - ime_frac - ni_frac
                    self.comp[i] = compute_comp(fe_frac, ni_frac, ime_frac, 0.)

                elif i == self.fe_edge_shell:
                    nickel_adv_mass = self.shell_mass[i+1:self.ni_edge_shell].sum()
                    nickel_adv_mass += self.shell_mass[self.ni_edge_shell] * self.comp[self.ni_edge_shell][NI_INDS]
                    ni_frac = (self.nickel_mass - nickel_adv_mass) / sm
                    fe_frac = 1 - ni_frac
                    self.comp[i] = compute_comp(fe_frac, ni_frac, 0., 0.)
                else:
                    self.comp[i] = FER
            elif i <= self.ni_edge_shell:
                if i == self.ime_edge_shell and i == self.ni_edge_shell:
                    # nickel ime and co
                    ime_frac = self.ime_mass / sm

                    # check for CO
                    co_adv_mass = self.shell_mass[i+1:].sum()
                    co_frac = (self.co_mass - co_adv_mass) / sm
                    ni_frac = 1 - co_frac - ime_frac
                    self.comp[i] = compute_comp(0., ni_frac, ime_frac, co_frac)

                elif i == self.ni_edge_shell:
                    ime_adv_mass = self.shell_mass[i+1:self.ime_edge_shell].sum()
                    ime_adv_mass += self.shell_mass[self.ime_edge_shell] * self.comp[self.ime_edge_shell][IME_INDS].sum()
                    ime_frac = (self.ime_mass - ime_adv_mass) / sm
                    ni_frac = 1 - ime_frac
                    self.comp[i] = compute_comp(0., ni_frac, ime_frac, 0.)
                else:
                    self.comp[i] = NIC

            elif i <= self.ime_edge_shell:
                if i == self.ime_edge_shell:
                    remaining_co = self.co_mass - self.shell_mass[i+1:].sum()
                    co_frac = remaining_co / sm
                    ime_frac = 1 - co_frac 
                    self.comp[i] = compute_comp(0., 0., ime_frac, co_frac)
                else:
                    self.comp[i] = IME
            else:
                self.comp[i] = CAR

    #==============================================================================#
    # Diffusion
    #==============================================================================#

        self.vol_cm3 = 4 * np.pi / 3 * (self.vgrid_outer**3 - self.vgrid_inner**3) * KM_CM**3 * self.texp**3
        self.vol_km3 = self.vol_cm3 / KM_CM**3
        self.rho_g_cm3 = self.shell_mass * MSUN_G / self.vol_cm3
        self.rho_Msun_km3 = self.shell_mass / self.vol_km3
        self.phi = self.rho_Msun_km3[:, None] / WEI[None, :] * self.comp

        self.mixing_length_km = self.mixing_length * self.texp
        self.x_avg_km = self.vgrid_avg * self.texp

        if self.mixing_length > 0:

            newphis = []
            newrhos = []
            
            self.D = 1. 
            self.t = np.linspace(0,self.mixing_length_km**2,self.nt)

            for i in range(len(WEI)):

                this_phi = self.phi.T[i]

                new_phi = diffuse1d.diffuse1d_crank(this_phi, self.D, self.x_avg_km, self.t)
                new_phi = new_phi[-1]
                new_rho = WEI[i] * new_phi

                newphis.append(new_phi)
                newrhos.append(new_rho)

            newphis = np.array(newphis).T
            newrhos = np.array(newrhos).T

            self.rho_Msun_km3 = newrhos.sum(1)
            self.comp = newrhos / self.rho_Msun_km3[:, None]
            
            self.shell_mass = self.rho_Msun_km3 * self.vol_km3
            self.interior_mass = np.cumsum(self.shell_mass)
            
        self.spec_mass = self.shell_mass[:, None] * self.comp


    def plot(self, show=True):
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.pyplot import cm
        except Exception as e:
            raise e 
        
        try:
            import seaborn as sns
        except:
            sb = False
        else:
            sb = True

        if sb:
            sns.set_style('ticks')
        
        fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(8,6))
        
        colors = ['b','g','r','purple','c','m','y','k']
                    
        for row,name,c in zip(self.comp.T,ELE,colors):
            axarr[0,0].semilogy(self.vgrid_avg,row,label=name,color=c)
        axarr[0,0].set_ylabel('mass fraction')
        axarr[0,0].set_ylim(1e-3,1)
        axarr[0,0].legend(frameon=True,loc='best')
        axarr[0,0].set_xlabel('velocity (km/s)')
        
    
        for row,name,c in zip(self.comp.T,ELE,colors):
            axarr[0,1].semilogy(self.interior_mass,row,label=name,color=c)
        axarr[0,1].set_xlabel('interior mass (msun)')
        axarr[0,1].set_ylabel('mass fraction')
        axarr[0,1].set_ylim(1e-3,1)
        
        axarr[1,0].semilogy(self.vgrid_avg,self.rho_Msun_km3)
        axarr[1,0].set_ylabel('rho (Msun / km3)')
        axarr[1,0].set_xlabel('velocity (km/s)')

        axarr[1,1].semilogy(self.interior_mass,self.rho_Msun_km3)
        axarr[1,1].set_ylabel('rho (Msun / km3)')
        axarr[1,1].set_xlabel('interior mass (msun)')

        for ax in axarr.ravel():
            ax.minorticks_on()

        if sb:
            sns.despine()

        title = 'mfe=%.3f,mni=%.3f,mime=%.3f,mco=%.3f'
        
        mfe = self.spec_mass[:, FE_INDS].sum()
        mni = self.spec_mass[:, NI_INDS].sum()
        mime = self.spec_mass[:, IME_INDS].sum()
        mco = self.spec_mass[:, CO_INDS].sum()
        
        fig.suptitle(title % (mfe,mni,mime,mco))

        if show:
            fig.show()
            
        return (fig, axarr)
        
        
        
        
        
