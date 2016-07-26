from layers import *
import numpy as np
import diffuse1d
import random

__whatami__ = 'Simple supernova atmosphere generator.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

MSUN_G = 1.99e33
KM_CM = 1e5

class SimpleAtmosphere(object):
    """A simple supernova atmosphere. Composed of one or more Layers."""
    
    def __init__(self, layers, masses, mixing_length, specific_ke,
                 nzones=100, v_outer=4.e4, texp=86400., nt=2000):

        self.layers = layers

        if len(set(self.layers)) != len(layers):
            raise ValueError('Currently cannot handle atmospheres '\
                                 'with repeated layers.')
            
        self.masses = masses
        self.specific_ke = specific_ke
        self.mixing_length = mixing_length

        self.nzones = nzones
        self.v_outer = v_outer
        self.texp = texp
        self.nt = nt

        self.ejecta_mass = sum(self.masses)

        # enumerate unique elements in the atmosphere
        self.spec = [] 
        for layer in self.layers:
            self.spec += layer.abundances.keys()
        self.spec = sorted(set(self.spec), key=lambda ele: ele.weight)[::-1]
        
        # initialize composition array
        self.nspec = len(self.spec)
        self.comp = np.zeros((self.nzones, self.nspec))
        self.fracs = np.zeros_like(self.comp)

        self.ke = self.specific_ke * self.ejecta_mass # erg 
        self.ve = 2455 * (self.ke / 1e51)**0.5 # km / s
        self.vgrid_outer = self.velocity(kind='outer') # km / s
        self.vgrid_inner = self.velocity(kind='inner') # km / s
        self.vgrid_avg   = self.velocity(kind='average') # km / s

        # exponential density profile
        self.interior_mass = 0.5 * (2.0 - np.exp(-self.vgrid_outer / self.ve) * \
                                   (2.0 + (self.vgrid_outer / self.ve) * \
                                        (2.0 + self.vgrid_outer / self.ve))) \
                                        * self.ejecta_mass
        
        self.shell_mass = np.concatenate(([self.interior_mass[0]],
                                          self.interior_mass[1:] - \
                                          self.interior_mass[:-1]))

        radii = [0]
        for i in range(len(self.layers)):
            radii.append(radii[i] + masses[i])
        self.radii = np.asarray(radii[1:])
        self.edge_shells = map(self.interior_mass.searchsorted, self.radii)

        if self.v_outer < 8.4059 * self.ve:
            raise ValueError('Model does not extend out far enough '
                             'in velocity space to capture all the '
                             'mass. Increase v_outer.')

        # Traverse the zones in reverse order. 
        for i in range(self.nzones)[::-1]:
            sm = self.shell_mass[i]
            isedge = np.asarray(self.edge_shells[:-1]) == i
            nedge = isedge[isedge].size
            if nedge == 0:
                # Pure layer
                bounds = [0] + self.edge_shells
                for j in range(len(self.layers)):
                    if bounds[j] <= i <= bounds[j+1]: # TODO: should one of these be strict?
                        self.fracs[i, j] = 1.
                        self.comp[i] = self._abun(self.fracs[i])

            else:
                # Transition layer

                # ending is the index of the layer that exists in
                # underlying zones that ends on entry into this zone.

                # starting is the index of the layer that first
                # appears in this zone.
                
                ending, kmax = np.argwhere(isedge)[[0, -1], 0]
                starting = kmax + 1

                nlayer = starting - ending + 1 # number of layers in this zone

                # The mass in the overlying zones of this zone's
                # topmost layer.
                adv_mass = sum([self._layermass(self.layers[starting], l)
                                for l in range(i+1, self.nzones)])

                # The mass in this zone of this zone's topmost layer.
                remaining_mass = self.masses[starting] - adv_mass

                # The fraction in this zone of this zone's topmost layer.
                self.fracs[i, starting] = remaining_mass / sm

                # The fractions of the other layers:
                for k in range(ending+1, starting)[::-1]:
                    self.fracs[i, k] = self.masses[k] / sm
                self.fracs[i, ending] = 1 - sum(self.fracs[i, ending+1:])
                self.comp[i] = self._abun(self.fracs[i])
            
    #==============================================================================#
    # Diffusion
    #==============================================================================#

        wei = np.asarray([ele.weight for ele in self.spec])
        self.vol_cm3 = 4 * np.pi / 3 * (self.vgrid_outer**3 - \
                                            self.vgrid_inner**3) * \
                                            KM_CM**3 * self.texp**3

        self.vol_km3 = self.vol_cm3 / KM_CM**3
        self.rho_g_cm3 = self.shell_mass * MSUN_G / self.vol_cm3
        self.rho_Msun_km3 = self.shell_mass / self.vol_km3
        self.phi = self.rho_Msun_km3[:, None] / wei[None, :] * self.comp

        self.mixing_length_km = self.mixing_length * self.texp
        self.x_avg_km = self.vgrid_avg * self.texp

        if self.mixing_length > 0:

            newphis = []
            newrhos = []
            
            self.D = 1. 
            self.t = np.linspace(0,self.mixing_length_km**2,self.nt)

            for i in range(len(wei)):

                this_phi = self.phi.T[i]

                new_phi = diffuse1d.diffuse1d(this_phi, self.D, self.x_avg_km, self.t)
                new_phi = new_phi[-1]
                new_rho = wei[i] * new_phi

                newphis.append(new_phi)
                newrhos.append(new_rho)

            newphis = np.array(newphis).T
            newrhos = np.array(newrhos).T

            self.rho_Msun_km3 = newrhos.sum(1)
            self.comp = newrhos / self.rho_Msun_km3[:, None]
            
            self.shell_mass = self.rho_Msun_km3 * self.vol_km3
            self.rho_g_cm3 = self.shell_mass * MSUN_G / self.vol_cm3
            self.interior_mass = np.cumsum(self.shell_mass)
            
        self.spec_mass = self.shell_mass[:, None] * self.comp

    def _layermass(self, layer, i):
        """The mass of layer `layer` in zone `i`."""
        return self.shell_mass[i] * self.fracs[i, self.layers.index(layer)]

    def _indexof(self, element):
        """The index of element `element` in the species array `spec`."""
        return self.spec.index(element)

    def _abun(self, fracs):
        """Compute the composition array (mass fractions of each element, in
        the order of self.spec), of an atmospheric zone composed of
        `fracs` relative fractions of `layers` layers."""

        result = np.zeros(self.nspec)
        for layer, frac in zip(self.layers, fracs):
            for elem in layer.abundances:
                result[self._indexof(elem)] += layer.abundances[elem] * frac
        return result
    
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
        
        colors = ['b','g','r','purple','c','m','y','k','orange','indigo','violet']

        ele = [elem.repr for elem in self.spec]
                    
        for row,name,c in zip(self.comp.T,ele,colors):
            axarr[0,0].semilogy(self.vgrid_avg,row,label=name,color=c)
        axarr[0,0].set_ylabel('mass fraction')
        axarr[0,0].set_ylim(1e-3,1)
        axarr[0,0].legend(frameon=True,loc='best')
        axarr[0,0].set_xlabel('velocity (km/s)')
    
        for row,name,c in zip(self.comp.T,ele,colors):
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
            
        layermasses = [self.spec_mass[:, [self._indexof(ele) for ele in layer]].sum() 
                       for layer in self.layers]

        title = ', '.join([r'$M_{%s}=%.3f$' % (e[0].name, e[1]) 
                           for e in zip(self.layers, layermasses)])        
        
        fig.suptitle(title)

        if show:
            fig.show()
            
        return (fig, axarr)

    def write(self, outfile):
        
        try:
            import h5py
        except ImportError as e:
            raise e
        
        mod = h5py.File(outfile)
        mod['Version'] = 1
        for i, comp in enumerate(self.comp.T):
            mod['comp%d' % i] = comp.astype('<f8')
        
        mod['A'] = A.astype('<i4')
        mod['Z'] = Z.astype('<i4')
        
        mod['rho'] = self.rho_g_cm3.astype('<f8')
        mod['vx'] = self.velocity(kind='average').astype('<f8') * 1e5 # cm / s
        mod['time'] = np.asarray(self.texp).astype('<f8')
        mod['vol'] = self.vol_cm3
        mod.close()
