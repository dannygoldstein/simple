import abc
from layers import *
import numpy as np
import random

__whatami__ = 'Simple supernova atmospheres.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

KM_CM = 1e5
MSUN_G = 1.99e33

class Atmosphere(object):

    __metaclass__ = abc.ABCMeta

    def _indexof(self, element):
        """The index of element `element` in the species array `spec`."""
        return self.spec.index(element)

    @abc.abstractproperty
    def ejecta_mass(self):
        pass

    @abc.abstractproperty
    def spec(self):
        pass

    @abc.abstractproperty
    def nzones(self):
        pass

    @abc.abstractproperty
    def comp(self):
        pass

    @abc.abstractproperty
    def v_outer(self):
        pass

    @abc.abstractproperty
    def rho_Msun_km3(self):
        pass

    @abc.abstractproperty
    def interior_mass(self):
        pass

    @abc.abstractproperty
    def texp(self):
        pass

    @property
    def vol_cm3(self):
        vo = self.velocity(kind='outer')
        vi = self.velocity(kind='inner')
        vol = 4 * np.pi / 3 * (vo**3 - vi**3) * KM_CM**3 * self.texp**3
        return vol

    @property
    def vol_km3(self):
        return self.vol_cm3 / KM_CM**3

    @property
    def shell_mass(self):
        im = self.interior_mass
        return np.concatenate(([im[0]], im[1:] - im[:-1]))
    
    @property
    def rho_g_cm3(self):
        return self.rho_Msun_km3 * MSUN_G / (KM_CM)**3

    @property
    def spec_mass(self):
        return self.shell_mass[:, None] * self.comp

    @property
    def nspec(self):
        return len(self.spec)

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
        vga = self.velocity(kind='average')
                    
        for row,name,c in zip(self.comp.T,ele,colors):
            axarr[0,0].semilogy(vga,row,label=name,color=c)
        axarr[0,0].set_ylabel('mass fraction')
        axarr[0,0].set_ylim(1e-3,1)
        axarr[0,0].legend(frameon=True,loc='best')
        axarr[0,0].set_xlabel('velocity (km/s)')
    
        for row,name,c in zip(self.comp.T,ele,colors):
            axarr[0,1].semilogy(self.interior_mass,row,label=name,color=c)
        axarr[0,1].set_xlabel('interior mass (msun)')
        axarr[0,1].set_ylabel('mass fraction')
        axarr[0,1].set_ylim(1e-3,1)
        
        axarr[1,0].semilogy(vga,self.rho_Msun_km3)
        axarr[1,0].set_ylabel('rho (Msun / km3)')
        axarr[1,0].set_xlabel('velocity (km/s)')

        axarr[1,1].semilogy(self.interior_mass,self.rho_Msun_km3)
        axarr[1,1].set_ylabel('rho (Msun / km3)')
        axarr[1,1].set_xlabel('interior mass (msun)')

        for ax in axarr.ravel():
            ax.minorticks_on()

        if sb:
            sns.despine()

        elenames = [e.repr for e in self.spec]
        elemasses = self.spec_mass.sum(axis=0)

        title = ', '.join([r'$M_{%s}=%.3f$' % e for e in zip(elenames, elemasses)])
        
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
            

class StratifiedAtmosphere(Atmosphere):
    """A simple supernova atmosphere. Composed of one or more Layers."""

    def __init__(self, layers, masses, profile, nzones=100, v_outer=4.e4,
                 texp=86400.):

        # state
        self.layers = layers
        self.profile = profile
        self.masses = masses

        # parameters
        self._nzones = nzones
        self._v_outer = v_outer
        self._texp = texp

        # grid
        self.vgrid_outer = self.velocity(kind='outer') # km / s
        self.vgrid_inner = self.velocity(kind='inner') # km / s
        self.vgrid_avg   = self.velocity(kind='average') # km / s

        # enumerate unique elements in the atmosphere
        spec = [] 
        for layer in self.layers:
            spec += layer.abundances.keys()
        self._spec = sorted(set(spec), key=lambda ele: ele.weight)[::-1]
        
        # initialize composition array
        self._comp = np.zeros((self.nzones, self.nspec))
        self.fracs = np.zeros_like(self._comp)

        radii = [0]
        for i in range(len(self.layers)):
            radii.append(radii[i] + self.masses[i])
        self.radii = np.asarray(radii[1:])
        self.edge_shells = map(self.interior_mass.searchsorted, self.radii)

        # if atmosphere does not contain >99% of the mass, raise an error
        if not self.profile.accuracy_criterion(self.v_outer, .99):
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
                        self._comp[i] = self._abun(self.fracs[i])

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
                self._comp[i] = self._abun(self.fracs[i])
                

    @property
    def spec(self):
        return self._spec

    @property
    def ejecta_mass(self):
        return self.interior_mass[-1]

    @property
    def interior_mass(self):
        vo = self.velocity(kind='outer')
        return self.profile(vo) * sum(self.masses)

    @property
    def comp(self):
        return self._comp

    @property
    def nzones(self):
        return self._nzones

    @property
    def v_outer(self):
        return self._v_outer

    @property
    def texp(self):
        return self._texp

    @property
    def rho_Msun_km3(self):
        return self.shell_mass / self.vol_km3            

    def _layermass(self, layer, i):
        """The mass of layer `layer` in zone `i`."""
        return self.shell_mass[i] * self.fracs[i, self.layers.index(layer)]

    def _abun(self, fracs):
        """Compute the composition array (mass fractions of each element, in
        the order of self.spec), of an atmospheric zone composed of
        `fracs` relative fractions of `layers` layers."""

        result = np.zeros(self.nspec)
        for layer, frac in zip(self.layers, fracs):
            for elem in layer.abundances:
                result[self._indexof(elem)] += layer.abundances[elem] * frac
        return result
    
class MixedAtmosphere(Atmosphere):

    @property
    def nzones(self):
        return self._nzones

    @property
    def v_outer(self):
        return self._v_outer

    @property
    def texp(self):
        return self._texp

    @property
    def spec(self):
        return self._spec

    @property
    def rho_Msun_km3(self):
        return self._rho_Msun_km3

    @property
    def ejecta_mass(self):
        return self.interior_mass[-1]

    @property
    def comp(self):
        return self._comp

    @property
    def interior_mass(self):
        return np.cumsum(self.rho_Msun_km3 * self.vol_km3)

    def __init__(self, spec, comp, dens, nzones, texp, v_outer):

        # state
        self._spec = spec
        self._rho_Msun_km3 = dens
        self._comp = comp
        
        # parameters
        self._texp = texp
        self._v_outer = v_outer
        self._nzones = nzones
        
