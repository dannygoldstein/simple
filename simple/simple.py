import abc
from .profile import *
from .constants import *
from .elements import find_element_by_AZ, find_element_by_name, Co56, Fe56
import numpy as np


__whatami__ = 'Simple supernova atmospheres.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__all__ = ['StratifiedAtmosphere', 'Atmosphere']

class _AtmosphereBase(object):

    __metaclass__ = abc.ABCMeta

    def _indexof(self, element):
        """The index of element `element` in the species array `spec`."""
        return self.spec.index(element)

    @abc.abstractproperty
    def interior_thermal_energy(self):
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
    def kinetic_energy_erg(self):
        density = self.rho_g_cm3
        velint = (self.velocity(kind='outer')**5 - self.velocity(kind='inner')**5) / 5.
        velint *= KM_CM**5 * self.texp**3
        return (0.5 * velint * density * 4 * np.pi).sum()
    
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
    def shell_thermal_energy(self):
        ie = self.interior_thermal_energy
        return np.concatenate(([ie[0]], ie[1:] - ie[:-1]))

    @property
    def rho_g_cm3(self):
        return self.rho_Msun_km3 * MSUN_G / (KM_CM)**3

    @property
    def spec_mass(self):
        return self.shell_mass[:, None] * self.comp

    @property
    def nspec(self):
        return len(self.spec)

    @property
    def ejecta_mass(self):
        return self.interior_mass[-1]
    
    @property
    def thermal_energy(self):
        return self.interior_thermal_energy[-1]

    @property
    def T_K(self):
        return (self.shell_thermal_energy / self.vol_cm3 / A)**0.25

    def plot(self, show=True, thermal=False, ncol=5):
        
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
        
        if thermal: 
            fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=(8.5,11))
        else:
            fig, axarr = plt.subplots(nrows=2,ncols=2, figsize=(9.5,8.5))
        
        colors = ['b','g','r','purple','c','m','y','k','orange','indigo','violet']
        ls = ['-', '--', ':', '-.']
        styles = [{'color':c, 'ls':linestyle} for c in colors for linestyle in ls]

        ele = [elem.repr for elem in self.spec]
        vga = self.velocity(kind='average')
                    
        for row,name,style in zip(self.comp.T,ele,styles):
            axarr[0,0].semilogy(vga,row,label=name,**style)
        axarr[0,0].set_ylabel('mass fraction')
        axarr[0,0].set_ylim(1e-6,1)
        handles, labels = axarr[0,0].get_legend_handles_labels()
        axarr[0,0].set_xlabel('velocity (km/s)')
    
        for row,name,style in zip(self.comp.T,ele,styles):
            axarr[0,1].semilogy(self.interior_mass,row,label=name, **style)
        axarr[0,1].set_xlabel('interior mass (msun)')
        axarr[0,1].set_ylabel('mass fraction')
        axarr[0,1].set_ylim(1e-6,1)
        
        axarr[1,0].semilogy(vga,self.rho_g_cm3)
        axarr[1,0].set_ylabel('rho (g / cm3)')
        axarr[1,0].set_xlabel('velocity (km/s)')

        axarr[1,1].semilogy(self.interior_mass,self.rho_g_cm3)
        axarr[1,1].set_ylabel('rho (g / cm3)')
        axarr[1,1].set_xlabel('interior mass (msun)')

        if thermal:
            axarr[2,0].semilogy(vga, self.T_K)
            axarr[2,0].set_ylabel('T(K)')
            axarr[2,0].set_xlabel('velocity (km/s)')

            axarr[2,1].semilogy(self.interior_mass, 
                                np.cumsum(self.shell_thermal_energy))
            axarr[2,1].set_ylabel('cumulative thermal energy (erg)')
            axarr[2,1].set_xlabel('interior mass (msun)')
        
        for ax in axarr.ravel():
            ax.minorticks_on()
            
        for ax in axarr[:, 0]:
            ax.set_xscale('log')

        if sb:
            sns.despine()

        elenames = [e.repr for e in self.spec]
        elemasses = self.spec_mass.sum(axis=0)
        newlabels = [label + r' $(%.3f M_{\odot})$' % mass for label, mass \
                     in zip(labels, elemasses)]

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.legend(handles, newlabels, loc='upper left', ncol=ncol)

        if show:
            fig.show()
            
        return (fig, axarr)

    def write(self, outfile, padnico=True, mintemp=None):
        # sedona6
        with open(outfile, 'w') as f:
            f.write('1D_sphere SNR\n')

            spec = self.spec
            comp = self.comp
            if padnico and Co56 not in spec:
                spec.append(Co56)
                comp = np.hstack((comp, np.zeros((comp.shape[0], 1))))
            if padnico and Fe56 not in spec:
                spec.append(Fe56)
                comp = np.hstack((comp, np.zeros((comp.shape[0], 1))))
            nspec = len(spec)

            order = zip(*sorted(list(enumerate(spec)), 
                                key=lambda tup: (tup[1].Z, tup[1].A)))[0]
            
            order = np.asarray(order, dtype='i')
            
            spec = [spec[i] for i in order]
            comp = comp.T[order].T
            
            
            f.write('%d %f %f %d\n' % (self.nzones, 
                                       self.velocity(kind='inner')[0] * KM_CM,
                                       self.texp,
                                       nspec))
            
            f.write(' '.join(['%d.%d' % (elem.Z, 
                                         elem.A) for elem in spec]) + '\n')
            v = self.velocity(kind='outer') * KM_CM
            for i in range(self.nzones):

                rho = self.rho_g_cm3[i]
                T = self.T_K[i] 
                if mintemp is not None and T < mintemp:
                    T = mintemp
                comps = comp[i]

                line = ['%e' % ff for ff in [v[i], rho, T] + comps.tolist()]

                f.write(' '.join(line) + '\n')

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
            

class StratifiedAtmosphere(_AtmosphereBase):
    """A simple supernova atmosphere. Composed of one or more Layers."""

    def __init__(self, layers, masses, profile, nzones=100, v_outer=4.e4,
                 texp=86400., thermal_energy=0., thermal_profile=None):

        # state
        self.layers = layers
        self.profile = profile
        self.masses = masses
        self._thermal_energy = thermal_energy
        if thermal_profile is not None:
            self.thermal_energy_profile = thermal_profile
        else:
            self.thermal_energy_profile = Flat(v_outer)
        
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
        self.fracs = np.zeros((self.nzones, len(self.layers)))

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
        for i in range(self.nzones):
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
                prev_mass = sum([self._layermass(ending, l)
                                 for l in range(0, i)])

                # The mass in this zone of this zone's topmost layer.
                remaining_mass = self.masses[ending] - prev_mass

                # The fraction in this zone of this zone's topmost layer.
                self.fracs[i, ending] = remaining_mass / sm

                # The fractions of the other layers:
                for k in range(ending+1, starting)[::-1]:
                    self.fracs[i, k] = self.masses[k] / sm
                self.fracs[i, starting] = 1 - sum(self.fracs[i, ending:starting])
                self._comp[i] = self._abun(self.fracs[i])
                

    @property
    def spec(self):
        return self._spec

    @property
    def interior_mass(self):
        vo = self.velocity(kind='outer')
        return self.profile(vo) * sum(self.masses)
    
    @property
    def interior_thermal_energy(self):
        vo = self.velocity(kind='outer')
        return self._thermal_energy * self.thermal_energy_profile(vo)

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

    def _layermass(self, layerind, i):
        """The mass of layer `layer` in zone `i`."""
        return self.shell_mass[i] * self.fracs[i, layerind]

    def _abun(self, fracs):
        """Compute the composition array (mass fractions of each element, in
        the order of self.spec), of an atmospheric zone composed of
        `fracs` relative fractions of `layers` layers."""

        result = np.zeros(self.nspec)
        for layer, frac in zip(self.layers, fracs):
            for elem in layer.abundances:
                result[self._indexof(elem)] += layer.abundances[elem] * frac
        return result


class Atmosphere(_AtmosphereBase):

    @property
    def nzones(self):
        return len(self.rho_Msun_km3)

    @property
    def v_outer(self):
        return self._v_outer_array[-1]

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
    def comp(self):
        return self._comp

    @property
    def interior_mass(self):
        return np.cumsum(self.rho_Msun_km3 * self.vol_km3)
    
    @property
    def interior_thermal_energy(self):
        return self._th_int

    def velocity(self, kind='average'):
        """Zone radial velocities (km/s).

        Parameters:
        -----------

        kind, str: 'inner', 'outer', and 'average'

        """
        
        if kind == 'outer':
            return self._v_outer_array
        elif kind == 'inner':
            outer = self._v_outer_array
            return np.concatenate(([0.], outer[:-1]))
        else:
            return 0.5 * (self.velocity(kind='inner') + self.velocity(kind='outer'))

    def __init__(self, spec, comp, dens, texp, v_outer_array, th_int):

        # state
        self._spec = spec
        self._rho_Msun_km3 = dens
        self._comp = comp
        self._v_outer_array = v_outer_array
        self._th_int = th_int

        # parameters
        self._texp = texp


class EjectaModelAtmosphere(Atmosphere):

    @classmethod
    def from_ken(cls, ejecta_model_fname):
        # midpoint convention
        # mass fractions

        with open(ejecta_model_fname, 'r') as f:
            labels = f.readline().split()
            spec = map(find_element_by_name, labels[4:])
            _ = f.readline() # skip next line
            data = np.genfromtxt(f)

        time = 10. # seconds, from ken (private communication)
        v_0 = 0.
        m_0 = 0.

        rho_Msun_km3 = []
        v_outer_km_s = []

        for row in data:
            v = row[1] / KM_CM
            m = row[0]
            dm = m - m_0
            shell_vol = 4 / 3. * np.pi * time**3 * (v**3 - v_0**3)
            rho = dm / shell_vol

            rho_Msun_km3.append(rho)
            v_outer_km_s.append(v)

            v_0 = v
            m_0 = m

        rho_Msun_km3 = np.asarray(rho_Msun_km3)
        v_outer_km_s = np.asarray(v_outer_km_s)

        T_K = 10**data[:, 3]
        comp = data[:, 4:]

        return cls(spec, comp, rho_Msun_km3, time, v_outer_km_s, T_K)

    @classmethod
    def from_sedona(cls, ejecta_model_fname):
        with open(ejecta_model_fname, 'r') as f:
            _ = f.readline() # skip first line
            nzones, v_inner_cm_s, texp, _ = f.readline().split()
            nzones = int(nzones)
            v_inner_km_s = float(v_inner_cm_s) / KM_CM
            texp = float(texp)

            if not np.isclose(v_inner_km_s, 0.):
                raise ValueError('Sedona model has v_inner < 0. Currently simple can only '
                                 'handle sedona models with v_inner == 0.')

            spec = [find_element_by_AZ(int(s.split('.')[1]), int(s.split('.')[0]))
                    for s in f.readline().split()]

            # read the rest of the data
            data = np.genfromtxt(f)

        # close the file
        if nzones != len(data):
            raise ValueError('badly formatted sedona model: nzones != number of data rows')

        v_outer_km_s = data[:, 0] / KM_CM
        rho_g_cm3 = data[:, 1]
        T_K = data[:, 2]
        comp = data[:, 3:]

        rho_Msun_km3 = rho_g_cm3 / MSUN_G * (KM_CM)**3

        return cls(spec, comp, rho_Msun_km3, texp, v_outer_km_s, T_K)

    def __init__(self, spec, comp, dens, texp, v_outer_array, T_K):
        # state
        self._spec = spec
        self._rho_Msun_km3 = dens
        self._comp = comp
        self._v_outer_array = v_outer_array
        self._T_K = T_K

        # parameters
        self._texp = texp

    @property
    def shell_thermal_energy(self):
        return self.T_K**4 * self.vol_cm3 * A

    @property
    def interior_thermal_energy(self):
        return np.cumsum(self.shell_thermal_energy)

    @property
    def T_K(self):
        return self._T_K

