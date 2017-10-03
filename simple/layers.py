from elements import *
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import os

__whatami__ = 'Layers of a supernova atmosphere.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

HEGER_COLUMNS = ['grid',
                 'cell_outer_total_mass',
                 'cell_outer_radius',
                 'cell_outer_velocity',
                 'cell_density',
                 'cell_temperature',
                 'cell_pressure',
                 'cell_specific_energy',
                 'cell_specific_entropy',
                 'cell_angular_velocity',
                 'cell_A_bar',
                 'cell_Y_e',
                 'stability',
                 'NETWORK',
                 'neutrons',
                 'H1',
                 'He3',
                 'He4',
                 'C12',
                 'N14',
                 'O16',
                 'Ne20',
                 'Mg24',
                 'Si28',
                 'S32',
                 'Ar36',
                 'Ca40',
                 'Ti44',
                 'Cr48',
                 'Fe52',
                 'Fe54',
                 'Ni56',
                 'Fe56',
                 'Fe58']

class Layer(object):
    """An unmixed layer of a supernova atmosphere. Keeps track of the
    elements in the layer and their relative abundances.
     
    Parameters
    ----------
    abundances: dict, 
      A dictionary mapping Elements to their abundances in the
      layer. Abundances should be given as mass fractions.  If
      abundances do not sum to 1, they will be renormalized. 
    """

    @classmethod
    def from_atmosphere(cls, atmosphere, m1, m2, exclude_elements=[]):
        m_interior = np.concatenate(([0.], atmosphere.interior_mass))

        if m1 < m_interior[0] or m2 > m_interior[-1]:
            raise ValueError('m1 and m2 must be within the atmosphere')

        m1cell = np.searchsorted(m_interior, m1) - 1
        m2cell = np.searchsorted(m_interior, m2) - 1

        mfs = dict()

        if m1cell < 0:
            m1cell = 0
        m1mass = m_interior[m1cell + 1] - m1
        m2mass = m2 - m_interior[m2cell]
        cell_mass = atmosphere.shell_mass

        for element, comp in zip(atmosphere.spec, atmosphere.comp.T):
            if element in exclude_elements:
                continue
            avmf = m2mass * comp[m2cell] + m1mass * comp[m1cell] \
                   + np.sum(comp[m1cell + 1:m2cell] * \
                            cell_mass[m1cell + 1:m2cell])
            avmf /= m2 - m1
            if not np.isclose(avmf, 0., atol=1e-10):
                mfs[element] = avmf
        return cls(mfs)

    @classmethod
    def from_heger(cls, heger_file, m1, m2, exclude_elements=[]):
        data = np.genfromtxt(heger_file, skip_header=1, missing_values='---',
                             filling_values=0., names=HEGER_COLUMNS,
                             usecols=range(len(HEGER_COLUMNS))[1:])
        mfs = dict()
        x = np.concatenate(([0.], data['cell_outer_total_mass']))
        if m1 < x[0] or m2 > x[-1]:
            raise ValueError('m1 and m2 must be within the atmosphere')

        m1cell = np.searchsorted(x, m1) - 1
        m2cell = np.searchsorted(x, m2) - 1
        if m1cell < 0:
            m1cell = 0
        m1mass = x[m1cell + 1] - m1
        m2mass = m2 - x[m2cell]
        cell_mass = x[1:] - x[:-1]
        
        for elname in HEGER_COLUMNS[-19:]:
            y = data[elname]
            avmf =  m2mass * y[m2cell] + m1mass *  y[m1cell] \
                    + np.sum(y[m1cell + 1:m2cell] * \
                             cell_mass[m1cell + 1:m2cell])
            avmf /= m2 - m1
            if not np.isclose(avmf, 0., atol=1e-10):
                mfs[eval(elname)] = avmf
        return cls(mfs)

    def __init__(self, abundances):
        self.normalization = sum(abundances.values())
        self.abundances = {key:abundances[key]/self.normalization
                           for key in abundances}

    def __repr__(self):
        s = 'Layer:\n'
        return s + '\n'.join(['\t%s: %.3e' % (key, self.abundances[key])
                              for key in self.abundances])
        
iron = Layer({Fe54:1.})
nickel = Layer({Ni56:1.})
ime = Layer({Si28:0.53,
             S32:0.32,
             Ca40:0.062,
             Ar36:0.083})
co = Layer({C12:0.5,
            O16:0.5})
he = Layer({He4:1.})
heh = Layer({He4:0.35,
             H2:0.65})

def heger_s15_layers():
    base = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    fname = os.path.join(base, '../data/s15.0')
    
    
    iron = Layer.from_heger(fname, 0, 1e33)
    nickel = Layer.from_heger(fname, 2.8e33, 3.1e33)
    oxy = Layer.from_heger(fname, 3.1e33, 5.6e33)
    he = Layer.from_heger(fname, 6e33, 8e33)
    h = Layer.from_heger(fname, 8e33, 2.1e34)
    return [iron, nickel, oxy, he, h]

