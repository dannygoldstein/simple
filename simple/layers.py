
__whatami__ = 'Layers of a supernova atmosphere.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

from elements import *

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

    def __init__(self, abundances, name=None):
        self.normalization = sum(abundances.values())
        self.abundances = {key:abundances[key]/self.normalization
                           for key in abundances}
        self.name = name
        
    def __repr__(self):
        s = 'Layer:\n'
        return s + '\n'.join(['\t%s: %.3f' % (key, self.abundances[key])
                              for key in self.abundances])
        
iron = Layer({Fe54:1.}, name='54Fe')
nickel = Layer({Ni56:1.}, name='56Ni')
ime = Layer({Si28:0.53,
             S32:0.32,
             Ca40:0.062,
             Ar36:0.083}, name='IME')
co = Layer({C12:0.5,
            O16:0.5}, name='CO')
he = Layer({He4:1.}, name='4He')
heh = Layer({He4:0.35,
             H2:0.65}, name='HHe')

