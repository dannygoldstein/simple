
__whatami__ = 'Elements of the periodic table.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

class Element(object):
    """An element of the periodic table.
    
    Parameters
    ----------

    A: int,
       The mass number.
       
    Z: int,
       The atomic number.
       
    weight: float,
       The atomic weight, in atomic mass units.
       
    repr: str,
       The string representation of the element, for use in plots and
       the like.
    """
    
    def __init__(self, A, Z, weight, repr):
        self.A = A
        self.Z = Z
        self.weight = weight
        self.repr = repr

    def __repr__(self):
        return '%s: A=%d, Z=%d, weight=%.3f' % (self.repr,
                                                self.A, self.Z,
                                                self.weight)

H1 = Element(1, 1, 1.00727, '1H')
H2 = Element(2, 1, 2.0140, '2H')
He3 = Element(3, 2, 3.0160293, '3He')
He4 = Element(4, 2, 4.00260, '4He')
C12 = Element(12, 6, 12., '12C')
N14 = Element(14, 7, 14.007, '14N')
O16 = Element(16, 8, 15.994, '16O')
Ne20 = Element(20, 10, 20.1797, '20He')
Mg24 = Element(24, 12, 24.3050, '24Mg')
Si28 = Element(28, 14, 27.976, '28Si')
S32 = Element(32, 16, 31.972, '32S')
Ar36 = Element(36, 18, 35.967, '36Ar')
Ca40 = Element(40, 20, 39.962, '40Ca')
Ti44 = Element(44, 22, 43.9596, '44Ti')
Cr48 = Element(48, 24, 47.9540, '48Cr')
Fe52 = Element(52, 26, 51.9481, '52Fe')
Fe54 = Element(54, 26, 53.939, '54Fe')
Ni56 = Element(56, 28, 55.940, '56Ni')
Fe56 = Element(56, 26, 55.9349, '56Fe')
