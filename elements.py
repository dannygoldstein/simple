
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

Fe54 = Element(54, 26, 53.939, '54Fe')
Ni56 = Element(56, 28, 55.940, '56Ni')
Si28 = Element(28, 14, 27.976, '28Si')
S32 = Element(32, 16, 31.972, '32S')
Ca40 = Element(40, 20, 39.962, '40Ca')
Ar36 = Element(36, 18, 35.967, '36Ar')
C12 = Element(12, 6, 12., '12C')
O16 = Element(16, 8, 15.994, '16O')
He4 = Element(4, 2, 4.00260, '4He')
H2 = Element(2, 1, 2.0140, '2H')
