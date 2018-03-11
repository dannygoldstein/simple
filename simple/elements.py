
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
Ne20 = Element(20, 10, 20.1797, '20Ne')
Na22 = Element(22, 11, 21.9944, '22Na')
Na23 = Element(23, 11, 22.9898, '23Na')
Mg24 = Element(24, 12, 24.3050, '24Mg')
Si28 = Element(28, 14, 27.976, '28Si')
S32 = Element(32, 16, 31.972, '32S')
Ar36 = Element(36, 18, 35.967, '36Ar')
Ca40 = Element(40, 20, 39.962, '40Ca')
Ti44 = Element(44, 22, 43.9596, '44Ti')
Cr48 = Element(48, 24, 47.9540, '48Cr')
Cr60 = Element(60, 24, 59.9501, '60Cr')
Fe52 = Element(52, 26, 51.9481, '52Fe')
Fe54 = Element(54, 26, 53.939, '54Fe')
Ni56 = Element(56, 28, 55.940, '56Ni')
Fe56 = Element(56, 26, 55.9349, '56Fe')
Fe57 = Element(57, 26, 56.9353, '57Fe')
Fe58 = Element(58, 26, 57.9332, '58Fe')
Co56 = Element(56, 27, 55.9398, '56Co')
Ni57 = Element(57, 28, 56.9397, '57Ni')
Ni58 = Element(58, 28, 57.9353, '58Ni')
Ni59 = Element(59, 28, 58.9343, '59Ni')
Ni60 = Element(60, 28, 59.9307, '60Ni')
Ni61 = Element(61, 28, 60.9310, '61Ni')
Ni62 = Element(62, 28, 61.9283, '62Ni')

_elements = [H1, H2, He3, He4, C12, N14, O16, Ne20, Na22, Na23, Mg24,
             Si28, S32, Ar36, Ca40, Ti44, Cr48, Cr60, Fe52, Fe54,
             Ni56, Fe56, Fe57, Fe58, Ni57, Ni58, Ni59, Ni60, Ni61,
             Ni62, Co56]



def find_element_by_AZ(A, Z):
    """A simple look up function for the elements"""
    for element in _elements:
        if element.A == A and element.Z == Z:
            return element
    raise ValueError('Element not found')


def find_element_by_name(name):
    """Accepts names case-insensitively of the form (weight)(element abbrev)
    eg 56ni or element abbrev)(weight) eg Ni56"""

    name = name.lower()
    if name[0].isalpha():
        # form ni56
        if name[1].isalpha():
            firstnum = 2
        else:
            firstnum = 1
        name = name[firstnum:] + name[:firstnum]
    for element in _elements:
        if element.repr.lower() == name:
            return element
    raise ValueError('element not found')
