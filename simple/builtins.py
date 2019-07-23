from .layers import *
from .profile import *
from .simple import *

def example_Ia(ke=1e51, nzones=100, v_outer=4e4):
    layers = [iron, nickel, ime]
    masses = [0.1, 0.6, 0.4]
    profile = Exponential(ke, sum(masses))
    return StratifiedAtmosphere(layers, masses, profile, nzones=nzones,
                                v_outer=v_outer)
