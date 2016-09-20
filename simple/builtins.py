from layers import *
from profile import *
from simple import *

def example_Ia():
    layers = [iron, nickel, ime]
    masses = [0.1, 0.6, 0.4]
    profile = Exponential(1e51)
    return StratifiedAtmosphere(layers, masses, profile)
