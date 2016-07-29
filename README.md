```python

import profile, simple, diffuse1d, layers

# layers in the atmosphere
l = [layers.iron, layers.nickel, layers.ime,
     layers.co, layers.he, layers.heh]
     
m = [2.] * len(l)  # masses of each layer

# density profile of the atmosphere
densprof = profile.BrokenPowerLaw(-1, -7, 2e4)

# how to handle mixing 
mixer = diffuse1d.DiffusionMixer(200)

# create the atmosphere
atm = simple.StratifiedAtmosphere(l, m, densprof, v_outer=5e4)

# mix it 
mixed_atm = mixer(atm)

mixed_atm.plot()

```

![result](http://astro.berkeley.edu/~dgold/static/figure_1.svg)