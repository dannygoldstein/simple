from setuptools import setup

setup(name='simple',
      version='0.2.3',
      description='simple supernova atmospheres in python',
      requires=['numpy', 'scipy', 'seaborn', 'matplotlib'],
      install_requires=['numpy', 'scipy', 'seaborn', 'matplotlib'],
      provides=['simple'],
      author='Danny Goldstein',
      author_email='dgold@berkeley.edu',
      license='MIT',
      url='http://github.com/dannygoldstein/simple',
      packages=['simple'],
      package_data={'simple':['data/1.1_5050_simpler.dat','data/s15.0']})
