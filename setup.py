from setuptools import setup

setup(name='simple',
      version='0.0.1',
      description='simple supernova atmospheres in python',
      requires=['numpy', 'scipy', 'seaborn', 'matplotlib'],
      install_requires=['numpy', 'scipy', 'seaborn', 'matplotlib'],
      provides=['simple'],
      author='Danny Goldstein',
      author_email='dgold@berkeley.edu',
      license='MIT',
      url='http://github.com/dannygoldstein/simple',
      packages=['simple'])
