from setuptools import setup

setup(name='MRPinterpolation',
      version='0.1',
      description='Markov reward process-based spatial interpolation',
      url='https://github.com/LaurensArp/MRPinterpolation',
      author='Laurens Arp',
      author_email='l.r.arp@liacs.leidenuniv.nl',
      license='GPL-3.0',
      packages=['MRPinterpolation'],
      install_requires=[
          'numpy',
          'networkx',
      ],
      zip_safe=False)