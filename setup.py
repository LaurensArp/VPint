from setuptools import setup

setup(name='VPint',
      version='0.1',
      description='Value propagation-based spatial and spatio-temporal interpolation',
      url='https://github.com/LaurensArp/VPint',
      author='Laurens Arp',
      author_email='l.r.arp@liacs.leidenuniv.nl',
      license='GPL-3.0',
      packages=['VPint', 'utils'],
      install_requires=[
          'numpy',
          'networkx',
      ],
      zip_safe=False)