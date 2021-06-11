from setuptools import setup

setup(name='VPint',
      version='0.2',
      description='Value propagation-based spatial and spatio-temporal interpolation',
      url='https://github.com/LaurensArp/VPint',
      author='Laurens Arp',
      author_email='l.r.arp@liacs.leidenuniv.nl',
      license='GPL-3.0',
      packages=['VPint'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)