from setuptools import setup, find_packages

setup(name='VPint',
      version='0.2.4',
      description='Value propagation-based spatial and spatio-temporal interpolation',
      url='https://github.com/LaurensArp/VPint',
      author='Laurens Arp',
      author_email='l.r.arp@liacs.leidenuniv.nl',
      license='GPL-3.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
      ],
      include_package_data=True,
      zip_safe=False)