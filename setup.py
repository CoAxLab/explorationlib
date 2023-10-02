from setuptools import setup, find_packages
import numpy as np

setup(name='explorationlib',
      version='0.0.1',
      description="Tools to simulate biological exploration.",
      url='',
      author='Erik J. Peterson',
      author_email='erik.exists@gmail.com',
      license='GPL3',
      packages=['explorationlib', 'ADMCode'],
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'numba', 'future'],
      include_dirs = [np.get_include()],
      zip_safe=False)
