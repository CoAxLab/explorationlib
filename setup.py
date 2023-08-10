from setuptools import setup, find_packages
import numpy as np

setup(name='explorationlib',
      version='0.0.1',
      description="Tools to simulate biological exploration.",
      url='',
      author='Erik J. Peterson',
      author_email='erik.exists@gmail.com',
      license='GPL3',
      packages=find_packages(include=['explorationlib', 'explorationlib.*']),
      zip_safe=False)
