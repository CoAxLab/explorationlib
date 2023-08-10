from setuptools import setup
import numpy as np

setup(name='explorationlib',
      version='0.0.1',
      description="Tools to simulate biological exploration.",
      url='',
      author='Erik J. Peterson',
      author_email='erik.exists@gmail.com',
      license='GPL3',
      packages=['explorationlib', 'ADMCode', 'ADMCode.snuz', 'ADMCode.snuz.ars', 'ADMCode.snuz.ppo', 'ADMCode.snuz.ppo.agents'],
      zip_safe=False)
