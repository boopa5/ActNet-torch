from setuptools import setup

setup(
   name='ActNet-torch',
   version='1.0',
   description='PyTorch implementation of ActNet',
   author='Richard Gao',
   author_email='rgao.2003@gmail.com',
   packages=['actnet'],  #same as name
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)