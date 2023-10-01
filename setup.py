# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:18:15 2023

@author: Gunay Dogan, Eve Fleisig
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='VEMOS',
     version='1.1.3',
     author="Gunay Dogan, Eve Fleisig",
     author_email="gunay.dogan@nist.gov",
     description="Visual Explorer for Metrics of Similarity",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="",
     packages=setuptools.find_packages("."),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
     install_requires=[
	    "PyQt5",
         "numpy>=1.15",
         "scipy>=1.2",
         "matplotlib>=1.4.3",
         "scikit-learn>=0.20",
         "scikit-image>=0.16"]
 )
