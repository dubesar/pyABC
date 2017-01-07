# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:37:43 2014

@author: emmanuel
"""

from setuptools import setup, find_packages


setup(install_requires=['numpy', 'scipy', 'pandas', 'cloudpickle',
                "flask", "flask_bootstrap", "bokeh", "redis",
                "dill",
                'gitpython', 'seaborn', 'scikit-learn',
                'matplotlib', 'sqlalchemy'],
     packages=find_packages(exclude=["IPythonNotebooks*", "figures*", "misc*", "pydokuwiki*", "scripts*", "style*", "data*"]),
     author='Emmanuel Klinger, Dennis Rickert',
     author_email='emmanuel.klinger@brain.mpg.de, dennis.rickert@helmholtz-muenchen.de',
     name="ICBayes",
     version="0.1.1",
     license="GPLv3",
     platforms="all",
     url="http:/icbayes.de",
     include_package_data=True,
     description='Massively parallel ABC for Python',
      zip_safe=False,  # not zip safe b/c of Flask templates
      entry_points={
            'console_scripts': [
                  'abc-server = visserver.server:run_app',
            ]
      },
      )
