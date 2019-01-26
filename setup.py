import os.path

from setuptools import setup
from setuptools.extension import Extension

import numpy as np


def main():

    packages = ['qfit', 'qfit.structure']
    package_data = {'qfit': [os.path.join('data', '*.npy'), ]}

    ext_modules = [Extension("qfit._extensions",
                             [os.path.join("src", "_extensions.c")],
                             include_dirs=[np.get_include()],),
                   ]
    install_requires = [
        'numpy>=1.14',
        'scipy>=1.00',
    ]

    setup(name="qfit",
          version='3.0.0',
          author='Gydo C.P. van Zundert, Saulo H.P. de Oliveira, and Henry van den Bedem',
          author_email='saulo@stanford.edu',
          packages=packages,
          package_data=package_data,
          ext_modules=ext_modules,
          install_requires=install_requires,
          entry_points={
              'console_scripts': [
                  'qfit_protein = qfit.qfit_protein:main',
                  'qfit_residue = qfit.qfit_residue:main',
                  'qfit_segment = qfit.qfit_segment:main',
                  'qfit_prep_map = qfit.qfit_prep_map:main',
                  'qfit_density = qfit.qfit_density:main',
                  'qfit_mtz_to_ccp4 = qfit.mtz_to_ccp4:main',
                  'edia = qfit.edia:main',
                  'side_chain_remover = qfit.side_chain_remover:main',
                  'normalize_occupancies = qfit.normalize_occupancies:main',
              ]
          },)


if __name__ == '__main__':
    main()
