'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import os.path
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import numpy as np


def main():
    package_dir = {'': 'src'}
    packages = find_packages('src')
    package_data = {'qfit': [os.path.join('data', '*.npy'), ]}

    ext_modules = [Extension("qfit._extensions",
                             [os.path.join("src", "_extensions.c")],
                             include_dirs=[np.get_include()],),
                   ]
    setup_requires = [
        'setuptools_scm',
    ]
    install_requires = [
        'numpy>=1.17',
        'scipy>=1.00',
        'pyparsing>=2.2.0',
        'tqdm>=4.0.0',
    ]

    setup(name="qfit",
          use_scm_version=True,
          author='Blake Riley, Stephanie A. Wankowicz, Gydo C.P. van Zundert, Saulo H.P. de Oliveira, and Henry van den Bedem',
          author_email='saulo@stanford.edu',
          project_urls={'Documentation': 'https://github.com/ExcitedStates/qfit-3.0/'},
          package_dir=package_dir,
          packages=packages,
          package_data=package_data,
          ext_modules=ext_modules,
          setup_requires=setup_requires,
          install_requires=install_requires,
          zip_safe=False,
          python_requires='>=3.6', 
          entry_points={
              'console_scripts': [
                  'qfit_protein = qfit.qfit_protein:main',
                  'qfit_residue = qfit.qfit_residue:main',
                  'qfit_ligand  = qfit.qfit_ligand:main',
                  'qfit_covalent_ligand = qfit.qfit_covalent_ligand:main',
                  'qfit_segment = qfit.qfit_segment:main',
                  'qfit_prep_map = qfit.qfit_prep_map:main',
                  'qfit_density = qfit.qfit_density:main',
                  'qfit_mtz_to_ccp4 = qfit.mtz_to_ccp4:main',
                  'edia = qfit.edia:main',
                  'relabel = qfit.relabel:main',
                  'remove_altconfs = qfit.remove_altconfs:main',
                  'side_chain_remover = qfit.side_chain_remover:main',
                  'redistribute_cull_low_occupancies = qfit.redistribute_cull_low_occupancies:main',
                  'fix_restraints = qfit.fix_restraints:main',
                  'qfit_ppiDesign = qfit.qfit_ppiDesign:main',
                  'add_non_rotamer_atoms = qfit.add_non_rotamer_atoms:main',
                  'remove_duplicates = qfit.remove_duplicates:main'
              ]},
          scripts=[
              'scripts/post/qfit_final_refine_xray.sh',
              'scripts/post/qfit_final_refine_cryoem.sh',
              'scripts/post/find_largest_ligand.py',
              'scripts/post/find_altlocs_near_ligand.py',
              'scripts/post/qfit_RMSF.py',
              'scripts/post/find_altlocs_near_ligand.py',
              'scripts/post/compare_apo_holo.py',
              'scripts/post/get_metrics.py',
              'scripts/post/b_factor.py',
              'scripts/post/subset_structure_AH.py'
          ],
    )


if __name__ == '__main__':
    main()
