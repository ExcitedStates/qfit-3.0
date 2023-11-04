import os.path
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import numpy as np


def main():
    package_dir = {"": "src"}
    packages = find_packages("src")
    package_data = {
        "qfit": [
            os.path.join("data", "*.npy"),
        ]
    }

    ext_modules = [
        Extension(
            "qfit._extensions",
            [os.path.join("src", "_extensions.c")],
            include_dirs=[np.get_include()],
        ),
    ]
    setup_requires = [
        "setuptools_scm",
    ]
    install_requires = [
        "numpy>=1.20,<2",
        "scipy>=1.0",
        "pandas>=1.2",
        "pyparsing>=2.2.0",
        "tqdm>=4.0.0",
        "molmass",
    ]

    setup(
        name="qfit",
        use_scm_version=True,
        author="Blake Riley, Stephanie A. Wankowicz, Gydo C.P. van Zundert, Saulo H.P. de Oliveira, and Henry van den Bedem",
        author_email="saulo@stanford.edu",
        project_urls={"Documentation": "https://github.com/ExcitedStates/qfit-3.0/"},
        package_dir=package_dir,
        packages=packages,
        package_data=package_data,
        ext_modules=ext_modules,
        setup_requires=setup_requires,
        install_requires=install_requires,
        extra_dependencies={
            "cplex": ["cvxopt", "cplex"],
            "osqp": ["osqp", "miosqp @ git+https://github.com/osqp/miosqp.git@ac672338b0593d865dd15b7a76434f25e24244a9#egg=miosqp"],
        },
        zip_safe=False,
        python_requires=">=3.9",
        entry_points={
            "console_scripts": [
                "qfit_protein = qfit.command_line.qfit_protein:main",
                "qfit_ligand  = qfit.command_line.qfit_ligand:main",
                "qfit_covalent_ligand = qfit.command_line.qfit_covalent_ligand:main",
                "qfit_prep_map = qfit.command_line.qfit_prep_map:main",
                "qfit_density = qfit.command_line.qfit_density:main",
                "qfit_mtz_to_ccp4 = qfit.command_line.mtz_to_ccp4:main",
                "edia = qfit.command_line.edia:main",
                "remove_altconfs = qfit.command_line.remove_altconfs:main",
                "redistribute_cull_low_occupancies = qfit.command_line.redistribute_cull_low_occupancies:main",
                "fix_restraints = qfit.command_line.fix_restraints:main",
                "add_non_rotamer_atoms = qfit.command_line.add_non_rotamer_atoms:main",
                "remove_duplicates = qfit.command_line.remove_duplicates:main",
                "create_rotamer_library = qfit.command_line.create_rot_lib:main",
            ]
        },
        scripts=[
            "scripts/post/qfit_final_refine_xray.sh",
            "scripts/post/qfit_final_refine_cryoem.sh",
            "scripts/post/find_largest_ligand.py",
            "scripts/post/find_altlocs_near_ligand.py",
            "scripts/post/qfit_RMSF.py",
            "scripts/post/find_altlocs_near_ligand.py",
            "scripts/post/b_factor.py",
            "scripts/post/subset_structure_AH.py",
            "scripts/post/alpha_carbon_rmsd.py",
            "scripts/post/relabel_chain.py",
            "scripts/post/water_stats.py",
            "scripts/post/water_clash.py",
            "scripts/post/lig_occ.py",
            "scripts/post/calc_OP.py",
            "scripts/post/make_methyl_df.py",
            "scripts/post/get_seq.py",
        ],
    )


if __name__ == "__main__":
    main()
