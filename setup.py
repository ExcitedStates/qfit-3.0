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
        "cvxpy",
        "pyscipopt>=5.1.1",
        "pandas>=1.2",
        "pyparsing>=2.2.0",
        "tqdm>=4.0.0",
        "rdkit",
    ]

    setup(
        name="qfit",
        use_scm_version=True,
        author="Stephanie A. Wankowicz, Blake Riley, Jessica Flowers, Gydo C.P. van Zundert, Saulo H.P. de Oliveira, and Henry van den Bedem",
        author_email="stephanie@wankowiczlab.com",
        project_urls={"Documentation": "https://github.com/ExcitedStates/qfit-3.0/"},
        package_dir=package_dir,
        packages=packages,
        package_data=package_data,
        ext_modules=ext_modules,
        setup_requires=setup_requires,
        install_requires=install_requires,
        zip_safe=False,
        python_requires=">=3.9",
        entry_points={
            "console_scripts": [
                "qfit_protein = qfit.qfit_protein:main",
                "qfit_residue = qfit.qfit_residue:main",
                "qfit_ligand  = qfit.qfit_ligand:main",
                "qfit_covalent_ligand = qfit.qfit_covalent_ligand:main",
                "qfit_prep_map = qfit.qfit_prep_map:main",
                "qfit_density = qfit.qfit_density:main",
                "qfit_mtz_to_ccp4 = qfit.mtz_to_ccp4:main",
                "edia = qfit.edia:main",
                "remove_altconfs = qfit.remove_altconfs:main",
                "event_map_bdc_scaler = qfit.event_map_bdc_scaler:main",
                "side_chain_remover = qfit.side_chain_remover:main",
                "redistribute_cull_low_occupancies = qfit.redistribute_cull_low_occupancies:main",
                "fix_restraints = qfit.fix_restraints:main",
                "add_non_rotamer_atoms = qfit.add_non_rotamer_atoms:main",
                "remove_duplicates = qfit.remove_duplicates:main",
                "create_rotamer_library = qfit.create_rot_lib:main",
                "calc_chi = qfit.calc_chi:main",
            ]
        },
        scripts=[
            "scripts/post/qfit_final_refine_xray.sh",
            "scripts/post/qfit_final_refine_cryoem.sh",
            "scripts/post/qfit_final_refine_ligand.sh",
            "scripts/post/qfit_final_refine_cryoem_ligand.sh",
            "scripts/post/find_largest_ligand.py",
            "scripts/post/find_altlocs_near_ligand.py",
            "scripts/post/get_lig_chain_res.py",
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
            "scripts/post/find_close_residues.py",
            "scripts/post/get_seq.py",
            "scripts/post/calc_rmsd.py",
            "scripts/post/get_smiles.py",
            "scripts/post/split_multiconformer_ligand.py",
            "scripts/post/compare_rscc_voxel.py",
            "scripts/post/compare_edia.py",
            "scripts/post/create_restraints_file.py",
            "scripts/post/compare_rotamers.py",
        ],
    )


if __name__ == "__main__":
    main()
