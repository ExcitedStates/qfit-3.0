# qFit 2024v3


![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

qFit is a collection of programs for modeling multiconformer protein structures. 

Electron density maps obtained from high-resolution X-ray diffraction data are a spatial and temporal average of all conformations within the crystal. 
qFit evaluates an extremely large number of combinations of sidechain conformers, backbone fragments, and small-molecule ligand conformations to locally explain the electron density.


If you use this software, please cite: 
- [Wankowicz SA, Ravikumar A, Sharma S, Riley BT, Raju A, Hogan DW, van den Bedem H, Keedy DA, & Fraser JS. Uncovering Protein Ensembles: Automated Multiconformer Model Building for X-ray Crystallography and Cryo-EM. eLife. (2024)](https://doi.org/10.7554/eLife.90606.3)
- [Riley BT, Wankowicz SA, de Oliveira SHP, van Zundert GCP, Hogan DW, Fraser JS, Keedy DA, van den Bedem H. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [Keedy DA, Fraser JS, & van den Bedem H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

qFit-ligand:
- [Flowers J, Echols N, Correy G, Jaishankar P, Togo T, Renslo A, van den Bedem H, Fraser J, Wankowicz SA. (2024) Expanding Automated Multiconformer Ligand Modeling to Macrocycles and Fragments. bioRxiv.](https://www.biorxiv.org/content/10.1101/2024.09.20.613996)
- [van Zundert GCP, Hudson BM, de Oliveira SHP, Keedy DA, Fonseca R, Heliou A, Suresh P, Borrelli K, Day T, Fraser JS, van den Bedem H. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)

As this software relies on CVXPY, please also cite:
- [Agrawal, Verschueren, Diamond, & Boyd. A Rewriting System for Convex Optimization Problems. Journal of Control and Decision. (2018).](https://arxiv.org/abs/1709.04494)
- [Diamond & Boyd. CVXPY: A Python-Embedded Modeling Language for Convex Optimization. Journal of Machine Learning Research. (2016)](https://www.jmlr.org/papers/volume17/15-408/15-408.pdf)


## Installation (conda recommended)

We recommend using the _conda_ package manager to install _qFit_.

1. Clone the latest release of the qFit source, and install to your conda env

   git clone -b main https://github.com/ExcitedStates/qfit-3.0.git
   
   cd qfit-3.0
   
3. Create the Conda environment using the downloaded file:

   mamba env create -f environment.yml

4. After creating the Conda environment, activate it:

   conda activate qfit

5. If you installing on M1 Mac:
   
     conda activate qfit; conda env config vars set CONDA_SUBDIR=osx-64; conda deactivate
     conda activate qfit

6. Install qFit

   pip install .
   

### Advanced

If you prefer to manage your environments using other methods, qFit has the following prerequisites:

* [Python 3.6+](https://python.org)
* [numpy](https://numpy.org)
* [scipy](https://scipy.org)
* [cvxpy](https://www.cvxpy.org)

Once dependencies are installed, you can clone the qFit source, and install to your env as above.


## Usage examples

The `qfit` package comes with several command line tools to model alternate
conformers into electron densities. You should select the command line tool that
is most suited for your task. Please refer below for a basic usage example. More specialized and advanced use case examples
are shown in [example](example/README.md) directory.

To remove single-conformer model bias, qFit should be used with a composite omit
map. One way of generating such a map is using the [Phenix software suite](https://www.phenix-online.org/):

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine`

An example test case (PDB: 1G8A) can be found in the [qFit protein example](example/qfit_protein_example/) directory. Additionally, you can find the [cryo-EM qFit protein example](example/qfit_cryoem_example/) (PDB: 7A4M) 
and the [qFit ligand example](example/qfit_ligand_example/) (PDB: 4MS6) in the [example](example/README.md) directory. 


### Recommended settings

To model alternate conformers for all residues in an *X-ray crystallography* model using qFit,
the following command should be used:

`qfit_protein [COMPOSITE_OMIT_MAP_FILE] -l [LABELS] [PDB_FILE] -p [# OF THREADS]`

This command will produce a multiconformer model that spans the entirety of the
input target protein. The final model, with consistent labeling of multiple conformers,
is output into *multiconformer_model2.pdb*. This file should then
be used as input to the post-qFit refinement script provided in the [scripts](scripts/post) directory. 

qFit can be run on a single thread, but speeds up significantly with multiple threads. To do this, use the *-p* flag.

If you wish to specify a different directory for the output, this can be done
using the flag *-d*.
 
By default, qFit expects the labels FWT,PHWT to be present in the input map.
Different labels can be set accordingly using the flag *-l*.

Using the example 18GA:

`qfit_protein example/qfit_protein_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT example/qfit_protein_example/1G8A_refine.pdb`

After *multiconformer_model2.pdb* has been generated, refine this model using:

`qfit_final_refine_xray.sh example/qfit_protein_example/18GA.mtz example/qfit_protein_example/multiconformer_model2.pdb`

Additionally, the qFit_occupancy.params file must exist in the directory (this is an output of qFit protein).

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. This script is currently written to work with version Phenix 1.21.

To model alternate conformers for all residues in a *cryo-EM* model using qFit,
the following command should be used:

`qfit_protein [MAP_FILE] -r [RES] [PDB_FILE] -em`
`qfit_protein example/qfit_cryoem_example/7A4M_box.ccp4 -r 1.7 example/qfit_cryoem_example/7A4M_box.pdb`

After *multiconformer_model2.pdb* has been generated, refine this model using:

`qfit_final_refine_cryoEM.sh example/qfit_cryoem_example/7A4M_box.ccp4 example/qfit_cryoem_example/multiconformer_model2.pdb example/qfit_cryoem_example/7A4M_box.pdb`

More advanced features of qFit (modeling single residue, more advanced options, and further explainations) are explained in the [example](example/README.md) directory.

To model alternate conformations of ligands using qFit, we recommend generating a composite omit map excluding bulk solvent with the following command:

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine exclude_bulk_solvent=True`

qFit-ligand can be executed the following command:

`qfit_ligand [COMPOSITE_OMIT_MAP_FILE] [PDB_FILE] -l [LABELS] [SELECTION] -sm [SMILES]`

This command facilitates the incorporation of alternate ligand conformations into your protein model. The results are outputted to two files: *multiconformer_ligand_bound_with_protein.pdb*, which is the multiconformer model of the protein-ligand complex, and *multiconformer_ligand_only.pdb*, which is the multiconformer model of the ligand alone. 

After running qFit-ligand, it is recommended to perform a final refinement using the script found in [scripts](scripts/post). Run this in the same directory as your models.

If you wish to specify the number of ligand conformers for qFit to sample, use the flag `-nc [NUM_CONFS]`. The default number is set to 10,000. 

Using the example 4MS6:

`qfit_ligand example/qfit_ligand_example/4ms6_composit_map.mtz example/qfit_ligand_example/4ms6.pdb -l 2FOFCWT,PH2FOFCWT A,702 -sm 'C1C[C@H](NC1)C(=O)CCC(=O)N2CCC[C@H]2C(=O)O' -nc 10000`

To refine *multiconformer_ligand_bound_with_protein.pdb*, use the following command

`qfit_final_refine_ligand.sh 4ms6.mtz`


## Contributing

qFit uses [Black](https://github.com/psf/black) to format its code and provides a git hook to verify that code is properly formatted before allowing you to commit.

Before creating a commit, you will have to perform two actions:
1. Install Black, either through a package manager or by running `python3 -m pip install --user black`
2. Run `git config core.hooksPath .githooks/` to use the provided pre-commit hook

## License

The code is licensed under the MIT licence (see `LICENSE`).

Several modules were taken from the `pymmlib` package, originally licensed
under the Artistic License 2.0. See the `licenses` directory for a copy of the
original source code and its full license.

The `elements.py` is licensed under MIT, Copyright (c) 2005-2015, Christoph
Gohlke. See file header.

The `Xpleo` software and `LoopTK` package have been major inspirations for the inverse kinematics
functionality.
