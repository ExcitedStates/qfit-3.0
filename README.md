# qFit

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

qFit is a collection of programs for modeling multi-conformer protein structures. 

Electron density maps obtained from high-resolution X-ray diffraction data are a spatial and temporal average of all conformations within the crystal. qFit evaluates an extremely large number of combinations of sidechain conformers, backbone fragments and small-molecule ligands to locally explain the electron density.

If you use this software, please cite: 
- [Wankowicz SA, Ravikumar A, Sharma S, Riley BT, Raju A, Hogan DW, van den Bedem H, Keedy DA, & Fraser JS. Uncovering Protein Ensembles: Automated Multiconformer Model Building for X-ray Crystallography and Cryo-EM. bioRxiv. (2023).](https://www.biorxiv.org/content/10.1101/2023.06.28.546963v2.abstract)
- [Riley BT, Wankowicz SA, et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)


## Installation

You will need the following tools:

* git
* pip

Once these are installed, you can:

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
1. Clone the latest release of the qFit source and install it using pip
   ```bash
   git clone https://github.com/ExcitedStates/qfit-3.0.git
   cd qfit-3.0
   pip install .
   ```

1. You're now ready to run qFit programs! See [usage examples](#sec:usage-examples) below for some examples.


### Advanced

If you prefer to manage your environments using other methods, qFit has the following prerequisites:

* [Python 3.6+](https://python.org)
* [numpy](https://numpy.org)
* [scipy](https://scipy.org)
* [cctbx](https://github.com/cctbx/cctbx_project)
* [cvxpy](https://www.cvxpy.org)

Once dependencies are installed, you can clone the qFit source, and install to your env as above.


## Contributing

qFit uses [Black](https://github.com/psf/black) to format its code and provides a git hook to verify that code is properly formatted before allowing you to commit.

Before creating a commit, you will have to perform two actions:
1. Install Black, either through a package manager or by running `python3 -m pip install --user black`
2. Run `git config core.hooksPath .githooks/` to use the provided pre-commit hook

## Usage examples

The `qfit` package comes with several command line tools to model alternate
conformers into electron densities. You should select the command line tool that
is most suited for your task. Please refer below for a basic usage example. More specialized and advanced use case examples
are shown in [TUTORIAL](example/README.md) in the [example](example/) directory.

To remove single-conformer model bias, qFit should be used with a composite omit
map. One way of generating such map is using the [Phenix software suite](https://www.phenix-online.org/):

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine`

An example test case (3K0N) can be found in the [qfit protein example](example/qfit_protein_example/) directory. Additionally, you can find the Cryo-EM example (PDB: 7A4M) and the qFit-ligand example (PDB: 4MS6) in the *example* directory. 


### Recommended settings

To model alternate conformers for all residues in a *X-ray crystallography* model using qFit,
the following command should be used:

`qfit_protein [COMPOSITE_OMIT_MAP_FILE] -l [LABELS] [PDB_FILE]`

This command will produce a multiconformer model that spans the entirety of the
input target protein. The final model, with consistent labeling of multiple conformers
is output into *multiconformer_model2.pdb*. This file should then
be used as input to the post-qFit refinement script provided in [scripts](scripts/post) folder.

If you wish to specify a different directory for the output, this can be done
using the flag *-d*.
 
By default, qFit expects the labels FWT,PHWT to be present in the input map.
Different labels can be set accordingly using the flag *-l*.

Using the example 3K0N:

`qfit_protein example/qfit_protein_example/3k0n_map.mtz -l 2FOFCWT,PH2FOFCWT example/qfit_protein_example/3k0n_refine.pdb`

After *multiconformer_model2.pdb* has been generated, refine this model using:

`qfit_final_refine_xray.sh example/qfit_protein_example/3k0n_structure_factors.mtz example/qfit_protein_example/multiconformer_model2.pdb`

Additionally, the qFit_occupancy.params file must exist in the folder.

(A pre-generated multiconformer_model2.pdb file is available in the [qfit protein example](example/qfit_protein_example/) folder)

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. This script is currently written to work with version Phenix 1.20.


To model alternate conformers for all residues in a *Cryo-EM* model using qFit,
the following command should be used:

`qfit_protein [MAP_FILE] -r [RES] [PDB_FILE] -em`

After *multiconformer_model2.pdb* has been generated, refine this model using:

`qfit_final_refine_cryoEM.sh example/qfit_protein_example/em_map.ccp4 example/qfit_protein_example/multiconformer_model2.pdb example/qfit_protein_example/input_pdb_file.pdb`

More advanced features of qFit (modeling single residue, more advanced options, and further explainations) are explained in [TUTORIAL](example/TUTORIAL.md).


## License

The code is licensed under the MIT licence (see `LICENSE`).

Several modules were taken from the `pymmlib` package, originally licensed
under the Artistic License 2.0. See the `licenses` directory for a copy of the
original source code and its full license.

The `elements.py` is licensed under MIT, Copyright (c) 2005-2015, Christoph
Gohlke. See file header.

The `Xpleo` software and `LoopTK` package have been major inspirations for the inverse kinematics
functionality.
