# qFit 3.2.0

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

qFit is a collection of programs for modeling multi-conformer protein structures.

Electron density maps obtained from high-resolution X-ray diffraction data are a spatial and temporal average of all conformations within the crystal. qFit evaluates an extremely large number of combinations of sidechain conformers, backbone fragments and small-molecule ligands to locally explain the electron density.

If you use this software, please cite:
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)


## Installation (conda recommended)

We recommend using the _conda_ package manager to install _qFit_.

You will need the following tools:

* git
* _conda_ package manager (which you can get by installing [Miniconda3](https://docs.conda.io/en/latest/miniconda.html))

Once these are installed, you can:

1. Create a new conda env & activate it
   ```bash
   conda create --name qfit "python>=3.6"
   conda activate qfit
   ```

1. Install dependencies
   ```bash
   conda install -c anaconda mkl numpy
   conda install -c anaconda -c ibmdecisionoptimization \
                 cvxopt cplex
   ```

1. Clone the qFit source, and install to your conda env
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
* [cvxopt](https://cvxopt.org)
* [IBM ILOG CPLEX Optimization Studio (Community Edition)](https://www.ibm.com/products/ilog-cplex-optimization-studio)

Installation instructions using `pip` can be found in the `docs` folder.

Once dependencies are installed, you can clone the qFit source, and install to your env as above.

(Note: `python setup.py install` will only work if numpy has _already_ been installed.)


## Usage examples

The `qfit` package comes with several command line tools to model alternate
conformers into electron densities. You should select the command line tool that
is most suited for your task. Please, refer below for a few use case examples
to understand which tool is best suited for your needs.

To remove single-conformer model bias, qFit should be used with a composite omit
map. One way of generating such map is using the [Phenix software suite](https://www.phenix-online.org/):

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine`

An example test case (3K0N) can be found in the *example* directory. Additionally, you can find the Cryo-EM example (PDB: 7A4M) and the qFit-ligand example (PDB: 4L2L) in the *example* directory. 


### 1. Recommended settings

To model alternate conformers for all residues in a protein of interest using qFit,
the following command should be used:

`qfit_protein [MAP_FILE] -l [LABELS] [PDB_FILE]`

This command will produce a multiconformer model that spans the entirety of the
input target protein. You will encounter two output files of interest in the
directory where this command was run: *multiconformer_model.pdb* and
*multiconformer_model2.pdb*.

If you wish to specify a different directory for the output, this can be done
using the flag *-d*.

The *multiconformer_model.pdb* contains the output of running the qfit_residue
routine for every protein residue. This file may contain up to five alternate
conformers per residue. However, some of these conformers may be spurious as
alternate conformers of neighboring residues are not consistent with regards to
each other.

After calculating the conformers described in *multiconformer_model.pdb*,
qfit_protein identifies consistent protein segments, where some of the spurious,
overlapping conformers, are discarded. The final model, with the consistent
multiconformer model is *multiconformer_model2.pdb*. This file should then
be used as input to the refinement script provided in *./scripts*.   

By default, qFit expects the labels FWT,PHWT to be present in the input map.
Different labels can be set accordingly using the flag *-l*.

Using the example 3K0N:

`qfit_protein /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb`

After *multiconformer_model2.pdb* has been generated, refine this model using:

`qfit_final_refine_xray.sh multiconformer_model2.pdb /path/to/3K0N.mtz`

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. 


### 2. Modelling alternate conformers for a residue of interest

`qfit_residue [MAP_FILE] -l [LABELS] [PDB_FILE] [CHAIN,RESIDUE]`

Using the example 3K0N:

`qfit_residue /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb A,113`

This will produce a parsimonious model containing up to 5 alternate conformers
for residue 113 of chain A of 3K0N.


### 3. Using a map file in ".ccp4" format as input for qFit

qFit can also use ccp4 map files as input. To model alternate conformers using
this type of map, it is also necessary to provide the resolution of the data,
which can be achieved by using the flag *-r*.

`qfit_protein [MAP_FILE] [PDB_FILE] -r [RESOLUTION]`

Using the example 3K0N:

`qfit_residue /path/to/3K0N.ccp4 /path/to/3K0N.pdb -r 1.39`

For Cyro-EM ccp4 maps, you can use the example from the Apoferritin Chain A (PDB:7A4M)

`qfit_protein /path/to/apoF_chainA.ccp4 /path/to/apoF_chainA.pdb -r 1.22 -z electron`

After *multiconformer_model2.pdb* has been generated, refine this model using:
`qfit_final_refine_cryoem.sh multiconformer_model2.pdb /path/to/apoF_chainA.ccp4`

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. 


### 4. Deactivate backbone sampling and bond angle sampling to model alternate conformers for a single residue of interest (faster, less precise)

In its default mode, *qfit_residue* and *qfit_protein* samples backbone conformations
using our KGS routine. This can be disabled using the *--no-backbone* flag.

For even faster (and less precise) results, one can also disable the sampling of
the bond angle Cα-Cβ-Cγ, which can be deactivated by means of the *--no-sample-angle* flag.

Other useful sampling parameters that can be tweaked to make qFit run faster at
the cost of precision:

* Increase step size (in degrees) of sampling around each rotamer: *-s* flag (default: 10)
* Decrease range/neighborhood of sampling about preferred rotamers: *-rn* flag (default: 60)
* Disable parsimonious selection of the number of conformers output by qFit using the Bayesian Information Criterion (BIC): *--no-threshold-selection* flag.

Using the example 3K0N:

`qfit_residue /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb A,113 --no-backbone --no-sample-angle -s 20 -rn 45 --no-threshold-selection`

For a full list of options, run:

`qfit_residue -h`


### 5. The same sampling parameters used in qfit_residue can be tweaked in qfit_protein:

Using the example 3K0N:

`qfit_protein /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb --no-backbone --no-sample-angle -s 20 -rn 45 --no-threshold-selection`


### 6.  Parallelization:

The *qfit_protein* program can be executed in parallel and the number of concurrent processes
can be adjusted using the *-p* flag.

Using the example 3K0N, spawning 30 parallel processes:

`qfit_protein /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb -p 30`


### 7. Revisiting the consistent protein segment output by qfit_protein

Depending on the resolution, the default parameters for the identification of
consistent protein segments may prove too strict, leading to the removal of
perfectly valid alternate conformers. We are currently working towards calibration
of the parameters in a resolution-dependent manner.

In the meantime, if you notice that your conformer of interest is present in
*multiconformer_model.pdb*, but was subsequently removed by *qfit_protein*, you
can re-process the initial model with less stringent parameters using the *qfit_segment* program:

`qfit_segment /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/multiconformer_model.pdb --no-segment-threshold-selection -f 3`


### 8. Modeling alternate conformers of a ligand

To model alternate conformers of ligands, the command line tool *qfit_ligand*
should be used:

`qfit_ligand [MAP_FILE] -l [LABEL] [PDB_FILE] [CHAIN,LIGAND]`

Using the example 4L2L:

`qfit_ligand /path/to/4l2l.mtz -l 2FOFCWT,PH2FOFCWT /path/to/4l2l.pdb A,702`

We then recommend re-refining the output of qFit ligand along with the protein using: 
`qfit_final_refine_xray.sh 4l2l_qFit_ligand.pdb /path/to/4L2L.mtz`

Where *LIGAND* corresponds to the numeric identifier of the ligand on the PDB
(aka res. number).

## License

The code is licensed under the MIT licence (see `LICENSE`).

Several modules were taken from the `pymmlib` package, originally licensed
under the Artistic License 2.0. See the `licenses` directory for a copy of the
original source code and its full license.

The `elements.py` is licensed under MIT, Copyright (c) 2005-2015, Christoph
Gohlke. See file header.

The `Xpleo` software and `LoopTK` package have been major inspirations for the inverse kinematics
functionality.

[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"
