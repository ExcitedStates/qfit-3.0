# qFit 3.1.0 (Version beta)

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

## Requirements

* Python3.6+
* NumPy
* SciPy
* cvxopt
* IBM ILOG CPLEX Optimization Studio (Community Edition)

Optional requirements used for installation

* git
* conda
* pip

## Installation

If you have access to the `conda` package manager ([Python 3.7 64bit Anaconda 2019.03](https://www.anaconda.com/distribution/)),
installing all dependencies can be done by issuing the following commands:

    conda install -c conda-forge -c ibmdecisionoptimization numpy scipy cvxopt cplex

Instructions using `miniconda` can be found
[here](https://github.com/fraser-lab/holton_scripts/blob/master/qfit_stuff/qfit_install_guide.txt).

You are now all set now to install `qfit`. Installation of `qfit` can be performed by:

    git clone https://github.com/excitedstates/qfit-3.0
    cd qfit-3.0
    pip install .

(NB: `python setup.py install` will not work unless numpy is _already_ installed.)


## Usage examples:

The `qfit` package comes with several command line tools to model alternate
conformers into electron densities. You should select the command line tool that
is most suited for your task. Please, refer below for a few use case examples
to understand which tool is best suited for your needs.

To remove single-conformer model bias, qFit should be used with a composite omit
map. One way of generating such map is using the [Phenix software suite](https://www.phenix-online.org/):

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine`

An example test case (3K0N) can be found in the *example* directory.

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

`/path/to/qfit-3.0/scripts/post_refine_phenix.csh multiconformer_model2.pdb /path/to/3K0N.mtz IOBS SIGIOBS`

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. We plan to remove this dependency in future releases.

### 2. Modelling alternate conformers for a residue of interest


`qfit_residue [MAP_FILE] -l [LABELS] [PDB_FILE] [CHAIN,RESIDUE]`


Using the example 3K0N:


`qfit_residue /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb A,113`


This will produce a parsimonious model containing up to 5 alternate conformers
for residue 113 of chain A of 3K0N.


-------------
### 3. Using a map file in ".ccp4" format as input for qFit

qFit can also use ccp4 map files as input. To model alternate conformers using
this type of map, it is also necessary to provide the resolution of the data,
which can be achieved by using the flag *-r*.


`qfit_residue [MAP_FILE] [PDB_FILE] [RESIDUE,CHAIN] -r [RESOLUTION]`



Using the example 3K0N:


`qfit_residue /path/to/3K0N.ccp4 /path/to/3K0N.pdb A,113 -r 1.39`

-------------

### 4. Deactivate backbone sampling and bond angle sampling to model alternate conformers for a single residue of interest (faster, less precise)


In its default mode, *qfit_residue* and *qfit_protein* samples backbone conformations
using our KGS routine. This can be disabled using the *-bb* flag.


For even faster (and less precise) results, one can also disable the sampling of
the bond angle N-CA-CB, which can be deactivated by means of the *-sa* flag.


Other useful sampling parameters that can be tweaked to make qFit run faster at
the cost of precision:


* Increase step size (in degrees) sampled around each rotamer: *-s* flag (default: 10).
* Decrease rotamer neighborhood sampled: *-rn* flag (default: 80)
* Disable parsimonious selection of the number of conformers output by qFit using the Bayesian
Information Criterion (BIC): *-T* flag.


Using the example 3K0N:

`qfit_residue /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb A,113 -bb -sa -s 20 -T -rn 45`

For a full list of options, run:

`qfit_residue -h`


-------------

### 5. The same sampling parameters used in qfit_residue can be tweaked in qfit_protein:


Using the example 3K0N:


`qfit_protein /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb -bb -sa -s 20 -T -rn 45`

-------------

### 6.  Parallelization:


The *qfit_protein* routine can be executed in parallel and the number of concurrent threads
can be adjusted using the *-p* flag.


Using the example 3K0N, spawning 30 parallel threads:


`qfit_protein /path/to/3K0N.mtz -l 2FOFCWT,PH2FOFCWT /path/to/3K0N.pdb  -p 30`

-------------

### 7. Revisiting the consistent protein segment output by qfit_protein


Depending on the resolution, the default parameters for the identification of
consistent protein segments may prove too strict, leading to the removal of
perfectly valid alternate conformers. We are currently working towards calibration
of the parameters in a resolution-dependent manner.


In the meantime, if you notice that your conformer of interest is present in
*multiconformer_model.pdb*, but was subsequently removed by *qfit_protein*, you
can re-process the initial model with less stringent parameters using the *qfit_segment* routine:


`qfit_segment /path/to/3K0N.mtz  -l 2FOFCWT,PH2FOFCWT /path/to/multiconformer_model.pdb -Ts -f 3`

-------------

### 8. Modeling alternate conformers of a ligand (under development)

To model alternate conformers of ligands, the command line tool *qfit_ligand*
should be used:


`qfit_ligand [MAP_FILE] -l [LABEL] [PDB_FILE] [CHAIN,LIGAND]`


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
