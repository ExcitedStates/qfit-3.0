# qFit 3.0.0 (Version alpha)

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

If you have access to the `conda` package manager, installing all dependencies
is straightforward

    conda install -c conda-forge -c ibmdecisionoptimization numpy scipy cvxopt cplex

If you prefer the more traditional `pip` tool, the requirements can be installed
as follows

    pip install numpy scipy cvxopt

Next, obtain a copy of CPLEX, the Community Edition will do, by registering and
downloading it from the [IBM website][1]. After installing and unpacking the
software, install the CPLEX Python interface

    cd <CPLEX_ROOT>/cplex/python/3.6/x86_64_<PLATFORM>
    python setup.py install

where `<CPLEX_ROOT>` is the directory where you installed CPLEX, and `<PLATFORM>` is
a platform dependent string, such as `linux` for Linux systems and `osx` for
macOSX.

You are now all set now to install `qfit`. Installation of `qfit` is
as simple as

    git clone https://github.com/excitedstates/qfit-3.0
    cd qfit-3.0
    python setup.py install


## Usage examples:

The `qfit` package comes with several command line tools to model alternate
conformers into electron densities. You should select the command line tool that
is most suited for your task. Please, refer below for a few use case examples
to understand which tool is best suited for your needs.

An example test case (3K0N) can be found in the *example* directory.


### 1. Modelling alternate conformers for a residue of interest (faster, not as precise)


`qfit_residue [MAP_FILE] [PDB_FILE] [RESIDUE,CHAIN]`


Using the example 3K0N:


`qfit_residue /path/to/3K0N.mtz /path/to/3K0N.pdb A,113`


This will produce a parsimonious model containing up to 5 alternate conformers
for residue 113 of chain A of 3K0N.


-------------
### 2. Using a map file in ".ccp4" format as input for qFit

qFit can also use ccp4 map files as input. To model alternate conformers using
this type of map, it is also necessary to provide the resolution of the data,
which can be achieved by using the flag *-r*.


`qfit_residue [MAP_FILE] [PDB_FILE] [RESIDUE,CHAIN] -r [RESOLUTION]`



Using the example 3K0N:


`qfit_residue /path/to/3K0N.ccp4 /path/to/3K0N.pdb A,113 -r 1.39`

-------------

### 3. Activate backbone sampling and bond angle sampling to model alternate conformers for a single residue of interest (slower, more precise)


In its default mode, *qfit_residue* only samples side chain conformations.
 While this is inherently faster, to take full advantage of qfit modeling
 capabilities, we recommend enabling backbone
sampling using our KGS routine. This can be achieved using the *-bb* flag.


For optimal results, we also recommend sampling the bond angle N-CA-CB, which can
be activated by means of the *-sa* flag.


Other useful sampling parameters that can be tweaked:


* Step size (in degrees) sampled around each rotamer: *-s* flag (default: 6)
* Rotamer neighborhood sampled: *-rn* flag (default: 40)
* Parsimonious selection of the number of conformers output by qFit using the Bayesian
Information Criterion (BIC): *-T* flag.


Using the example 3K0N:


`qfit_residue /path/to/3K0N.mtz /path/to/3K0N.pdb A,113 -bb -sa -s 5 -T -rn 45`


For a full list of options, run:

`qfit_residue -h`

-------------

### 4. Modeling alternate conformers for all residues in a protein of interest


`qfit_protein [MAP_FILE] [PDB_FILE]`


This command will produce a multiconformer model that spans the entirety of your
target protein, as described in the PDB file. You will encounter two output files
of interest in the directory where this command was run: *multiconformer_model.pdb*
and *multiconformer_model2.pdb*.


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
multiconformer model is *multiconformer_model2.pdb*.


Using the example 3K0N:


`qfit_protein /path/to/3K0N.mtz /path/to/3K0N.pdb`

-------------

### 5. The same sampling parameters used in qfit_residue can be tweaked in qfit_protein:


Using the example 3K0N:


`qfit_protein /path/to/3K0N.mtz /path/to/3K0N.pdb -bb -sa -s 5 -T -rn 45`

-------------

### 6.  Parallelization:


The *qfit_protein* routine can be executed in parallel and the number of concurrent threads
can be adjusted using the *-p* flag.


Using the example 3K0N, spawning 30 parallel threads:


`qfit_protein /path/to/3K0N.mtz /path/to/3K0N.pdb -bb -sa -s 5 -T -rn 45 -p 30`

-------------

### 7. Revisiting the consistent protein segment output by qfit_protein


Depending on the resolution, the default parameters for the identification of
consistent protein segments may prove too strict, leading to the removal of
perfectly valid alternate conformers. We are currently working towards calibration
of the parameters in a resolution-dependent manner.


In the meantime, if you notice that your conformer of interest is present in
*multiconformer_model.pdb*, but was subsequently removed by *qfit_protein*, you
can re-process the initial model with less stringent parameters using the *qfit_segment* routine:


`qfit_segment /path/to/3K0N.mtz /path/to/multiconformer_model.pdb -Ts -T 14 -f 3`

-------------

### 8. Modeling alternate conformers of a ligand


To model alternate conformers of ligands, the command line tool *qfit_ligand*
should be used:


`qfit_ligad [MAP_FILE] [PDB_FILE] [LIGAND,CHAIN]`


Where *LIGAND* corresponds to the numeric identifier of the ligand on the PDB
(aka res. number).

## License

The code is licensed under the MIT licence (see `LICENSE`).

Several modules were taken from the `pymmlib` package, originally licensed
under the Artistic License 2.0. See the `licenses` directory for a copy of the
original source code and its full license.

The `elements.py` is licensed under MIT, Copyright (c) 2005-2015, Christoph
Gohlke. See file header.

The `LoopTK` package has been a major inspiration for the inverse kinematics
functionality.

[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"
