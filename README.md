# qFit-ligand

## Requirements

* Python3.6+
* NumPy
* SciPy
* cvxopt
* IBM ILOG CPLEX Optimization Studio (Community Edition)

Optional for MTZ to CCP4 file conversion

* CCTBX or Phenix

Optional requirements used for installation

* pip
* conda
* git


## Installation

If you have access to the `conda` package manager, installing all dependencies
is straightforward

    conda install -c conda-forge numpy scipy cvxopt
    conda install -c ibmdecisionoptimization cplex

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

You are now all set now to install `qfit_ligand`. Installation of `qfit_ligand` is
as simple as

    git clone https://github.com/excitedstates/qfit_ligand
    cd qfit_ligand
    python setup.py install


## Usage

qfit\_protein
qfit\_residue
qfit\_ligand

The main program that comes with installing `qfit_ligand` is the eponymously named
`qfit_ligand` command line tool. It has two calling interfaces

    qfit_ligand <CCP4-map> <resolution> <PDB-of-ligand> -r <PDB-of-receptor>
    qfit_ligand <CCP4-map> <resolution> <PDB-of-ligand-and-receptor> --selection <chain>,<resi>

where `<CCP4-map>` is a 2mFo-DFc electron density map in CCP4 format, and
`<resolution>` is its corresponding resolution in angstrom. In the first
command,`<PDB-of-ligand>` is a PDB file containing solely the ligand and
`<PDB-of-receptor>` a PDB file containing the receptor (and other ligands).
In the second command, `<PDB-of-ligand-and-receptor` is a PDB file containing
both the ligand and receptor, and `--selection` requires the chain and residue
id of the ligand as a comma separated value, e.g. `A,1234`. Note that the
receptor (and other ligands) are used to determine the scaling factor of the
density map and used for collision detection during conformer sampling.

To see all options, type

    qfit_ligand -h

The main options are `-s` to give the angular stepsize in degree, and `-b` to
provide the number of degrees of freedom that are sampled simultaneously.
Reasonably values are `-s 1 -b 1`, `-s 6 -b 2`, and `-s 24 -b 3`. Decreasing
`-s` and especially increasing `-b` further will be RAM memory intensive and
will take significantly longer.

Other useful options are `-d` to store the results in a directory to your
chosing, and if `-v` to be more verbose about the output, especially when using
it interactively in the shell.

The output of `qfit_ligand` consists of the following files:

* *multiconformer.pdb*: Final occupancy weighted ligand multiconformer model.
* *conformer_N.pdb*: Conformers found before the final rescoring round, where *N* is an integer.
* *qfit_ligand.log*: Logging file of run.


## Combining multiconformer ligand with receptor

To combine the output multiconformer ligand with the receptor use the following command

    qfit_combine <multiconformer-ligand-pdbs> -r <receptor> --remove -o <output>

where `<multiconformer-ligand-pdbs>` is one or more PDB files of the ligand you
want to recombine with the receptor, `<receptor>` the PDB file of the receptor,
the `--remove` flag is optionally used the remove the ligand in the current
`<receptor>` file, and `<output>` is the file name of the combined PDB
structure.


## Converting MTZ to CCP4

If you have access to CCTBX/Phenix use `phenix.mtz2map` to convert a MTZ file
to CCP4. Make sure it outputs the 2mFo-DFc map. Read the documentation for
available options.


## License

The code is licensed under the MIT licence (see `LICENSE`).

The `spacegroups.py` module is based on the `SpaceGroups.py` module of the
`pymmlib` package, originally licensed under the Artistic License 2.0. See the
`licenses` directory for a copy of the original source code and its full license.

The `elements.py` is licensed under MIT, Copyright (c) 2005-2015, Christoph
Gohlke. See file header.

[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"

