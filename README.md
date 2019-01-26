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


## Usage

The `qfit` package comes with several command line tools of which the most
important are

* `qfit_protein`: 
* `qfit_residue`: 

The options for each tool can be shown by typing

    qfit_<name> -h


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

