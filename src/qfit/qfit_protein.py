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
import gc
import pkg_resources  # part of setuptools
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
import multiprocessing as mp
import os.path
import os
import sys
import time
import copy
import argparse
from math import ceil
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS

os.environ["OMP_NUM_THREADS"] = "1"


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
    pass


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=CustomHelpFormatter,
                                description=__doc__)
    p.add_argument("map", type=str,
                   help="Density map in CCP4 or MRC format, or an MTZ file "
                        "containing reflections and phases. For MTZ files "
                        "use the --label options to specify columns to read.")
    p.add_argument("structure",
                   help="PDB-file containing structure.")

    # Map input options
    p.add_argument("-l", "--label", default="FWT,PHWT",
                   metavar="<F,PHI>",
                   help="MTZ column labels to build density.")
    p.add_argument('-r', "--resolution", default=None,
                   metavar="<float>", type=float,
                   help="Map resolution in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-m", "--resolution_min", default=None,
                   metavar="<float>", type=float,
                   help="Lower resolution bound in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
                   help="Scattering type.")
    p.add_argument("-rb", "--randomize-b", action="store_true", dest="randomize_b",
                   help="Randomize B-factors of generated conformers.")
    p.add_argument('-o', '--omit', action="store_true",
                   help="Map file is an OMIT map. This affects the scaling procedure of the map.")

    # Map prep options
    p.add_argument("-ns", "--no-scale", action="store_false", dest="scale",
                   help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", default=0.3,
                   metavar="<float>", type=float,
                   help="Density values below this value are set to <density-cutoff-value>.")
    p.add_argument("-dv", "--density-cutoff-value", default=-1,
                   metavar="<float>", type=float,
                   help="Density values below <density-cutoff> are set to this value.")
    p.add_argument("-nosub", "--no-subtract", action="store_false", dest="subtract",
                   help="Do not subtract Fcalc of the neighboring residues when running qFit.")
    p.add_argument("-pad", "--padding", default=8.0,
                   metavar="<float>", type=float,
                   help="Padding size for map creation.")
    p.add_argument("-nw", "--no-waters", action="store_true", dest="nowaters",
                   help="Keep waters, but do not consider them for soft clash detection.")

    # Sampling options
    p.add_argument('-bb', "--no-backbone", action="store_false", dest="sample_backbone",
                   help="Do not sample backbone using inverse kinematics.")
    p.add_argument('-bbs', "--backbone-step", default=0.1, dest="sample_backbone_step",
                   metavar="<float>", type=float,
                   help="Stepsize for the amplitude of backbone sampling.")
    p.add_argument('-bba', "--backbone-amplitude", default=0.3, dest="sample_backbone_amplitude",
                   metavar="<float>", type=float,
                   help="Maximum backbone amplitude.")
    p.add_argument('-sa', "--no-sample-angle", action="store_false", dest="sample_angle",
                   help="Do not sample N-CA-CB angle.")
    p.add_argument('-sas', "--sample-angle-step", default=3.75, dest="sample_angle_step",
                   metavar="<float>", type=float,
                   help="N-CA-CB bond angle sampling step in degrees.")
    p.add_argument('-sar', "--sample-angle-range", default=7.5, dest="sample_angle_range",
                   metavar="<float>", type=float,
                   help="N-CA-CB bond angle sampling range in degrees [-x,x].")
    p.add_argument("-b", "--dofs-per-iteration", default=2,
                   metavar="<int>", type=int,
                   help="Number of internal degrees that are sampled/built per iteration.")
    p.add_argument("-s", "--dihedral-stepsize", default=10,
                   metavar="<float>", type=float,
                   help="Stepsize for dihedral angle sampling in degrees.")
    p.add_argument("-rn", "--rotamer-neighborhood", default=60,
                   metavar="<float>", type=float,
                   help="Neighborhood of rotamer to sample in degrees.")
    p.add_argument("--remove-conformers-below-cutoff", action="store_true", dest="remove_conformers_below_cutoff",
                   help=("Remove conformers during sampling that have atoms "
                         "with no density support, i.e. atoms are positioned "
                         "at density values below <density-cutoff>."))
    p.add_argument('-cf', "--clash_scaling_factor", default=0.75,
                   metavar="<float>", type=float,
                   help="Set clash scaling factor.")
    p.add_argument('-ec', "--external_clash", action="store_true", dest="external_clash",
                   help="Enable external clash detection during sampling.")
    p.add_argument("-bs", "--bulk_solvent_level", default=0.3,
                   metavar="<float>", type=float,
                   help="Bulk solvent level in absolute values.")
    p.add_argument("-c", "--cardinality", default=5,
                   metavar="<int>", type=int,
                   help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", default=0.2,
                   metavar="<float>", type=float,
                   help="Threshold constraint used during MIQP.")
    p.add_argument("-hy", "--hydro", action="store_true", dest="hydro",
                   help="Include hydrogens during calculations.")
    p.add_argument('-rmsd', "--rmsd_cutoff", default=0.01,
                   metavar="<float>", type=float,
                   help="RMSD cutoff for removal of identical conformers.")
    p.add_argument("-T", "--no-threshold-selection", dest="bic_threshold",
                   action="store_false", help="Do not use BIC to select the most parsimonious MIQP threshold")
    p.add_argument("-p", "--nproc", type=int, default=1, metavar="<int>",
                   help="Number of processors to use.")

    # qFit Segment options
    p.add_argument("-f", "--fragment-length", default=4, dest="fragment_length",
                   metavar="<int>", type=int,
                   help="Fragment length used during qfit_segment.")
    p.add_argument("-Ts", "--no-segment-threshold-selection", action="store_false", dest="seg_bic_threshold",
                   help="Do not use BIC to select the most "
                        "parsimonious MIQP threshold (segment).")

    # Output options
    p.add_argument("-d", "--directory", default='.',
                   metavar="<dir>", type=os.path.abspath,
                   help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
                   help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    return p


class QFitProteinOptions(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.checkpoint = False
        self.pdb = None


class QFitProtein:
    def __init__(self, structure, xmap, options):
        self.xmap = xmap
        self.structure = structure
        self.options = options

    def run(self):
        if self.options.pdb is not None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        multiconformer = self._run_qfit_residue()
        structure = Structure.fromfile('multiconformer_model.pdb')#.reorder()
        structure = structure.extract('e', 'H', '!=')
        multiconformer = self._run_qfit_segment(structure)
        return multiconformer

    def _run_qfit_residue(self):
        """Run qfit independently over all residues."""
        # This function hands out the job in parallel to a Pool of Workers.
        # To create Workers, we will use "forkserver" where possible,
        #     and default to "spawn" elsewhere (e.g. on Windows).
        try:
            ctx = mp.get_context(method="forkserver")
        except ValueError:
            ctx = mp.get_context(method="spawn")

        # Print execution stats
        residues = list(self.structure.single_conformer_residues)
        print(f"RESIDUES: {len(residues)}")
        print(f"NPROC: {self.options.nproc}")

        # Launch a Pool and run Jobs
        # Here, we calculate alternate conformers for individual residues.
        with ctx.Pool(processes=self.options.nproc, maxtasksperchild=4) as pool:
            futures = [pool.apply_async(QFitProtein._run_qfit_instance,
                                        kwds={'residue': residue,
                                              'structure': self.structure,
                                              'xmap': self.xmap,
                                              'options': self.options})
                       for residue in residues]

            # Make sure all jobs are finished
            for f in futures:
                f.wait()

        # Extract non-protein atoms
        hetatms = self.structure.extract('record', 'HETATM', '==')
        waters = self.structure.extract('record', 'ATOM', '==')
        waters = waters.extract('resn', 'HOH', '==')
        hetatms = hetatms.combine(waters)

        # Combine all multiconformer residues into one structure
        for residue in residues:
            if residue.resn[0] not in ROTAMERS:
                hetatms = hetatms.combine(residue)
                continue
            chain = residue.chain[0]
            resid, icode = residue.id
            directory = os.path.join(self.options.directory,
                                     f"{chain}_{resid}")
            if icode:
                directory += f"_{icode}"
            fname = os.path.join(directory, 'multiconformer_residue.pdb')
            if not os.path.exists(fname):
                continue
            residue_multiconformer = Structure.fromfile(fname)
            try:
                multiconformer = multiconformer.combine(residue_multiconformer)
            except UnboundLocalError:
                multiconformer = residue_multiconformer
            except FileNotFoundError:
                print("File not found!", fname)
                pass

        multiconformer = multiconformer.combine(hetatms)
        fname = os.path.join(self.options.directory,
                             "multiconformer_model.pdb")
        multiconformer.tofile(fname)
        return multiconformer

    def _run_qfit_segment(self, multiconformer):
        self.options.randomize_b = False
        self.options.bic_threshold = self.options.seg_bic_threshold
        if self.options.seg_bic_threshold:
            self.options.fragment_length = 3
        else:
            self.options.threshold = 0.2
        self.xmap = self.xmap.extract(self.structure.coor, padding=5)
        qfit = QFitSegment(multiconformer, self.xmap, self.options)
        multiconformer = qfit()
        fname = os.path.join(self.options.directory,
                             self.pdb + "multiconformer_model2.pdb")
        multiconformer.tofile(fname)
        return multiconformer

    @staticmethod
    def _run_qfit_instance(residue, structure, xmap, options):
        """Run qfit on a single residue to determine density-supported conformers."""

        # Don't run qfit if we have a ligand or water
        if residue.type != 'rotamer-residue':
            return

        # This function is run in a subprocess, so `structure` and `residue` have
        #     been 'copied' (pickled+unpickled) as best as possible.

        # However, `structure`/`residue` objects pickled and passed to subprocesses do
        #     not contain attributes decorated by @_structure_properties.
        #     This decorator attaches 'getter' and 'setter' _local_ functions to the attrs
        #     (defined within, and local to the _structure_properties function).
        #     Local functions are **unpickleable**, and as a result, so are these attrs.
        # This includes:
        #     (record, atomid, name, altloc, resn, chain, resi, icode,
        #      q, b, e, charge, coor, active, u00, u11, u22, u01, u02, u12)
        # Similarly, these objects are also missing attributes wrapped by @property:
        #     (covalent_radius, vdw_radius)
        # Finally, the _selector object is only partially pickleable,
        #     as it contains a few methods that are defined by a local lambda inside
        #     pyparsing._trim_arity().

        # Since all these attributes are attached by __init__ of the
        #     qfit.structure.base_structure._BaseStructure class,
        #     here, we call __init__ again, to make sure these objects are
        #     correctly initialised in a subprocess.
        structure.__init__(
            structure.data,
            selection=structure._selection,
            parent=structure.parent,
        )
        residue.__init__(
            residue.data,
            resi=residue.id[0],
            icode=residue.id[1],
            type=residue.type,
            selection=residue._selection,
            parent=residue.parent,
        )

        # We don't want this subprocess to be verbose
        options.verbose = False

        # Build the residue results directory
        chainid = residue.chain[0]
        resi, icode = residue.id
        identifier = f"{chainid}_{resi}"
        if icode:
            identifier += f'_{icode}'
        base_directory = options.directory
        options.directory = os.path.join(base_directory, identifier)
        try:
            os.makedirs(options.directory)
        except OSError:
            pass

        # Exit early if we have already run qfit for this residue
        fname = os.path.join(options.directory, 'multiconformer_residue.pdb')
        if os.path.exists(fname):
            return

        # Copy the structure
        structure_new = structure
        structure_resi = structure.extract(f'resi {resi} and chain {chainid}')
        if icode:
            structure_resi = structure_resi.extract('icode', icode)
        chain = structure_resi[chainid]
        conformer = chain.conformers[0]
        residue = conformer[residue.id]
        altlocs = sorted(list(set(residue.altloc)))
        if len(altlocs) > 1:
            try:
                altlocs.remove('')
            except ValueError:
                pass
            for altloc in altlocs[1:]:
                sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                sel_str = f"not ({sel_str})"
                structure_new = structure_new.extract(sel_str)

        # Copy the map
        xmap_reduced = xmap.extract(residue.coor, padding=options.padding)

        # Exception handling in case qFit-residue fails:
        qfit = QFitRotamericResidue(residue, structure_new,
                                    xmap_reduced, options)
        try:
            qfit.run()
        except RuntimeError:
            print(f"[WARNING] qFit was unable to produce an alternate conformer for residue {resi} of chain {chainid}.")
            print(f"Using deposited conformer A for this residue.")
            qfit.conformer = residue.copy()
            qfit._occupancies = [residue.q]
            qfit._coor_set = [residue.coor]
            qfit._bs = [residue.b]

        # Save multiconformer_residue
        qfit.tofile()

        # How many conformers were found?
        n_conformers = len(qfit.get_conformers())

        # Freeing up some memory to avoid memory issues:
        del xmap_reduced
        del qfit
        gc.collect()

        # Return a string about the residue that was completed.
        return f"{identifier} {residue.resn[0]}: {n_conformers} conformers"


def prepare_qfit_protein(options):
    """Loads files to build a QFitProtein job."""

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure).reorder()
    if not options.hydro:
        structure = structure.extract('e', 'H', '!=')

    # Load map and prepare it
    xmap = XMap.fromfile(
        options.map, resolution=options.resolution, label=options.label
    )
    xmap = xmap.canonical_unit_cell()
    if options.scale is True:
        scaler = MapScaler(xmap, scattering=options.scattering)
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        scaler.scale(structure, radius=radius)

    return QFitProtein(structure, xmap, options)


def main():
    """Default entrypoint for qfit_protein."""

    # Collect and act on arguments
    #   (When args==None, argparse will default to sys.argv[1:])
    p = build_argparser()
    args = p.parse_args(args=None)
    try:
        os.mkdir(args.directory)
    except OSError:
        pass
    print_run_info(args)
    options = QFitProteinOptions()
    options.apply_command_args(args)

    # Build a QFitProtein job
    qfit = prepare_qfit_protein(options)

    # Run the QFitProtein job
    time0 = time.time()
    multiconformer = qfit.run()
    print(f"Total time: {time.time() - time0}s")
