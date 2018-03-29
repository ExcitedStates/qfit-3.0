"""
Usage:
    qfit_protein [options] <map> <resolution> <pdb>
    qfit_protein [options] <mtz> <pdb>
    qfit_protein (-h | --help)

Arguments:
    <map>             File containing density map, in either CCP4 or MRC format.
    <resolution>      Resolution of map in angstrom.
    <pdb>             PDB-file containing protein to fit.

Options:
    -h, --help         Show this screen.
    -c, --cardinality=<int>          Cardinality constraint used during MIQP.
    -t, --threshold=<float>        Threshold constraint used during MIQP.
    -d, --directory=<dir>  Directory to store the results.
    -v, --verbose               Be verbose.
"""

import logging
import multiprocessing as mp
import os.path
import sys
import time
from argparse import ArgumentParser
from math import ceil

from . import MapScaler, Structure, XMap
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions, QFitSegment, QFitSegmentOptions


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("map",
           help="File containing density map, in either CCP4 or MRC format.")
    p.add_argument("resolution", type=float,
           help='Resolution of map in angstrom.')
    p.add_argument("structure",
           help="PDB-file containing protein to analyze.")
    p.add_argument("-c", "--cardinality", type=int, default=2, metavar="<int>",
           help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.3, metavar="<float>",
           help="Threshold constraint used during MIQP.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=10, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument("-r", "--rotamer-neighborhood", type=float, default=40, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("-p", "--nproc", type=int, default=1, metavar="<int>",
           help="Number of processors to use.")
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
           help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
           help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
           help="Be verbose.")

    args = p.parse_args()
    return args


class _Counter:
    """Thread-safe counter object to follow progress"""

    def __init__(self):
        self.val = mp.RawValue('i', 0)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class QFitProteinOptions(QFitRotamericResidueOptions, QFitSegmentOptions):

    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True


class QFitProtein:

    def __init__(self, structure, xmap, options):

        self.xmap = xmap
        self.structure = structure
        self.options = options

    def run(self):

        multiconformer = self._run_qfit_residue()
        #multiconformer = self._run_qfit_segment(multiconformer)
        return multiconformer

    def _run_qfit_residue(self):
        """Run qfit on each residue separately."""

        processes = []
        residues = list(self.structure.residues)
        nresidues = len(residues)
        nproc = min(self.options.nproc, nresidues)
        nresidues_per_job = int(ceil(nresidues / nproc))
        counter = _Counter()
        for n in range(nproc):
            init_residue = n * nresidues_per_job
            end_residue = min(init_residue + nresidues_per_job, nresidues)
            residues_to_qfit = residues[init_residue: end_residue]
            args = (residues_to_qfit, self.xmap, self.options, counter)
            process = mp.Process(target=self._run_qfit_instance, args=args)
            processes.append(process)

        for p in processes:
            p.start()

        # Update on progress
        if self.options.verbose and sys.stdout.isatty():
            line = '{n} / {total}  time passed: {passed:.0f}s        \r'
            time0 = time.time()
            while True:
                n = counter.value()
                percentage = (n + 1) / float(nresidues) * 100
                time_passed = time.time() - time0
                msg = line.format(n=n, total=nresidues, passed=time_passed)
                sys.stdout.write(msg)
                sys.stdout.flush()
                if n >= nresidues:
                    sys.stdout.write('\n')
                    break
                time.sleep(0.5)

        for p in processes:
            p.join()

        # Combine all multiconformer residues into one structure
        for residue in residues:
            if residue.type == 'ligand':
                continue
            chain = residue.chain[0]
            resid, icode = residue.id
            directory = os.path.join(self.options.directory, f"{chain}_{resid}")
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
                pass
        fname = os.path.join(self.options.directory, "multiconformer_model.pdb")
        multiconformer.tofile(fname)
        return multiconformer

    def _run_qfit_segment(self, multiconformer):

        qfit = QFitSegment(multiconformer, self.xmap, self.options)
        multiconformer = qfit()
        fname = os.path.join(self.options.directory, "multiconformer_model2.pdb")
        multiconformer.tofile(multiconformer)
        return multiconformer

    @staticmethod
    def _run_qfit_instance(residues, xmap, options, counter):

        options.verbose = False
        base_directory = options.directory
        base_density = xmap.array.copy()
        for residue in residues:
            if residue.type == 'rotamer-residue':
                chain = residue.chain[0]
                resi, icode = residue.id

                identifyer = f"{chain}_{resi}"
                if icode:
                    identifyer += f'_{icode}'
                options.directory = os.path.join(base_directory, identifyer)
                try:
                    os.makedirs(options.directory)
                except OSError:
                    pass
                try:
                    xmap.array[:] = base_density
                    scaler = MapScaler(xmap, scale=False, subtract=True)
                    receptor = residue.parent.parent.parent
                    if icode:
                        footprint = receptor.extract(f'not (resi {resi} and icode {icode})')
                    else:
                        footprint = receptor.extract('resi', resi, '!=')
                    scaler(footprint)
                    qfit = QFitRotamericResidue(residue, xmap, options)
                    qfit()
                    qfit.tofile()
                except RuntimeError:
                    pass
            counter.increment()


def main():

    args = parse_args()

    try:
        os.mkdir(args.directory)
    except OSError:
        pass

    xmap = XMap.fromfile(args.map)
    resolution = args.resolution
    # Remove alternate conformers except for the A conformer
    structure = Structure.fromfile(args.structure).extract('altloc', ('', 'A'))
    structure.altloc = ''
    structure.q = 1

    # Prepare map once
    scaler = MapScaler(xmap)
    scaler(structure)

    options = QFitProteinOptions()
    options.apply_command_args(args)

    time0 = time.time()
    qfit = QFitProtein(structure, xmap, options)
    multiconformer = qfit.run()
    print(f"Total time: {time.time() - time0}s")


