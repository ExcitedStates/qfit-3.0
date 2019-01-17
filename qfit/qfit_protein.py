from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
import multiprocessing as mp
import os.path
import sys
import time
import copy
from argparse import ArgumentParser
from math import ceil
from . import MapScaler, Structure, XMap


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
                   help="Density map in CCP4 or MRC format, or an MTZ file "
                        "containing reflections and phases. For MTZ files "
                        "use the --label options to specify columns to read.")
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Map input options
    p.add_argument("-l", "--label", default="FWT,PHWT", metavar="<F,PHI>",
                   help="MTZ column labels to build density.")
    p.add_argument('-r', "--resolution", type=float, default=None, metavar="<float>",
            help="Map resolution in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
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
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.3, metavar="<float>",
            help="Densities values below cutoff are set to <density_cutoff_value")
    p.add_argument("-dv", "--density-cutoff-value", type=float, default=-1, metavar="<float>",
            help="Density values below <density-cutoff> are set to this value.")

    # Sampling options
    p.add_argument('-bb', "--backbone", dest="sample_backbone", action="store_true",
            help="Sample backbone using inverse kinematics.")
    p.add_argument('-bbs', "--backbone-step", dest="sample_backbone_step",
                   type=float, default=0.1, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument('-bba', "--backbone-amplitude", dest="sample_backbone_amplitude",
                   type=float, default=0.3, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument('-sa', "--sample-angle", dest="sample_angle", action="store_true",
            help="Sample N-CA-CB angle.")
    p.add_argument('-sas', "--sample-angle-step", dest="sample_angle_step",
                   type=float, default=3.75, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument('-sar', "--sample-angle-range", dest="sample_angle_range",
                   type=float, default=7.5, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=6, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-rn", "--rotamer-neighborhood", type=float,
            default=40, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("--no-remove-conformers-below-cutoff", action="store_false",
                   dest="remove_conformers_below_cutoff",
                   help=("Remove conformers during sampling that have atoms "
                         "that have no density support for, ie atoms are "
                         "positioned at density values below cutoff value."))
    p.add_argument('-cf', "--clash_scaling_factor", type=float, default=0.75, metavar="<float>",
            help="Set clash scaling factor. Default = 0.75")
    p.add_argument('-ec', "--external_clash", dest="external_clash", action="store_true",
            help="Enable external clash detection during sampling.")
    p.add_argument("-bs", "--bulk_solvent_level", default=0.3, type=float,
                   metavar="<float>", help="Bulk solvent level in absolute values.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
                   help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.3,
                   metavar="<float>", help="Treshold constraint used during MIQP.")
    p.add_argument("-hy", "--hydro", dest="hydro", action="store_true",
                   help="Include hydrogens during calculations.")
    p.add_argument("-M", "--miosqp", dest="cplex", action="store_false",
                   help="Use MIOSQP instead of CPLEX for the QP/MIQP calculations.")
    p.add_argument("-T", "--threshold-selection", dest="bic_threshold",
                   action="store_true", help="Use BIC to select the most parsimonious MIQP threshold")
    p.add_argument("-p", "--nproc", type=int, default=1, metavar="<int>",
                   help="Number of processors to use.")

    # qFit Segment options
    p.add_argument("-f", "--fragment-length", type=int,
                   default=4, metavar="<int>", help="Fragment length used during qfit_segment.")
    p.add_argument('-rmsd', "--rmsd_cutoff", type=float, default=0.01, metavar="<float>",
            help="RMSD cutoff for removal of identical conformers. Default = 0.01")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
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
        self.omit = False


class QFitProtein:

    def __init__(self, structure, xmap, options):

        self.xmap = xmap
        self.structure = structure
        self.options = options

    def run(self):

        multiconformer = self._run_qfit_residue()
        self.options.bic_threshold = False
        multiconformer = self._run_qfit_segment(multiconformer)
        return multiconformer

    def _run_qfit_residue(self):
        """Run qfit on each residue separately."""
        processes = []
        residues = list(self.structure.residues)
        nresidues = len(residues)
        print(f"RESIDUES: {nresidues}")
        nproc = min(self.options.nproc, nresidues)
        nresidues_per_job = int(ceil(nresidues / nproc))
        counter = _Counter()
        for n in range(nproc):
            init_residue = n * nresidues_per_job
            end_residue = min(init_residue + nresidues_per_job, nresidues)
            residues_to_qfit = residues[init_residue: end_residue]
            args = (residues_to_qfit, self.structure, self.xmap,
                    self.options, counter)
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
        hetatms = self.structure.extract('record', 'HETATM', '==')
        multiconformer = multiconformer.combine(hetatms)
        fname = os.path.join(self.options.directory,
                             "multiconformer_model.pdb")
        multiconformer.tofile(fname)
        return multiconformer

    def _run_qfit_segment(self, multiconformer):

        qfit = QFitSegment(multiconformer, self.xmap, self.options)
        multiconformer = qfit()
        fname = os.path.join(self.options.directory,
                             "multiconformer_model2.pdb")
        multiconformer.tofile(fname)
        return multiconformer

    @staticmethod
    def _run_qfit_instance(residues, structure, xmap, options, counter):
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

                structure_new = copy.deepcopy(structure)
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



                xmap.array[:] = base_density
                # Prepare X-ray map
                if options.scale:
                    # Prepare X-ray map
                    scaler = MapScaler(xmap, scattering=options.scattering)
                    if options.omit:
                        footprint = residue
                    else:
                        sel_str = f"resi {resi} and chain {chain}"
                        if icode:
                            sel_str += f" and icode {icode}"
                        sel_str = f"not ({sel_str})"
                        footprint = structure_new.extract(sel_str)
                        footprint = footprint.extract('record', 'ATOM')
                    scaler.scale(footprint, radius=1)
                    # scaler.cutoff(options.density_cutoff,
                    #                options.density_cutoff_value)
                xmap_reduced = xmap.extract(residue.coor, padding=5)

                qfit = QFitRotamericResidue(residue, structure_new,
                                            xmap_reduced, options)
                # Exception handling in case qFit-residue fails:
                try:
                    qfit.run()
                except RuntimeError:
                    print(f"[WARNING] qFit was unable to produce an alternate conformer for residue {resi} of chain {chain}.")
                    print(f"Using deposited conformer A for this residue.")
                    qfit.conformer = residue.copy()
                    qfit._occupancies = [residue.q]
                    qfit._coor_set = [residue.coor]
                qfit.tofile()
            counter.increment()


def main():

    args = parse_args()
    try:
        os.mkdir(args.directory)
    except OSError:
        pass

    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    if not args.hydro:
        structure = structure.extract('e', 'H', '!=')

    options = QFitProteinOptions()
    options.apply_command_args(args)

    xmap = XMap.fromfile(args.map, resolution=args.resolution,
                         label=args.label)
    xmap = xmap.canonical_unit_cell()


    time0 = time.time()
    qfit = QFitProtein(structure, xmap, options)
    multiconformer = qfit.run()
    print(f"Total time: {time.time() - time0}s")
