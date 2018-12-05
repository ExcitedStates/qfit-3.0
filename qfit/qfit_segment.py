import os.path
import time
from argparse import ArgumentParser

from . import MapScaler, Structure, XMap
from .qfit import QFitSegment, QFitSegmentOptions


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
                   help="Density map in CCP4 or MRC format, or an MTZ file \
                   containing reflections and phases. For MTZ files \
                   use the --label options to specify columns to read.")
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("-l", "--label", default="FWT,PHWT", metavar="<F,PHI>",
            help="MTZ column labels to build density.")
    p.add_argument('-o', '--omit', action="store_true",
            help="Map file is a 2mFo-DFc OMIT map.")
    p.add_argument('-r', "--resolution", type=float, default=None, metavar="<float>",
            help="Map resolution in angstrom.")
    p.add_argument("-ns", "--no-scale", action="store_false", dest="scale",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.1, metavar="<float>",
            help="Densities values below cutoff are set to <density_cutoff_value")
    p.add_argument("-dv", "--density-cutoff-value", type=float, default=-1, metavar="<float>",
            help="Density values below <density-cutoff> are set to this value.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=1, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=5, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument("-rn", "--rotamer-neighborhood", type=float, default=40, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.3, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-f", "--fragment-length", type=int, default=5, metavar="<int>",
            help="Number of subsequent elements used during optimization.")
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
           help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    p.add_argument("-M", "--miosqp", dest="cplex", action="store_false",
            help="Use MIOSQP instead of CPLEX for the QP/MIQP calculations.")
    p.add_argument("-T","--threshold-selection", dest="bic_threshold", action="store_true",
            help="Use BIC to select the most parsimonious MIQP threshold")

    args = p.parse_args()

    return args


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass
    time0 = time.time()

    options = QFitSegmentOptions()
    options = options.apply_command_args(args)

    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract('e', 'H', '!=')

    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, scattering=options.scattering)
        footprint = structure.extract('record', 'ATOM')
        scaler.scale(footprint, radius=1)
        #scaler.cutoff(options.density_cutoff, options.density_cutoff_value)
    xmap = xmap.extract(structure.coor, padding=5)

    qfit = QFitSegment(structure, xmap, options)
    qfit()
    # Write to file

if __name__ == '__main__':
    main()
