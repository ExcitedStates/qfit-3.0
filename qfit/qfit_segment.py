import os.path
from argparse import ArgumentParser

from . import MapScaler, Structure, XMap
from .qfit import QFitSegment, QFitSegmentOptions


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")
    p.add_argument("-ns", "--no-scale", action="store_true",
            help="Do not scale density.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument("-c", "--cardinality", type=int, default=2, metavar="<int>",
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
    args = p.parse_args()

    return args


def main():

    options = QFitSegmentOptions()
    args = parse_args()
    options = options.apply_command_args(args)

    xmap = XMap.fromfile(args.xmap)
    structure = Structure.fromfile(args.structure).reorder()

    if not args.no_scale:
        scaler = MapScaler(
            xmap, mask_radius=1, scattering=options.scattering)
        scaler(structure)
        fname = os.path.join(options.directory, 'scaled.mrc')
        xmap.tofile(fname)


    # Divide structure up in connected segments
    #print("Dividing")
    #segment_resids = []
    #for rg in structure.residue_groups:
    #del segment

    #print("Building segments")
    #segments = []
    #for resids in segment_resids:
    #    for resid in resids:
    #        resi, icode = resid
    #        residue = structure.extract('resi', resi)
    #        if icode:
    #            residue = residue.extract('icode', icode)
    #        try:
    #            segment = segment.combine(residue)
    #        except UnboundLocalError:
    #            segment = residue
    #    segments.append(segment)

    #print("Writing segments.")
    #for n, segment in enumerate(segments, start=1):
    #    segment.tofile(f'segment_{n}.pdb')

    qfit = QFitSegment(structure, xmap, options)
    qfit()
    # Write to file

if __name__ == '__main__':
    main()
