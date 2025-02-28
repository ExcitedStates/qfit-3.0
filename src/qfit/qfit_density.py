"Transform a structure into a density."

import argparse
import os

from . import XMap, EMMap
from . import Structure
from .transformer import FFTTransformer, Transformer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("pdb_file", type=str, help="PDB file containing structure.")
    p.add_argument("xmap", type=str, help="")
    p.add_argument(
        "-r", "--resolution", type=float, help="Resolution of map in angstrom."
    )
    p.add_argument(
        "-nf", "--no-fft", action="store_true", help="No FFT density map creation."
    )
    p.add_argument(
        "-o", "--output", type=str, default=None, help="Name of output density."
    )
    p.add_argument(
        "-em", "--em", type=bool, default=False, help="Calculate EM map or not."
    )

    args = p.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.pdb_file)[0] + ".ccp4"
    return args


def main():
    args = parse_args()
    
    structure = Structure.fromfile(args.pdb_file)
    
    if os.path.splitext(args.xmap)[1] != ".mtz":
        xmap = XMap.fromfile(args.xmap, resolution=args.resolution)
    else:
        xmap = XMap.fromfile(args.xmap)

    resolution = args.resolution
    out = XMap.zeros_like(xmap)

    if hasattr(xmap, "hkl") and not args.no_fft:
        transformer = FFTTransformer(structure, out, em=args.em)
    else:
        if args.resolution is not None:
            smax = 0.5 / resolution
            simple = False
        else:
            smax = None
            simple = True
        
        transformer = Transformer(structure, out, smax=smax, simple=simple, em=args.em)
    
    transformer.density()
    out.tofile(args.output)
