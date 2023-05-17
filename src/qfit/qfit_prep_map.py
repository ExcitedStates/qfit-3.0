import argparse
from qfit import XMap, Structure
from qfit.scaler import MapScaler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=XMap.fromfile)
    p.add_argument("structure", type=Structure.fromfile)
    p.add_argument("-s", "--selection", default=None)
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    scaler = MapScaler(args.xmap)
    scaler.scale(args.structure)
    args.xmap.tofile("scaled.ccp4")
    if args.selection is not None:
        chain, resi = args.selection.split(",")
        sel_str = f"chain {chain} and resi {resi}"
        if ":" in resi:
            resi, icode = resi.split(":")
            sel_str = f"chain {chain} and resi {resi} and icode {icode}"
        else:
            sel_str = f"chain {chain} and resi {resi}"
        footprint = args.structure.extract(sel_str)
    else:
        footprint = args.structure
    # scaler.subtract(footprint)
    args.xmap.tofile("final.ccp4")
