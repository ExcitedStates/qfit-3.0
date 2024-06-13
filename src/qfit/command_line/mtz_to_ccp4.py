import argparse

from qfit import XMap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mtz")
    p.add_argument("-l", "--label", default="FWT,PHWT")

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    xmap = XMap.fromfile(args.mtz, label=args.label)
    space_group = xmap.unit_cell.space_group
    print("Spacegroup:", space_group.pdb_name)
    print("Number of primitive:", space_group.num_primitive_sym_equiv)
    print("Number of sym:", space_group.num_sym_equiv)
    print("Operations:")
    for symop in space_group.symop_list:
        print(symop)
    xmap.tofile("map.ccp4")
