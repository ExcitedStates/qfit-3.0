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
    xmap.tofile('map.ccp4')
