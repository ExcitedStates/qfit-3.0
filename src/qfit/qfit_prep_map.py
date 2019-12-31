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
    args.xmap.tofile('scaled.ccp4')
    if args.selection is not None:
        chain, resi = args.selection.split(',')
        sel_str = f'chain {chain} and resi {resi}'
        if ':' in resi:
            resi, icode = resi.split(':')
            sel_str = f'chain {chain} and resi {resi} and icode {icode}'
        else:
            sel_str = f'chain {chain} and resi {resi}'
        footprint = args.structure.extract(sel_str)
    else:
        footprint = args.structure
    #scaler.subtract(footprint)
    args.xmap.tofile('final.ccp4')
