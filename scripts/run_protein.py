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
import os
import sys
import subprocess
import time

from qfit import Structure


p = argparse.ArgumentParser(description="Run qfit_residue on a whole structure.")
p.add_argument("map",
    help=("Input map file. Can be in CCP4, MRC or MTZ file. The MTZ file needs"
          " to have amplitudes and phases specified. Use --label to provide"
          " columns names."))
p.add_argument("structure",
    help="Input structure in PDB format.")
p.add_argument("-d", "--directory",
    help="Directory where results are stored.")
p.add_argument("-p", "--nprocessors", type=int, default=1,
    help="Number of processors to run.")
p.add_argument('-bb', '--backbone', action='store_true',
    help="Sample backbone locally using inverse kinematics null space sampling.")

args = p.parse_args()

s = Structure.fromfile(args.structure)
run_dirs = []
selections = []
for chain in s.chains:
    chainid = chain.chain[0]
    conformer = chain.conformers[0]
    for residue in conformer.residues:
        resseq, icode = residue.id
        sel = f'{chainid},{resseq}'
        if icode:
            sel += f':{icode}'
        run_dir = os.path.join(args.directory, f'{chainid}_{resseq}{icode}')

        run_dirs.append(run_dir)
        selections.append(sel)

processes = []
for run_dir, selection in zip(run_dirs, selections):
    cmd = ['qfit_residue', args.map, args.structure, selection, '-d', run_dir]
    if args.backbone:
        cmd.append('-bb')

    print("running:", ' '.join(cmd))
    p = subprocess.Popen(cmd)
    processes.append(p)
    while len(processes) >= args.nprocessors:
        time.sleep(1)
        running_processes = []
        for p in processes:
            if p.poll() is None:
                running_processes.append(p)
        processes = running_processes

while processes:
    time.sleep(1)
    running_processes = []
    for p in processes:
        if p.poll() is None:
            running_processes.append(p)
    processes = running_processes

print("Done running all residues.")
print("Combining conformers in structure.")
all_lines = []
for run_dir, selection in zip(run_dirs, selections):
    chainid, resseq = selection.split(',')
    icode = ''
    if ':' in resseq:
        resseq, icode = resseq.split(':')
    multiconf_fn = os.path.join(run_dir, f'multiconformer_{chainid}_{resseq}.pdb')
    if icode:
        multiconf_fn = os.path.join(run_dir, f'multiconformer_{chainid}_{resseq}_{icode}.pdb')
    if not os.path.exists(multiconf_fn):
        continue
    with open(multiconf_fn) as f:
        for line in f:
            if line.startswith('END'):
                continue
            all_lines.append(line)

multiconf_fn = os.path.join(args.directory, 'multiconformer.pdb')
with open(multiconf_fn, 'w') as f:
    for line in all_lines:
        f.write(line)
print(f"Finished. Multiconformer protein written to {multiconf_fn}")
