#!/usr/bin/env python
"""Calculate RMSF by residue and altloc.

Iterate through altlocs of each residue,
reporting mean heavy-atom RMSD from all other altlocs of the residue.
"""

from __future__ import annotations

import os
import argparse
import itertools as itl
from typing import Generator, Sequence, TypeVar
from qfit import Structure

T = TypeVar('T')


def _build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")

    return p


def pivot_and_remainder(seq: Sequence[T]) -> Generator[tuple[T, list[T]], None, None]:
    """Iterates over a sequence, yielding an element and the remainder of the sequence as a list.

    >>> p = pivot_and_remainder([2, 3, 5, 8])
    >>> next(p)
    (2, [3, 5, 8])
    >>> next(p)
    (3, [2, 5, 8])

    >>> p = pivot_and_remainder([2])
    >>> next(p)
    (2, [])
    """
    for i in range(len(seq)):
        yield seq[i], list(itl.chain(seq[:i], seq[i+1:]))


def get_metrics(structure: Structure) -> None:
    """Calculate RMSF by residue and altloc.

    Iterate through altlocs of each residue,
    reporting mean heavy-atom RMSD from all other altlocs of the residue."""

     # Print a column header
    print("resi", "chain", "residue_altloc_rmsd", "n_altlocs")

    for residue in (
        structure.extract('record', "ATOM")     # Don't analyse metals/ligands
                 .extract('resn', "HOH", "!=")  # Don't analyse waters
                 .extract('name', "H", "!=")    # Sometimes backbone N-H atoms are present in some altlocs, not all. Avoid analysing them.
                 .extract('e', "H", "!=")       # Sometimes His protonation states differ between altlocs. Avoid analysing all H.
    ).residue_groups:
        altlocs = sorted(list(set(residue.altloc)))
        resi = residue.resi[0]
        chainid = residue.chain[0]

        # Guard: if there's only 1 altloc at this residue...
        if len(altlocs) == 1:
            print(resi, chainid, 0., len(altlocs))
            continue

        try:
            altlocs.remove('')  # Remove the 'common backbone' from analysis, if present
        except ValueError:
            pass

        for altloc1, remaining_altlocs in pivot_and_remainder(altlocs):
            conf1 = residue.extract('altloc', altloc1)
            tot_rmsd: float = 0.
            numlocs: int = 0

            for altloc2 in remaining_altlocs:
                conf2 = residue.extract('altloc', altloc2)
                tot_rmsd += conf1.rmsd(conf2)
                numlocs += 1

            try:
                avg_rmsd: float = tot_rmsd / numlocs
            except ZeroDivisionError:
                avg_rmsd = 0.

            print(resi, chainid, round(avg_rmsd, 2), len(altlocs))


def _main():
    # Collect and act on arguments
    #   (When args==None, argparse will default to sys.argv[1:])
    argparser = _build_argparser()
    cmdline_args = argparser.parse_args(args=None)

    try:
        os.mkdir(cmdline_args.directory)
    except OSError:
        pass

    # Run main script
    get_metrics(Structure.fromfile(cmdline_args.structure).reorder())


if __name__ == '__main__':
    _main()
