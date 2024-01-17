#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qfit.structure import Structure
from qfit.validator import Validator


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("map", type=str, help="Map.")
    p.add_argument("ligand", type=str, help="name of ligand")
    p.add_argument("--pdb", type=str, help="name of PDB")

    return p


def main():
  p = build_argparser()
  options = p.parse_args()
  # Load structure and prepare it
  structure = Structure.fromfile(options.structure)
  ligand = structure.extract("resn", options.ligand, "==")


  # Load and process the electron density map:
  xmap = XMap.fromfile(
    options.map
  )
  xmap = xmap.canonical_unit_cell()
  if options.scale:
    footprint = ligand
    scaler.scale(footprint, radius=options.scale_rmask * radius)

  xmap = xmap.extract(ligand.coor, padding=11)



  # Now that the conformers have been generated, the resulting
  # # conformations should be examined via GoodnessOfFit:
  validator = Validator(xmap, xmap.resolution)
