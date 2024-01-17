#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import csv
from qfit.scaler import MapScaler
from qfit.structure import Structure
from qfit.volume import XMap
from qfit.validator import Validator

'''
This script will calculate the RSCC of a ligand (or any residue) defined by their ligand name (--ligand) or residue number and chain id (--resi_chain). 
It will only work on mtz maps with 2FOFCWT,PH2FOFCWT. 

To run: 
 calc_rscc.py PDB_FILE.pdb MTZ_FILE.mtz --ligand AR6 --pdb PDB_NAME --directory /path/for/output/csv/file
'''

def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("map", type=str, help="Map.")
    p.add_argument("--ligand", type=str, help="name of ligand for RSCC to be calculated on")
    p.add_argument("--residue", type=str, help="Chain_ID, Residue_ID for RSCC to be calculated on")
    p.add_argument("--pdb", type=str, help="name of PDB")
    p.add_argument("--directory", type=str, help="Where to save RSCC info")
    return p


def main():
  p = build_argparser()
  options = p.parse_args()
  # Load structure and prepare it
  structure = Structure.fromfile(options.structure)
  if options.ligand is not None:
     ligand = structure.extract("resn", options.ligand, "==")
  elif options.residue is not None:
    chainid, resi = options.residue.split(",")
    ligand = structure.extract(f"resi {resi} and chain {chainid}")
  else:
      print('Please provide ligand name or residue ID and chain ID')

  # Load and process the electron density map:
  xmap = XMap.fromfile(
    options.map, label='2FOFCWT,PH2FOFCWT'
  )
  scaler = MapScaler(xmap)
  xmap = xmap.canonical_unit_cell()
  footprint = ligand
  scaler.scale(footprint, radius=1.5)

  xmap = xmap.extract(ligand.coor, padding=8)

  # Now that the conformers have been generated, the resulting
  # # conformations should be examined via GoodnessOfFit:
  validator = Validator(xmap, xmap.resolution, options.directory)
  rscc = validator.rscc(ligand)
  print("rscc = ", rscc)

  csv_filename = f"{options.pdb}_rscc.csv"

  # Write to CSV
  with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["PDB", "RSCC"])
    # Write the data
    writer.writerow([options.pdb, rscc])

if __name__ == "__main__":
    main()
