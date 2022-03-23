import numpy as np
import pandas as pd

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args


args = parse_args()

methyl = []
structure = Structure.fromfile(args.structure).reorder()
structure = structure.extract('record', 'HETATM', '!=')
for chain in np.unique(structure.chain):
    for resi in np.unique(structure.extract('chain', chain, '==').resi):
        resname = structure.extract(f'chain {chain} and resi {resi}').resn[0]
        else:
          a1 = "HB2"
          a2 = "CB"
          if resname in ["THR", "ILE", "VAL"]:
            a1 = "HB"
            a2 = "CB"
          if resname == "GLY"
             a1 = "HA2"
             a2 = "CA"
          methyl.append(resi, a1, resi, a2, chain, resname) 

methyl_df = pd.DataFrame(methyl, columns =['resi', 'a1', 'resi', 'a2', 'chain', 'resn'])
methyl_df.to_csv(args.pdb + '_qFit_methyl.dat', sep='', index=False)

  
