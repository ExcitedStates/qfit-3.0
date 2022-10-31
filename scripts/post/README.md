# qFit 3.2.2

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

If you use this software, please cite:
- [Riley BT, Wankowicz SA, et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

## Refinement
After *multiconformer_model2.pdb* has been generated, the model must need to be refined. Bear in mind that this final step currently depends on an existing installation of the Phenix software suite. This script is currently written to work with version Phenix 1.20.

[Phenix installation](https://phenix-online.org/documentation/install-setup-run.html)

X-ray crystallography:
`qfit_final_refine_xray.sh /path/to/mtz_file.mtz multiconformer_model2.pdb`

Cryo-EM: 
`qfit_final_refine_cryoem.sh /path/to/ccp4_file.ccp4 original_pdb.pdb multiconformer_model2.pdb`


## Analysis Scripts

### 1. Calculating Order Parameters


Usage: 
`make_methyl_df.py ${PDB}_qFit.pdb`
`calc_OP.py ${PDB}_qFit.dat ${PDB}_qFit.pdb ${PDB}_qFit.out -r ${res} -b ${b_fac}`



