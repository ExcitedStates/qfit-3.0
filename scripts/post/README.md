# qFit 3.2.2

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

If you use this software, please cite:
- [Riley BT, Wankowicz SA, et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

Bear in mind that this final step currently depends on an existing installation
of the Phenix software suite. This script is currently written to work with version Phenix 1.20.

After *multiconformer_model2.pdb* has been generated, refine this model using:
`qfit_final_refine_xray.sh /path/to/3K0N.mtz multiconformer_model2.pdb`

After *multiconformer_model2.pdb* has been generated, refine this model using:
`qfit_final_refine_cryoem.sh /path/to/apoF_chainA.ccp4 apoF_chainA.pdb multiconformer_model2.pdb`
