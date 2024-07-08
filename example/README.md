## Advanced qFit features and options

Some of the advanced and specialized options available in qFit are demonstrated below. The PDB and map files for each of the examples are placed within their corresponding folders. 

### 1. Modelling alternate conformers for a residue of interest

`qfit_residue [COMPOSITE_OMIT_MAP_FILE] -l [LABELS] [PDB_FILE] [CHAIN,RESIDUE]`

Using the example 1G8A:

`qfit_residue qfit_residue_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_residue_example/1G8A.pdb A,175`

This will produce a parsimonious model containing up to 5 alternate conformers
for residue 175 of chain A of 1G8A.


### 2. Running qFit on cyro-EM structures. 

qFit can also use ccp4 map files as input. To model alternate conformers using
this type of map, it is also necessary to provide the resolution of the data,
which can be achieved by using the flag *-r*.

`qfit_protein [MAP_FILE] [PDB_FILE] -r [RESOLUTION]`

#### You also must use the -em flag for cryo-EM structures. 

For Cyro-EM ccp4 maps, you can use the example from the Apoferritin Chain A (PDB:7A4M). 

`qfit_protein qfit_cryoem_example/7A4M_box.ccp4 qfit_cryoem_example/7A4M_box.pdb -r 1.22 -em`

We recommend users run [qscore](https://github.com/gregdp/mapq) to determine which residues should be modeled using qFit. After running qscore, run qFit protein using the following command:

`qfit_protein qfit_cryoem_example/7A4M_box.ccp4 qfit_cryoem_example/7A4M_box.pdb -r 1.22 -em --qscore 7A4M.pdb__Q__apoF_chainA.ccp4_All.txt`

After *multiconformer_model2.pdb* has been generated, refine this model using:
`qfit_final_refine_cryoem.sh qfit_cryoem_example/apoF_chainA.ccp4 qfit_cryoem_example/apoF_chainA.pdb multiconformer_model2.pdb`

Note: a pre-generated *multiconformer_model2.pdb* file is place in the folder for reference.
Bear in mind that this final step currently depends on an existing installation
of the [Phenix software suite](https://phenix-online.org/). 

### 3. Run qFit segment only. If you have manually edited the output of qFit and would like to re-label the alt confs and normalized the occupancies in the PDB, run:
`qfit_protein qfit_segment_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_segment_example/multiconformer_model2_pruned.pdb --segement-only`

### 4. Deactivate backbone sampling and bond angle sampling to model alternate conformers for a single residue of interest (faster, less precise)

In its default mode, *qfit_residue* and *qfit_protein* samples backbone conformations
using our KGS routine. This can be disabled using the *--no-backbone* flag.

For even faster (and less precise) results, one can also disable the sampling of
the bond angle Cα-Cβ-Cγ, which can be deactivated by means of the *--no-sample-angle* flag.

Other useful sampling parameters that can be tweaked to make qFit run faster at
the cost of precision:

* Increase step size (in degrees) of sampling around each rotamer: *-s* flag (default: 10)
* Decrease range/neighborhood of sampling about preferred rotamers: *-rn* flag (default: 60)
* Disable parsimonious selection of the number of conformers output by qFit using the Bayesian Information Criterion (BIC): *--no-threshold-selection* flag.

Using the example 3K0N:

`qfit_residue qfit_residue_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_residue_example/1G8A.pdb A,113 --no-backbone --no-sample-angle -s 20 -rn 45 --no-threshold-selection`

For a full list of options, run:

`qfit_residue -h`


### 5. The same sampling parameters used in qfit_residue can be tweaked in qfit_protein:

Using the example 3K0N:

`qfit_protein qfit_protein_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_protein_example/1G8A.pdb --no-backbone --no-sample-angle -s 20 -rn 45 --no-threshold-selection`

### 6.  Parallelization:

The *qfit_protein* program can be executed in parallel and the number of concurrent processes
can be adjusted using the *-p* flag.

Using the example 3K0N, spawning 30 parallel processes:

`qfit_protein qfit_protein_example/composite_omit_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_protein_example/1G8A.pdb -p 30`

### 7. Modeling alternate conformers of a ligand

To model alternate conformers of ligands, the command line tool *qfit_ligand*
should be used:

`qfit_ligand [COMPOSITE_OMIT_MAP_FILE] -l [LABEL] [PDB_FILE] [CHAIN,LIGAND] -sm [SMILES]`

Where *LIGAND* corresponds to the numeric identifier of the ligand on the PDB
(aka res. number). The main output file is named *multiconformer_ligand_bound_with_protein.pdb*


If you wish to specify the number of ligand conformers for qFit to sample, use the flag `-nc [NUM_CONFS]`. The default number is set to 10,000. 

Using the example 4MS6:

`qfit_ligand qfit_ligand_example/4ms6_composite_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_ligand_example/4ms6.pdb A,702 -sm 'C1C[C@H](NC1)C(=O)CCC(=O)N2CCC[C@H]2C(=O)O' -nc 10000`


To refine *multiconformer_ligand_bound_with_protein.pdb*, use the following command

`qfit_final_refine_ligand.sh 4ms6.mtz`
