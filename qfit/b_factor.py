#Edited by Stephanie Wankowicz
#began: 2019-05-01
'''
Excited States software: qFit 3.0
Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu

How to run:
b_factor $pdb.mtz $pdb.pdb --pdb $pdb
'''

import pkg_resources  # part of setuptools
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
from .qfit_protein import QFitProteinOptions, QFitProtein
import multiprocessing as mp
import os.path
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from math import ceil
from . import MapScaler, Structure, XMap
from .structure.base_structure import _BaseStructure


os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
                   help="Density map in CCP4 or MRC format, or an MTZ file "
                        "containing reflections and phases. For MTZ files "
                        "use the --label options to specify columns to read.")
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Map input options
    p.add_argument("-l", "--label", default="FWT,PHWT", metavar="<F,PHI>",
                   help="MTZ column labels to build density.")
    p.add_argument('-r', "--resolution", type=float, default=None, metavar="<float>",
            help="Map resolution in angstrom. Only use when providing CCP4 map files.")

    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument("-rb", "--randomize-b", action="store_true", dest="randomize_b",
            help="Randomize B-factors of generated conformers.")
    p.add_argument('-o', '--omit', action="store_true",
            help="Map file is an OMIT map. This affects the scaling procedure of the map.")

    # Map prep options
    p.add_argument("-ns", "--no-scale", action="store_false", dest="scale",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.3, metavar="<float>",
            help="Densities values below cutoff are set to <density_cutoff_value")
    p.add_argument("-dv", "--density-cutoff-value", type=float, default=-1, metavar="<float>",
            help="Density values below <density-cutoff> are set to this value.")

    # Sampling options
    p.add_argument('-bb', "--backbone", dest="sample_backbone", action="store_true",
            help="Sample backbone using inverse kinematics.")
    p.add_argument('-bbs', "--backbone-step", dest="sample_backbone_step",
                   type=float, default=0.1, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument('-sa', "--sample-angle", dest="sample_angle", action="store_true",
            help="Sample N-CA-CB angle.")
    p.add_argument('-sas', "--sample-angle-step", dest="sample_angle_step",
                   type=float, default=3.75, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument('-sar', "--sample-angle-range", dest="sample_angle_range",
                   type=float, default=7.5, metavar="<float>",
                   help="Sample N-CA-CB angle.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=6, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-rn", "--rotamer-neighborhood", type=float,
            default=60, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("--no-remove-conformers-below-cutoff", action="store_false",
                   dest="remove_conformers_below_cutoff",
                   help=("Remove conformers during sampling that have atoms "
                         "that have no density support for, ie atoms are "
                         "positioned at density values below cutoff value."))
    p.add_argument('-cf', "--clash_scaling_factor", type=float, default=0.75, metavar="<float>",
            help="Set clash scaling factor. Default = 0.75")
    p.add_argument('-ec', "--external_clash", dest="external_clash", action="store_true",
            help="Enable external clash detection during sampling.")
    p.add_argument("-bs", "--bulk_solvent_level", default=0.3, type=float,
                   metavar="<float>", help="Bulk solvent level in absolute values.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
                   help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.2,
                   metavar="<float>", help="Treshold constraint used during MIQP.")
    p.add_argument("-hy", "--hydro", dest="hydro", action="store_true",
                   help="Include hydrogens during calculations.")
    p.add_argument("-M", "--miosqp", dest="cplex", action="store_false",
                   help="Use MIOSQP instead of CPLEX for the QP/MIQP calculations.")
    p.add_argument("-T", "--threshold-selection", dest="bic_threshold",
                   action="store_true", help="Use BIC to select the most parsimonious MIQP threshold")
    p.add_argument("-p", "--nproc", type=int, default=1, metavar="<int>",
                   help="Number of processors to use.")

   # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
                   help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args

class Bfactor_options(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.pdb = None

class B_factor():
    def __init__(self, structure, options):
        self.structure = structure #PDB with HOH at the bottom
        self.options = options #user input
    def run(self):
        if not self.options.pdb==None:
            self.pdb=self.options.pdb+'_'
        else:
            self.pdb=''
        self.get_bfactors()

    def get_bfactors(self):
        B_factor = pd.DataFrame()
        atom_name = []
        chain_ser = []
        residue_name = []
        b_factor = []
        residue_num = []
        model_number = []
        select = self.structure.extract('record', 'ATOM', '==')
        n=0
        for chain in np.unique(select.chain):
            select2 = select.extract('chain', chain, '==')
            print('select 2:')
            print(select2)
            residues = set(list(select2.resi))
            residue_ids = []
            for i in residues:
                print(i)
                tmp_i = str(i)
                if ':' in tmp_i:
                    resi = int(tmp_i.split(':')[1][1:])
                else:
                    resi = tmp_i
                print(resi)
                residue_ids.append(resi)
        n=1
        for id in residue_ids:
            print(id)
            res_tmp = select2.extract('resi', int(id), '==') #this is seperating each residues
            print(res_tmp)
            #is this going to give us the alternative coordinate for everything?
            resn_name = (np.array2string(np.unique(res_tmp.resi)), np.array2string(np.unique(res_tmp.resn)),np.array2string(np.unique(res_tmp.chain)))
            #print(resn_name)
            b_factor = res_tmp.b
            #print(type(b_factor))
            B_factor.loc[n,'resseq'] = resn_name[0]
            B_factor.loc[n,'AA'] = resn_name[1]
            B_factor.loc[n,'Chain'] = resn_name[2]
            B_factor.loc[n,'Max_Bfactor'] = np.amax(b_factor)
            B_factor.loc[n, 'Averaage_Bfactor'] = np.average(b_factor)
            n+=1
        #print(B_factor)
        B_factor.to_csv(args.pdb_name+'_B_factors.csv', index=False)

def main():
    args = parse_args()
    try:
        os.mkdir(args.directory)
    except OSError:
        pass
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder() #put H20 on the bottom
    if not args.hydro:
        structure = structure.extract('e', 'H', '!=')
    B_options = Bfactor_options()
    B_options.apply_command_args(args)
    time0 = time.time()
    b_factor = B_factor(structure, B_options)
    test = b_factor.run()



'''
for model in Bio.PDB.PDBParser().get_structure(args.pdb_name, args.pdb):
    for chain in model.get_list():
        for residue in chain.get_list():
            for atom in residue.get_list():
                model_number.append(model)
                atom_name.append(atom.get_name())
                chain_ser.append(chain.get_id())
                residue_name.append(str(residue)[9:13])
                #print(str(residue)[9:13])
                residue_num.append(residue.get_full_id()[3][1])
                b_factor.append(atom.get_bfactor())
                n=+1
'''
