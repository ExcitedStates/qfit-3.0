#Edited by Stephanie Wankowicz
#began: 2019-05-01
'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu
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
    p.add_argument('-bba', "--backbone-amplitude", dest="sample_backbone_amplitude",
                   type=float, default=0.3, metavar="<float>",
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

    #new RMSF arguments
    args = p.parse_args()
    return args

class RMSF_options(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.pdb = None



class RMSF():
    def __init__(self, structure, options):
        self.structure = structure #PDB with HOH at the bottom
        self.options = options #user input
    def run(self):
        if not self.options.pdb == None:
            self.pdb = self.options.pdb+'_'
        else:
            self.pdb = ''
        self.average_coor_heavy_atom()


    def average_coor_heavy_atom(self):
        rmsf = pd.DataFrame()
        select = self.structure.extract('record', 'ATOM', '==')
        n=0
        for chain in np.unique(select.chain):
            select2 = select.extract('chain', chain, '==')
            residues = set(list(select2.resi))
            residue_ids = []
            for i in residues:
                tmp_i = str(i)
                if ':' in tmp_i:
                    resi = int(tmp_i.split(':')[1][1:])
                else:
                    resi = tmp_i
                residue_ids.append(resi)
            for id in residue_ids:
                res_tmp=select2.extract('resi', int(id), '==') #this is seperating each residues
                resn_name=(np.array2string(np.unique(res_tmp.resi)), np.array2string(np.unique(res_tmp.resn)),np.array2string(np.unique(res_tmp.chain)))
                if len(np.unique(res_tmp.altloc))>1:
                    RMSF_list=[]
                    num_alt=len(np.unique(res_tmp.altloc))
                    #now iterating over each atom and getting center for each atom
                    for atom in np.unique(res_tmp.name):
                        RMSF_atom_list=[]
                        tmp_atom=res_tmp.extract('name', atom, '==')
                        atom_center=tmp_atom.coor.mean(axis=0)
                        for i in np.unique(tmp_atom.altloc):
                            atom_alt=tmp_atom.extract('altloc', i, '==')
                            RMSF_atom=np.linalg.norm(atom_alt.coor-atom_center, axis=1)
                            RMSF_atom_list.append(RMSF_atom)
                        RMSF=(sum(RMSF_atom_list)/len(RMSF_atom_list))
                        RMSF_list.append(RMSF)
                        rmsf.loc[n,'resseq']=resn_name[0]
                        rmsf.loc[n,'AA']=resn_name[1]
                        rmsf.loc[n,'Chain']=resn_name[2]
                        rmsf.loc[n,'rmsf']=(sum(RMSF_list)/ len(RMSF_list))
                else:
                    rmsf.loc[n,'resseq']=resn_name[0]
                    rmsf.loc[n,'AA']=resn_name[1]
                    rmsf.loc[n,'Chain']=resn_name[2]
                    rmsf.loc[n,'rmsf']=0
                n+=1
        rmsf['resseq'] = rmsf['resseq'].str.replace('[', '')
        rmsf['resseq'] = rmsf['resseq'].str.replace(']', '')
        rmsf['AA'] = rmsf['AA'].str.replace('[', '')
        rmsf['AA'] = rmsf['AA'].str.replace(']', '')
        rmsf['AA'] = rmsf['AA'].str.replace('\'', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace('[', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace(']', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace('\'', '')
        rmsf['PDB_name']=self.options.pdb
        print(rmsf)
        rmsf.to_csv(self.pdb+'qfit_RMSF.csv')


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
    R_options = RMSF_options()
    R_options.apply_command_args(args)

    time0 = time.time()
    rmsf = RMSF(structure, R_options)
    test = rmsf.run()
