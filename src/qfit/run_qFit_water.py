import gc
from .qfit_water import QFitWaterOptions, QFitWater
import multiprocessing as mp
from tqdm import tqdm
import os.path
import os
import sys
import itertools
import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import traceback
import numpy as np
import time
import math
import copy
from string import ascii_uppercase
from scipy.optimize import least_squares

from .clash import ClashDetector
from .logtools import (
    setup_logging,
    log_run_info,
    poolworker_setup_logging,
    QueueListener,
)
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .structure.residue import _RotamerResidue
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter, description=__doc__
    )
    p.add_argument(
        "map",
        type=str,
        help="Density map in CCP4 or MRC format, or an MTZ file "
        "containing reflections and phases. For MTZ files "
        "use the --label options to specify columns to read.",
    )
    p.add_argument("structure", help="PDB-file containing multiconformer structure.")

    # Map input options
    p.add_argument(
        "-em",
        "--cryo_em",
        action="store_true",
        dest="em",
        help="Run qFit with EM options",
    )
    p.add_argument(
        "-l",
        "--label",
        default="FWT,PHWT",
        metavar="<F,PHI>",
        help="MTZ column labels to build density",
    )
    p.add_argument(
        "-r",
        "--resolution",
        default=None,
        metavar="<float>",
        type=float,
        help="Map resolution (Å) (only use when providing CCP4 map files)",
    )
    p.add_argument(
        "-m",
        "--resolution-min",
        default=None,
        metavar="<float>",
        type=float,
        help="Lower resolution bound (Å) (only use when providing CCP4 map files)",
    )
    p.add_argument(
        "-o",
        "--omit",
        action="store_true",
        help="Treat map file as an OMIT map in map scaling routines",
    )

    # Map prep options
    p.add_argument(
        "--scale",
        action=ToggleActionFlag,
        dest="scale",
        default=True,
        help="Scale density",
    )
    p.add_argument(
        "-sv",
        "--scale-rmask",
        dest="scale_rmask",
        default=0.8,
        metavar="<float>",
        type=float,
        help="Scaling factor for soft-clash mask radius",
    )
    p.add_argument(
        "-dc",
        "--density-cutoff",
        default=0.2,
        metavar="<float>",
        type=float,
        help="Density values below this value are set to <density-cutoff-value>",
    )
    p.add_argument(
        "-dv",
        "--density-cutoff-value",
        default=-1,
        metavar="<float>",
        type=float,
        help="Density values below <density-cutoff> are set to this value",
    )
    p.add_argument(
        "--subtract",
        action=ToggleActionFlag,
        dest="subtract",
        default=True,
        help="Subtract Fcalc of neighboring residues when running qFit",
    )
    p.add_argument(
        "--padding",
        default=10.0,
        metavar="<float>",
        type=float,
        help="Padding size for map creation",
    )

    # arguments
    p.add_argument(
        "-cf",
        "--clash-scaling-factor",
        default=0.7,
        metavar="<float>",
        type=float,
        help="Set clash scaling factor",
    )
    p.add_argument(
        "-ec",
        "--external-clash",
        action="store_true",
        dest="external_clash",
        help="Enable external clash detection during sampling",
    )
    p.add_argument(
        "-bs",
        "--bulk-solvent-level",
        default=0.0,
        metavar="<float>",
        type=float,
        help="Bulk solvent level in absolute values",
    )
    p.add_argument(
        "-c",
        "--cardinality",
        default=10,
        metavar="<int>",
        type=int,
        help="Cardinality constraint used during MIQP",
    )
    p.add_argument(
        "-t",
        "--threshold",
        default=0.1,
        metavar="<float>",
        type=float,
        help="Threshold constraint used during MIQP",
    )
    p.add_argument(
        "-hy",
        "--hydro",
        action="store_true",
        dest="hydro",
        help="Include hydrogens during calculations",
    )
    p.add_argument(
        "-p",
        "--nproc",
        type=int,
        default=1,
        metavar="<int>",
        help="Number of processors to use",
    )

    # Output options
    p.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    p.add_argument(
        "--debug", action="store_true", help="Log as much information as possible"
    )
    p.add_argument(
        "--write-intermediate-conformers",
        action="store_true",
        help="Write intermediate structures to file (useful with debugging)",
    )
    p.add_argument("--pdb", help="Name of the input PDB")

    return p

def main():
    p = build_argparser()
    args = p.parse_args(args=None)
    try:
        os.makedirs(args.directory)
    except OSError:
        pass
    time0 = time.time()

    # Apply the arguments to options
    options = QFitWaterOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    if not args.hydro:
        structure = structure.extract("e", "H", "!=")

    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if not args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, em=options.em)
        if args.omit:
            footprint = structure_resi
        else:
            footprint = structure
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        scaler.scale(footprint, radius=args.scale_rmask * radius)
    
    full_occ = structure.extract("resn", "HOH", "!=")

    residues = list(
            structure.extract("record", "HETATM", "!=")
            .extract("resn", "HOH", "!=")
            .single_conformer_residues
        )

    for residue in residues:
        xmap_reduced = xmap.extract(residue.coor, padding=options.padding)
        qfit = QFitWater(residue, full_occ, xmap_reduced, options)
        try:
            qfit.run()
        except RuntimeError:
            print(f"RuntimeError occurred for residue {residue}, skipping to next residue.")
            continue
        conformers = qfit.get_water_conformers()
        nconformers = len(conformers)
        altloc = ""
        for n, conformer in enumerate(conformers, start=0):
            if nconformers > 1:
                altloc = ascii_uppercase[n]
            conformer.altloc = ""
            conformer.altloc = altloc
            try:
                multiconformer = multiconformer.combine(conformer)
            except Exception:
                multiconformer = Structure.fromstructurelike(conformer.copy())
        fname = f'{residue.resi[0]}_{residue.chain[0]}_qFit_water.pdb'
        multiconformer.tofile(fname)
        del xmap_reduced
        del qfit

            
        # Now that all the individual residues have run...
        # Combine all multiconformer residues into one structure

        # for res, chain in unique_residue_chain_pairs:
        #     directory = os.path.join(self.options.directory)
        #     fname = os.path.join(directory, f"{chain}_{res}_resi_waternew.pdb")
        #     # if not os.path.exists(fname): continue
        #     residue_multiconformer = Structure.fromfile(fname)
        #     for water in residue_multiconformer.extract("resn", "HOH", "==").resi:
        #         residue_multiconformer.extract("resi", water, "==").resi = n
        #         n += 1
        #     try:
        #         multiconformer = multiconformer.combine(residue_multiconformer)
        #     except:
        #         multiconformer = residue_multiconformer

        # fname = os.path.join(self.options.directory, "multiconformer_model_water.pdb")
        # multiconformer = multiconformer.reorder()
        # multiconformer.tofile(fname, self.structure.scale, self.structure.cryst_info)
