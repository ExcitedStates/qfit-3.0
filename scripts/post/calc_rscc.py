#!/usr/bin/env python
import argparse
from qfit.custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import os.path
import os
import numpy as np
from qfit.scaler import MapScaler
from qfit.structure import Structure
from qfit.volume import XMap
from qfit.structure.ligand import _Ligand
from qfit.qfit import QFitLigand, QFitOptions, _BaseQFit
from qfit.solvers import available_qp_solvers, available_miqp_solvers, get_miqp_solver_class

"""
Compute the RSCC (Real-Space Correlation Coefficient) of a ligand model using a density map.

This script supports two use cases:
1. Calculate the RSCC of a single model of interest (e.g., a multiconformer model from qFit-ligand).
2. Compare the RSCC of two models, e.g. a multiconformer qFit-ligand model and a deposited single-conformer model, 
   by evaluating them against the same density map in the same voxel space.

Usage:
------

To compute RSCC for one model:
    calc_rscc.py MAP_FILE MODEL.pdb CHAIN,RESIDUE_ID

Example:
    calc_rscc.py composite_omit_map.mtz multiconformer_ligand_bound_with_protein.pdb A,401 -l 2FOFCWT,PH2FOFCWT

To compare RSCC between two models (e.g. qFit-ligand vs deposited):
    calc_rscc.py MAP_FILE MODEL.pdb CHAIN,RESIDUE_ID -comp COMPARISON_MODEL.pdb

Example:
    calc_rscc.py composite_omit_map.mtz multiconformer_ligand_bound_with_protein.pdb A,401 -l 2FOFCWT,PH2FOFCWT -comp 8H2F_001.pdb
"""

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
    p.add_argument("structure", 
                   type=str, 
                   help="Model of interest for the RSCC calculation. PDB-file containing structure.")
    
    p.add_argument(
        "selection",
        type=str,
        help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.",
    )

    # Map input options
    p.add_argument("--compare_structure", 
        "-comp",
        type=str, 
        default=None,
        help="Model to compare against the model of interest for RSCC calcualtion. PDB-file containing structure.")

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
        "--subtract",
        action=ToggleActionFlag,
        dest="subtract",
        default=True,
        help="Subtract Fcalc of neighboring residues when running qFit",
    )
    
    p.add_argument(
        "-pad",
        "--padding",
        default=8.0,
        metavar="<float>",
        type=float,
        help="Padding size for map creation",
    )
    p.add_argument(
        "--waters-clash",
        action=ToggleActionFlag,
        dest="waters_clash",
        default=True,
        help="Consider waters for soft clash detection",
    )

    p.add_argument(
        "-cf",
        "--clash-scaling-factor",
        default=0.75,
        metavar="<float>",
        type=float,
        help="Set clash scaling factor",
    )

    p.add_argument(
        "-bs",
        "--bulk-solvent-level",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Bulk solvent level in absolute values",
    )

    p.add_argument(
        "-t",
        "--threshold",
        default=0.2,
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

    # Solver options
    p.add_argument(
        "--qp-solver",
        dest="qp_solver",
        choices=available_qp_solvers.keys(),
        default=next(iter(available_qp_solvers.keys())),
        help="Select the QP solver",
    )
    p.add_argument(
        "--miqp-solver",
        dest="miqp_solver",
        choices=available_miqp_solvers.keys(),
        default=next(iter(available_miqp_solvers.keys())),
        help="Select the MIQP solver",
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
    return p


def setup_for_calculation(options):
    """Loads files to build a QFitLigand job."""

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure)

    if not options.hydro:
        structure = structure.extract("e", "H", "!=")

    chainid, resi = options.selection.split(",")
    if ":" in resi:
        resi, icode = resi.split(":")
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ""

    # Extract the ligand:
    structure_ligand = structure.extract(f"resi {resi} and chain {chainid}")

    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"

    receptor = structure.extract(
        sel_str
    )  # selecting everything that is no the ligand of interest

    # Get all non-empty altlocs in the ligand
    altlocs = sorted(set(structure_ligand.altloc))
    non_empty_altlocs = [a for a in altlocs if a != ""]

    if not non_empty_altlocs: # if input ligand is a single conformer model
        all_confs = [structure_ligand.coor]  
        base_conf = structure_ligand
    else: # if input ligand is a multiconformer model
        all_confs = []
        for alt in non_empty_altlocs:
            conf = structure_ligand.extract("altloc", alt)
            all_confs.append(conf.coor)
        base_conf = structure_ligand.extract("altloc", non_empty_altlocs[0])

    all_confs = np.stack(all_confs)

    # Build the ligand
    ligand = _Ligand(
        base_conf.data,
        base_conf._selection,
        link_data=base_conf.link_data,
    )
    ligand.altloc = ""
    ligand.q = 1

    # do the same ligand extraction procedure if a comparision strucure is included at the command line
    if options.compare_structure:
        # Load structure and prepare it
        compare_structure = Structure.fromfile(options.compare_structure)

        if not options.hydro:
            compare_structure = compare_structure.extract("e", "H", "!=")

        chainid, resi = options.selection.split(",")
        if ":" in resi:
            resi, icode = resi.split(":")
            residue_id = (int(resi), icode)
        else:
            residue_id = int(resi)
            icode = ""

        # Extract the ligand:
        compare_structure_ligand = compare_structure.extract(f"resi {resi} and chain {chainid}")

        sel_str = f"resi {resi} and chain {chainid}"
        sel_str = f"not ({sel_str})"

        receptor = compare_structure.extract(
            sel_str
        )  # selecting everything that is no the ligand of interest

        # Get all non-empty altlocs in the ligand
        compare_altlocs = sorted(set(compare_structure_ligand.altloc))
        non_empty_altlocs = [a for a in compare_altlocs if a != ""]

        if not non_empty_altlocs: # if input ligand is a single conformer model
            compare_all_confs = [compare_structure_ligand.coor]  
            compare_base_conf = compare_structure_ligand
        else: # if input ligand is a multiconformer model
            compare_all_confs = []
            for alt in non_empty_altlocs:
                compare_conf = compare_structure_ligand.extract("altloc", alt)
                compare_all_confs.append(compare_conf.coor)
            compare_base_conf = compare_structure_ligand.extract("altloc", non_empty_altlocs[0])

        compare_all_confs = np.stack(compare_all_confs)

        # Build the comparision ligand
        compare_ligand = _Ligand(
            compare_base_conf.data,
            compare_base_conf._selection,
            link_data=compare_base_conf.link_data,
        )

        compare_ligand.altloc = ""
        compare_ligand.q = 1

    # Load and process the electron density map:
    xmap = XMap.fromfile(
        options.map, resolution=options.resolution, label=options.label
    )
    xmap = xmap.canonical_unit_cell()
    if options.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, em=options.em)
        if options.omit:
            footprint = structure_ligand
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
        scaler.scale(footprint, radius=options.scale_rmask * radius)

    xmap = xmap.extract(ligand.coor, padding=options.padding)

    # qfit.py has necessary functions for map processing, create instance 
    qfit = QFitLigand.__new__(QFitLigand)
    _BaseQFit.__init__(qfit, ligand, receptor, xmap, options) 

    if options.compare_structure:
        qfit._starting_coor_set = compare_all_confs
        qfit._starting_bs = [compare_ligand.b for _ in compare_altlocs]

    qfit._coor_set = all_confs  
    qfit._bs = [ligand.b for _ in altlocs]  

    # select protein atoms whose density we are going to subtract from the experimental map. This gives you a 'cleaner' map just from the ligand contribution  
    if options.subtract:
        qfit._subtract_transformer(ligand, receptor)
    qfit._update_transformer(ligand)

    return qfit

def main():
    p = build_argparser()
    args = p.parse_args()

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)
    options.compare_structure = args.compare_structure


    qfit_ = setup_for_calculation(options=options)

    qfit_._transformer.reset(full=True)
    for n, coor in enumerate(qfit_._coor_set): # this is the coordinate set of the input PDB file
        qfit_.conformer.coor = coor
        qfit_._transformer.mask(qfit_._rmask)
    mask = qfit_._transformer.xmap.array > 0 # mask around the total density converage for the conformers in the input PDB file 
    qfit_._transformer.reset(full=True)

    nvalues = mask.sum()
    qfit_._target = qfit_.xmap.array[mask] # values of the target map (experimental map) at locations where the conformers exist 
    
    # convert the input PDB conformers into density for the RSCC calculation
    nmodels = len(qfit_._coor_set)
    qfit_._models = np.zeros((nmodels, nvalues), float)
    for n, coor in enumerate(qfit_._coor_set):
        qfit_.conformer.coor = coor
        qfit_.conformer.b = qfit_._bs[n]
        qfit_._transformer.density()
        model = qfit_._models[n]
        model[:] = qfit_._transformer.xmap.array[mask]
        np.maximum(model, qfit_.options.bulk_solvent_level, out=model)
        qfit_._transformer.reset(full=True)    

    # convert the comparision ligand into density for the RSCC calulation 
    if options.compare_structure:
        in_model = len(qfit_._starting_coor_set)
        qfit_._in_model = np.zeros((in_model, nvalues), float)
        for n, coor in enumerate(qfit_._starting_coor_set): 
            qfit_.conformer.coor = coor
            qfit_.conformer.b = qfit_._starting_bs[n]
            qfit_._transformer.density() 

            input_model = qfit_._in_model[n]
            input_model[:] = qfit_._transformer.xmap.array[mask]
            np.maximum(input_model, qfit_.options.bulk_solvent_level, out=input_model)
            qfit_._transformer.reset(full=True)

    
        miqp_solver_class = get_miqp_solver_class(qfit_.options.miqp_solver)
        solver = miqp_solver_class(qfit_._target, qfit_._models, qfit_._in_model)
    else:
        miqp_solver_class = get_miqp_solver_class(qfit_.options.miqp_solver)
        solver = miqp_solver_class(qfit_._target, qfit_._models)

    # Run solver 
    solver.rscc_solve_miqp(cardinality=options.cardinality, threshold=options.threshold)

    # Update occupancies from solver weights
    qfit_._occupancies = solver.weights

if __name__ == "__main__":
    main()
