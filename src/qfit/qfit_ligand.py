"""Hierarchically build a multiconformer ligand."""

import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import os.path
import os
import sys
import time
import numpy as np
from string import ascii_uppercase
from . import MapScaler, Structure, XMap, _Ligand
from .qfit import QFitLigand, QFitOptions
from .logtools import setup_logging, log_run_info
from .solvers import available_qp_solvers, available_miqp_solvers


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
    p.add_argument("structure", type=str, help="PDB-file containing structure.")

    p.add_argument(
        "-cif",
        "--cif_file",
        type=str,
        default=None,
        help="CIF file describing the ligand",
    )
    p.add_argument(
        "selection",
        type=str,
        help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.",
    )
    
    # RDKit input options
    p.add_argument(
        "-sm",
        "--smiles",
        type=str,
        help="SMILES string for molecule",
    )
    p.add_argument(
        "-nc",
        "--numConf",
        type=int,
        default=10000,
        help="Number of RDKit conformers to generate",
    )

    p.add_argument(
        "-lb",
        "--ligand_bic",
        action="store_true",
        help="Flag to run with ligand BIC on",
    )

    p.add_argument(
        "-rr",
        "--rot_range",
        type=float,
        default=15.0,
        help="Rotation range for RDKit conformers",
    )
    p.add_argument(
        "-tr",
        "--trans_range",
        type=float,
        default=0.3,
        help="Translation range for RDKit conformers",
    )

    p.add_argument(
        "-rs",
        "--rotation_step",
        type=float,
        default=5.0,
        help="Rotation step size for RDKit conformers",
    )

    # Map input options
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
        default=1.0,
        metavar="<float>",
        type=float,
        help="Scaling factor for soft-clash mask radius",
    )
    p.add_argument(
        "-dc",
        "--density-cutoff",
        default=0.3,
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
        "--remove-conformers-below-cutoff",
        action="store_true",
        dest="remove_conformers_below_cutoff",
        help=(
            "Remove conformers during sampling that have atoms "
            "with no density support, i.e. atoms are positioned "
            "at density values below <density-cutoff>"
        ),
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
        "-ec",
        "--no-external-clash",
        action="store_false",
        dest="external_clash",
        help="Turn off external clash detection during sampling",
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
        "-c",
        "--cardinality",
        default=5,
        metavar="<int>",
        type=int,
        help="Cardinality constraint used during MIQP",
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
        "-ic",
        "--intermediate-cardinality",
        default=5,
        metavar="<int>",
        type=int,
        help="Cardinality constraint used during intermediate MIQP",
    )
    p.add_argument(
        "-it",
        "--intermediate-threshold",
        default=0.01,
        metavar="<float>",
        type=float,
        help="Threshold constraint during intermediate MIQP",
    )
    p.add_argument(
        "-hy",
        "--hydro",
        action="store_true",
        dest="hydro",
        help="Include hydrogens during calculations",
    )
    p.add_argument(
        "-rmsd",
        "--rmsd-cutoff",
        default=0.01,
        metavar="<float>",
        type=float,
        help="RMSD cutoff for removal of identical conformers",
    )
    p.add_argument(
        "--threshold-selection",
        dest="bic_threshold",
        action=ToggleActionFlag,
        default=False,
        help="Use BIC to select the most parsimonious MIQP threshold",
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
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Be verbose"
    )
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


def prepare_qfit_ligand(options):
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

    if icode:
        structure_ligand = structure_ligand.extract("icode", icode)
    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"

    receptor = structure.extract(
        sel_str
    )  # selecting everything that is no the ligand of interest

    # Check which altlocs are present in the ligand. If none, take the
    # A-conformer as default.

    altlocs = sorted(list(set(structure_ligand.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove("")
        except ValueError:
            pass
        for altloc in altlocs[1:]:
            sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
            sel_str = f"not ({sel_str})"
            structure_ligand = structure_ligand.extract(sel_str)
    altloc = structure_ligand.altloc[-1]

    if options.cif_file:
        ligand = _Ligand(
            structure_ligand.data,
            structure_ligand._selection,
            link_data=structure_ligand.link_data,
            cif_file=args.cif_file,
        )
    else:
        ligand = _Ligand(
            structure_ligand.data,
            structure_ligand._selection,
            link_data=structure_ligand.link_data,
        )
    if ligand.natoms == 0:
        raise RuntimeError(
            "No atoms were selected for the ligand. Check " " the selection input."
        )

    ligand.altloc = ""
    ligand.q = 1

    # save ligand pdb file to working directory 
    try:
        os.makedirs(options.directory)
    except OSError:
        pass
        
    input_ligand = os.path.join(
        options.directory, "ligand.pdb"
    )
    ligand.tofile(input_ligand)

    logger.info("Ligand atoms selected: {natoms}".format(natoms=ligand.natoms))

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
    ext = ".ccp4"

    if not np.allclose(xmap.origin, 0):
        ext = ".mrc"
    scaled_fname = os.path.join(
        options.directory, f"scaled{ext}"
    )  # this should be an option
    xmap.tofile(scaled_fname)

    return QFitLigand(ligand, receptor, xmap, options), chainid, resi, icode, receptor


def main():
    p = build_argparser()
    args = p.parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass
    if not args.pdb == None:
        pdb_id = args.pdb + "_"
    else:
        pdb_id = ""
    time0 = time.time()

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options, filename="qfit_ligand.log")
    log_run_info(options, logger)

    qfit_ligand, chainid, resi, icode, receptor = prepare_qfit_ligand(options=options)

    time0 = time.time()
    qfit_ligand.run()
    logger.info(f"Total time: {time.time() - time0}s")

    # POST QFIT LIGAND WRITE OUTPUT
    conformers = qfit_ligand.get_conformers()
    nconformers = len(conformers)
    altloc = ""
    for n, conformer in enumerate(conformers, start=0):
        if nconformers > 1:
            altloc = ascii_uppercase[n]
        conformer.altloc = ""
        fname = os.path.join(options.directory, f"conformer_{n}.pdb")
        conformer.tofile(fname)
        conformer.altloc = altloc
        try:
            multiconformer_ligand_bound = multiconformer_ligand_bound.combine(conformer)
        except NameError:
            # First time through, multiconformer_ligand_bound does not exist, so we fall back on this
            multiconformer_ligand_bound = Structure.fromstructurelike(conformer.copy())

    # Print multiconformer_ligand_only as an output file
    multiconformer_ligand_only = os.path.join(
        options.directory, "multiconformer_ligand_only.pdb"
    )
    multiconformer_ligand_bound.tofile(multiconformer_ligand_only)

    # Stitch back protein and other HETATM to the multiconformer ligand output
    multiconformer_ligand_bound = receptor.combine(multiconformer_ligand_bound)
    fname = os.path.join(
        options.directory, pdb_id + f"multiconformer_ligand_bound_with_protein.pdb"
    )
    if icode:
        fname = os.path.join(
            options.directory, pdb_id + f"multiconformer_ligand_bound_with_protein.pdb"
        )
    try:
        multiconformer_ligand_bound.tofile(fname)
    except NameError:
        logger.error("qFit-ligand failed to produce any valid conformers.")
