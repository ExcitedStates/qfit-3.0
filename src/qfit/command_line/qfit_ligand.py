"""Hierarchically build a multiconformer ligand."""

import logging
import os.path
import os
import time
from string import ascii_uppercase

import numpy as np

from qfit.command_line.common_options import get_base_argparser
from qfit.command_line.custom_argparsers import ToggleActionFlag
from qfit import MapScaler, Structure, XMap, Ligand
from qfit.qfit import QFitLigand, QFitOptions
from qfit.logtools import setup_logging, log_run_info

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def build_argparser():
    p = get_base_argparser(__doc__)
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

    # Sampling options
    p.add_argument(
        "--build",
        action=ToggleActionFlag,
        dest="build",
        default=True,
        help="Build ligand",
    )
    p.add_argument(
        "--local",
        action=ToggleActionFlag,
        dest="local_search",
        default=True,
        help="Perform a local search",
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
        "--threshold-selection",
        dest="bic_threshold",
        action=ToggleActionFlag,
        default=False,
        help="Use BIC to select the most parsimonious MIQP threshold",
    )
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
        ligand = Ligand(
            structure_ligand.data,
            structure_ligand.selection,
            link_data=structure_ligand.link_data,
            cif_file=options.cif_file,
        )
    else:
        ligand = Ligand(
            structure_ligand.data,
            structure_ligand.selection,
            link_data=structure_ligand.link_data,
        )
    if ligand.natoms == 0:
        raise RuntimeError(
            "No atoms were selected for the ligand. Check " " the selection input."
        )

    ligand.altloc = ""
    ligand.q = 1

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

    return QFitLigand(ligand, structure, xmap, options), chainid, resi, icode, receptor


def main():
    p = build_argparser()
    args = p.parse_args()
    os.makedirs(args.directory, exist_ok=True)
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
    pdb_ext = qfit_ligand.file_ext
    for n, conformer in enumerate(conformers, start=0):
        if nconformers > 1:
            altloc = ascii_uppercase[n]
        conformer.altloc = ""
        fname = os.path.join(options.directory, f"conformer_{n}.{pdb_ext}")
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
        options.directory, f"{pdb_id}multiconformer_ligand_bound_with_protein.pdb"
    )
    if icode:
        fname = os.path.join(
            options.directory, f"{pdb_id}multiconformer_ligand_bound_with_protein.pdb"
        )
    try:
        multiconformer_ligand_bound.tofile(fname)
    except NameError:
        logger.error("qFit-ligand failed to produce any valid conformers.")
