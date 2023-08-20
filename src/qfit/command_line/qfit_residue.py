"""Automatically build a multiconformer residue."""

import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase

import numpy as np

from qfit import MapScaler, Structure, XMap
from qfit.command_line.common_options import get_base_argparser
from qfit.command_line.custom_argparsers import ToggleActionFlag
from qfit.logtools import setup_logging, log_run_info
from qfit.qfit import (QFitOptions, QFitRotamericResidue)
from qfit.structure import residue_type

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def build_argparser():
    p = get_base_argparser(__doc__)
    p.add_argument(
        "selection",
        type=str,
        help="Chain, residue id, and optionally insertion code for "
        "residue in structure, e.g. A,105, or A,105:A.",
    )

    p.add_argument(
        "-em",
        "--cryo_em",
        action="store_true",
        dest="em",
        help="Run qFit with EM options",
    )

    p.add_argument(
        "-rb",
        "--randomize-b",
        action="store_true",
        dest="randomize_b",
        help="Randomize B-factors of generated conformers",
    )
    p.add_argument(
        "-par",
        "--phenix-aniso",
        action="store_true",
        dest="phenix_aniso",
        help="Use phenix to perform anisotropic refinement of individual sites."
        "This option creates an OMIT map and uses it as a default.",
    )

    # Sampling options
    p.add_argument(
        "--backbone",
        action=ToggleActionFlag,
        dest="sample_backbone",
        default=True,
        help="Sample backbone using inverse kinematics",
    )
    p.add_argument(
        "-bbs",
        "--backbone-step",
        default=0.1,
        dest="sample_backbone_step",
        metavar="<float>",
        type=float,
        help="Stepsize for the amplitude of backbone sampling (Å)",
    )
    p.add_argument(
        "-bba",
        "--backbone-amplitude",
        default=0.3,
        dest="sample_backbone_amplitude",
        metavar="<float>",
        type=float,
        help="Maximum backbone amplitude (Å)",
    )
    p.add_argument(
        "-bbv",
        "--backbone-sigma",
        default=0.125,
        dest="sample_backbone_sigma",
        metavar="<float>",
        type=float,
        help="Backbone random-sampling displacement (Å)",
    )
    p.add_argument(
        "--sample-angle",
        action=ToggleActionFlag,
        dest="sample_angle",
        default=True,
        help="Sample CA-CB-CG angle for aromatic F/H/W/Y residues",
    )
    p.add_argument(
        "-sas",
        "--sample-angle-step",
        default=3.75,
        dest="sample_angle_step",
        metavar="<float>",
        type=float,
        help="CA-CB-CG bond angle sampling step in degrees",
    )
    p.add_argument(
        "-sar",
        "--sample-angle-range",
        default=7.5,
        dest="sample_angle_range",
        metavar="<float>",
        type=float,
        help="CA-CB-CG bond angle sampling range in degrees [-x,x]",
    )
    p.add_argument(
        "--sample-rotamers",
        action=ToggleActionFlag,
        dest="sample_rotamers",
        default=True,
        help="Sample sidechain rotamers",
    )
    p.add_argument(
        "-rn",
        "--rotamer-neighborhood",
        default=60,
        metavar="<float>",
        type=float,
        help="Chi dihedral-angle sampling range around each rotamer in degrees [-x,x]",
    )
    p.add_argument(
        "--threshold-selection",
        dest="bic_threshold",
        action=ToggleActionFlag,
        default=True,
        help="Use BIC to select the most parsimonious MIQP threshold",
    )
    return p


def main(argv=sys.argv):
    p = build_argparser()
    args = p.parse_args(argv[1:])
    os.makedirs(args.directory, exist_ok=True)
    time0 = time.time()

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Skip over if everything is completed
    # try:
    if os.path.isfile(args.directory + "/multiconformer_residue.pdb"):
        print("This residue has completed")
        exit()
    else:
        print("Beginning qfit_residue")

    # Setup logger
    logging_fname = os.path.join(args.directory, "qfit_residue.log")
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(filename=logging_fname, level=level)
    logger.info(" ".join(sys.argv))
    logger.info(time.strftime("%c %Z"))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(level)
        logging.getLogger("").addHandler(console_out)

    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    if not args.hydro:
        structure = structure.extract("e", "H", "!=")
    chainid, resi = args.selection.split(",")
    if ":" in resi:
        resi, icode = resi.split(":")
        residue_id = (int(resi), icode)
    elif "_" in resi:
        resi, icode = resi.split("_")
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ""

    # Extract the residue:
    structure_resi = structure.extract(f"resi {resi} and chain {chainid}")
    if icode:
        structure_resi = structure_resi.extract("icode", icode)

    chain = structure_resi[chainid]
    conformer = chain.conformers[0]
    residue = conformer[residue_id]
    rtype = residue_type(residue)
    if rtype != "rotamer-residue":
        logger.info("Residue has no known rotamers. Stopping qfit_residue.")
        sys.exit()
    # Check which altlocs are present in the residue. If none, take the
    # A-conformer as default.
    altlocs = sorted(list(set(residue.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove("")
        except ValueError:
            pass

        # If more than 1 conformer were included, we want to select the
        # most complete conformer. If more than one conformers are complete,
        # Select the one with the highest occupancy:
        longest_conf = 0
        best_q = -1
        for i, altloc in enumerate(altlocs):
            conformer = structure_resi.extract("altloc", ("", altloc))
            if len(conformer.name) > longest_conf:
                idx = i
                longest_conf = len(conformer.name)
            elif len(conformer.name) == longest_conf:
                if conformer.q[0] > best_q:
                    idx = i
                    best_q = conformer.q[0]
        # Delete all the unwanted conformers:
        for altloc in altlocs:
            if altloc != altlocs[idx]:
                sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                sel_str = f"not ({sel_str})"
                structure = structure.extract(sel_str)
    residue_name = residue.resn[0]
    logger.info(f"Residue: {residue_name} {chainid}_{resi}{icode}")

    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
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
    xmap = xmap.extract(residue.coor, padding=args.padding)
    ext = ".ccp4"
    if not np.allclose(xmap.origin, 0):
        ext = ".mrc"
    scaled_fname = os.path.join(args.directory, f"scaled{ext}")
    xmap.tofile(scaled_fname)
    qfit = QFitRotamericResidue(residue, structure, xmap, options)
    qfit.run()
    conformers = qfit.get_conformers()
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
            multiconformer = multiconformer.combine(conformer)
        except Exception:
            multiconformer = Structure.fromstructurelike(conformer.copy())
    multiconformer.normalize_occupancy()  # normalize the occupancy of each conformation
    fname = os.path.join(options.directory, f"multiconformer_{chainid}_{resi}.pdb")
    if icode:
        fname = os.path.join(
            options.directory, f"multiconformer_{chainid}_{resi}_{icode}.pdb"
        )
    multiconformer.tofile(fname)

    passed = time.time() - time0
    logger.info(f"Time passed: {passed}s")
