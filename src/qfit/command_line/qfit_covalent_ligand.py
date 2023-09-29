"""Hierarchically build a multiconformer ligand."""

import logging
import os.path
import os
import sys
import time
from string import ascii_uppercase

import numpy as np

from qfit.command_line.common_options import get_base_argparser
from qfit import MapScaler, Structure, XMap, CovalentLigand
from qfit import QFitCovalentLigand, QFitOptions

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def parse_args(argv):
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

    p.add_argument(
        "-z",
        "--scattering",
        choices=["xray", "electron"],
        default="xray",
        help="Scattering type.",
    )
    p.add_argument(
        "-rb",
        "--randomize-b",
        action="store_true",
        dest="randomize_b",
        help="Randomize B-factors of generated conformers.",
    )

    p.add_argument(
        "-nw",
        "--no-waters",
        action="store_true",
        dest="nowaters",
        help="Keep waters, but do not consider them for soft clash detection.",
    )

    # Sampling options
    p.add_argument(
        "-bb",
        "--no-backbone",
        dest="sample_backbone",
        action="store_false",
        help="Do not sample backbone using inverse kinematics.",
    )
    p.add_argument(
        "-bbs",
        "--backbone-step",
        dest="sample_backbone_step",
        type=float,
        default=0.1,
        metavar="<float>",
        help="Backbone sampling step (default = 0.1)",
    )
    p.add_argument(
        "-bba",
        "--backbone-amplitude",
        dest="sample_backbone_amplitude",
        type=float,
        default=0.3,
        metavar="<float>",
        help="Backbone sampling amplitude (default = 0.3)",
    )
    p.add_argument(
        "-sa",
        "--no-sample-angle",
        dest="sample_angle",
        action="store_false",
        help="Do not sample N-CA-CB angle.",
    )
    p.add_argument(
        "-sas",
        "--sample-angle-step",
        dest="sample_angle_step",
        type=float,
        default=3.75,
        metavar="<float>",
        help="Bond angle sampling step (default = 3.75)",
    )
    p.add_argument(
        "-sar",
        "--sample-angle-range",
        dest="sample_angle_range",
        type=float,
        default=7.5,
        metavar="<float>",
        help="Bond angle sampling range (default = 7.5)."
        "Sampling is carried out in the [-x,x] range, where x is"
        " determined by this sampling parameter.",
    )
    p.add_argument(
        "-rn",
        "--rotamer-neighborhood",
        type=float,
        default=80,
        metavar="<float>",
        help="Neighborhood of rotamer to sample in degree.",
    )
    p.add_argument(
        "-nl",
        "--no-ligand",
        dest="sample_ligand",
        action="store_false",
        help="Disable ligand sampling.",
    )
    p.add_argument(
        "-ls",
        "--sample-ligand-stepsize",
        type=float,
        default=10,
        metavar="<float>",
        dest="sample_ligand_stepsize",
        help="Stepsize for ligand sampling in degrees.",
    )
    p.add_argument(
        "-T",
        "--no-threshold-selection",
        dest="bic_threshold",
        action="store_false",
        help="Do not use BIC to select the most parsimonious MIQP threshold",
    )
    return p.parse_args(argv[1:])


def main(argv=sys.argv):
    args = parse_args(argv)
    os.makedirs(args.directory, exist_ok=True)
    time0 = time.time()

    # Setup logger
    logging_fname = os.path.join(args.directory, "qfit_covalent_ligand.log")
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(filename=logging_fname, level=level)
    logger.info(" ".join(argv))
    logger.info(time.strftime("%c %Z"))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(level)
        logging.getLogger("").addHandler(console_out)

    # Load structure and prepare it
    structure = Structure.fromfile(args.structure)
    if not args.hydro:
        structure = structure.extract("e", "H", "!=")

    logger.info("Extracting receptor and ligand from input structure.")
    chainid, resi = args.selection.split(",")
    if ":" in resi:
        resi, icode = resi.split(":")
    else:
        icode = ""

    # Extract the ligand:
    structure_ligand = structure.extract(f"resi {resi} and chain {chainid}")
    if icode:
        structure_ligand = structure_ligand.extract("icode", icode)

    # Select all ligand conformers:
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

    covalent_ligand = CovalentLigand.from_structure(structure_ligand,
                                                    args.cif_file)
    if covalent_ligand.natoms == 0:
        raise RuntimeError(
            "No atoms were selected for the ligand. Check the " "selection input."
        )
    covalent_ligand.altloc = ""
    covalent_ligand.q = 1

    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"
    receptor = structure.extract(sel_str)
    logger.info("Receptor atoms selected: {natoms}".format(natoms=receptor.natoms))

    options = QFitOptions()
    options.apply_command_args(args)

    # Load and process the electron density map:
    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, em=args.scattering == "electron")
        if args.omit:
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
        scaler.scale(footprint, radius=radius)
    xmap = xmap.extract(covalent_ligand.coor, padding=args.padding)
    ext = ".ccp4"
    if not np.allclose(xmap.origin, 0):
        ext = ".mrc"
    scaled_fname = os.path.join(args.directory, f"scaled{ext}")
    xmap.tofile(scaled_fname)

    qfit = QFitCovalentLigand(covalent_ligand, receptor, xmap, options)
    qfit.run()
    conformers = qfit.get_conformers_covalent()
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
    fname = os.path.join(options.directory, f"multiconformer_{chainid}_{resi}.pdb")
    if icode:
        fname = os.path.join(
            options.directory, f"multiconformer_{chainid}_{resi}_{icode}.pdb"
        )
    multiconformer.tofile(fname)

    passed = time.time() - time0
    logger.info(f"Time passed: {passed}s")
