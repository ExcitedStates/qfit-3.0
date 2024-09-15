"""Common argparse setup and related CLI utility methods"""

import argparse
import logging
import os.path

from qfit.command_line.custom_argparsers import (
    ToggleActionFlag,
    CustomHelpFormatter,
    ValidateMapFileArgument,
    ValidateStructureFileArgument,
)
from qfit.solvers import available_qp_solvers, available_miqp_solvers
from qfit import MapScaler, XMap

logger = logging.getLogger(__name__)


def get_base_argparser(description,
                       default_enable_external_clash=False,
                       default_transformer="cctbx"):
    p = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter, description=description
    )

    p.add_argument(
        "map",
        help="Density map in CCP4 or MRC format, or an MTZ file "
        "containing reflections and phases. For MTZ files "
        "use the --label options to specify columns to read. "
        "For CCP4 files, use the -r to specify resolution.",
        type=str,
        action=ValidateMapFileArgument,
    )
    p.add_argument(
        "structure",
        help="PDB or mmCIF file containing structure.",
        type=str,
        action=ValidateStructureFileArgument,
    )

    # Map input options
    mo = p.add_argument_group("Map options")
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
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
    mo.add_argument(
        "-o",
        "--omit",
        action="store_true",
        help="Treat map file as an OMIT map in map scaling routines",
    )

    # Map prep options
    mo.add_argument(
        "--scale",
        action=ToggleActionFlag,
        dest="scale",
        default=True,
        help="Scale density",
    )
    mo.add_argument(
        "-sv",
        "--scale-rmask",
        dest="scale_rmask",
        default=1.0,
        metavar="<float>",
        type=float,
        help="Scaling factor for soft-clash mask radius",
    )
    mo.add_argument(
        "-dc",
        "--density-cutoff",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Density values below this value are set to <density-cutoff-value>",
    )
    mo.add_argument(
        "-dv",
        "--density-cutoff-value",
        default=-1,
        metavar="<float>",
        type=float,
        help="Density values below <density-cutoff> are set to this value",
    )
    mo.add_argument(
        "--subtract",
        action=ToggleActionFlag,
        dest="subtract",
        default=True,
        help="Subtract Fcalc of neighboring residues when running qFit",
    )
    mo.add_argument(
        "-pad",
        "--padding",
        default=8.0,
        metavar="<float>",
        type=float,
        help="Padding size for map creation",
    )
    p.add_argument(
        "--transformer",
        choices=["cctbx","qfit"],
        default=default_transformer,
        dest="transformer",
        help="Map sampling algorithm")
    p.add_argument(
        "--transformer-map-coeffs",
        choices=["cctbx","qfit"],
        default=None,
        dest="transformer_map_coeffs",
        help="Map coefficients FFT implementation (for testing effect of gridding behavior)")
    p.add_argument(
        "--no-expand-p1",
        action="store_true",
        dest="no_expand_p1",
        default=False,
        help="Disable P1 map expansion for QFit transformer only")

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
        help="Remove conformers during sampling that have atoms "
        "with no density support, i.e. atoms are positioned "
        "at density values below <density-cutoff>",
    )
    p.add_argument(
        "-cf",
        "--clash-scaling-factor",
        default=0.75,
        metavar="<float>",
        type=float,
        help="Set clash scaling factor",
    )
    if default_enable_external_clash:
        p.add_argument(
            "-ec",
            "--no-external-clash",
            action="store_false",
            dest="external_clash",
            help="Turn off external clash detection during sampling",
        )
    else:
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
    og = p.add_argument_group("Output options")
    og.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )
    og.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    og.add_argument(
        "--debug", action="store_true", help="Log as much information as possible"
    )
    og.add_argument(
        "--write_intermediate_conformers",
        action="store_true",
        help="Write intermediate structures to file (useful with debugging)",
    )
    og.add_argument("--pdb", help="Name of the input PDB")
    return p


def load_and_scale_map(options, structure):
    """
    Load target experimental map (or map coefficients) and apply a scale
    factor derived from the model-based map.
    """
    # Load map and prepare it
    map_transformer = options.transformer_map_coeffs
    if map_transformer is None:
        map_transformer = options.transformer
    xmap = XMap.fromfile(
        options.map,
        resolution=options.resolution,
        label=options.label,
        transformer=map_transformer
    )
    xmap = xmap.canonical_unit_cell(no_expand_p1=options.no_expand_p1)

    # Scale map based on input structure
    if options.scale is True:
        scaler = MapScaler(xmap, em=options.em, debug=options.debug)
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        logger.info("Scaling with resolution=%.3f radius=%.3f", reso, radius)
        scaler.scale(structure,
                     radius=options.scale_rmask * radius,
                     transformer=options.transformer)
    return xmap
