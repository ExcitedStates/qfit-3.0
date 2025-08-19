import gc
import multiprocessing as mp
import os.path
import os
import time
import logging
import traceback
import itertools as itl

import pandas as pd
from tqdm import tqdm
import numpy as np

from qfit.command_line.common_options import get_base_argparser, load_and_scale_map
from qfit.command_line.custom_argparsers import ToggleActionFlag
from qfit.logtools import (
    setup_logging,
    log_run_info,
    poolworker_setup_logging,
    QueueListener,
)
from qfit.qfit import (QFitOptions, QFitRotamericResidue, QFitSegment)
from qfit import Structure
from qfit.structure.rotamers import ROTAMERS


logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def build_argparser():
    p = get_base_argparser(__doc__)

    p.add_argument(
        "-em",
        "--cryo_em",
        action="store_true",
        dest="em",
        help="Run qFit with EM options",
    )
    p.add_argument(
        "--residue",
        type=str,
        help="If selected will run qFit Residue"
        "Chain, residue id, and optionally insertion code for "
        "residue in structure, e.g. A,105, or A,105:A.",
    )
    p.add_argument(
        "-sb",
        "--no-sampling-b",
        action="store_false",
        dest="sample_bfactors",
        help="Do not sample b-factors within qFit",
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
        "-b",
        "--dofs-per-iteration",
        default=1,
        metavar="<int>",
        type=int,
        help="Number of internal degrees that are sampled/built per iteration",
    )
    p.add_argument(
        "-s",
        "--dihedral-stepsize",
        default=6,
        metavar="<float>",
        type=float,
        help="Stepsize for dihedral angle sampling in degrees",
    )
    p.add_argument(
        "-rn",
        "--rotamer-neighborhood",
        default=24,
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
    # qFit Segment options
    p.add_argument(
        "--only-segment",
        action="store_true",
        dest="only_segment",
        help="Only perform qfit segment",
    )
    p.add_argument(
        "-f",
        "--fragment-length",
        default=4,
        dest="fragment_length",
        metavar="<int>",
        type=int,
        help="Fragment length used during qfit_segment",
    )
    p.add_argument(
        "--segment-threshold-selection",
        action=ToggleActionFlag,
        dest="seg_bic_threshold",
        default=True,
        help="Use BIC to select the most parsimonious MIQP threshold (segment)",
    )

    # EM options
    p.add_argument(
        "-q",
        "--qscore",
        help="Q-score text output file",
    )
    p.add_argument(
        "-q_cutoff",
        "--qscore_cutoff",
        help="Q-score value where we should not model in alternative conformers.",
        default=0.7,
    )
    return p


class QFitProtein:
    def __init__(self, structure, xmap, options):
        self.xmap = xmap
        self.structure = structure
        self.options = options
        self.hetatms = None

    def run(self):
        if self.options.pdb is not None:
            self.pdb = self.options.pdb + "_"
        else:
            self.pdb = ""
        # Get information about all backbone atoms. If the input structure has multiple backbones, we will initiate backbone sampling (and all residue sampling) using all backbone coordinates
        if self.options.residue is not None:
            chainid, resi = self.options.residue.split(",")  # pylint: disable=unused-variable

        if self.options.residue is not None:  # run qFit residue
            multiconformer = self._run_qfit_residue_parallel()
        elif self.options.only_segment:
            multiconformer = self._run_qfit_segment(self.structure)
        else:
            multiconformer = self._run_qfit_residue_parallel()
            multiconformer = self._run_qfit_segment(multiconformer)
        return multiconformer

    @property
    def file_ext(self):
        # we can't rely on this being propagated in self.structure....
        path_fields = self.options.structure.split(".")
        if path_fields[-1] == "gz":
            return ".".join(path_fields[-2:])
        return path_fields[-1]

    def _get_output_model_path(self, prefix):
        return os.path.join(self.options.directory, f"{prefix}.{self.file_ext}")

    def _get_partial_model_path(self, residue):
        return os.path.join(self.options.directory,
                            residue.shortcode,
                            "multiconformer_residue.pdb")

    def get_map_around_substructure(self, substructure):
        """Make a subsection of the map near the substructure.

        Args:
            substructure (qfit.structure.base_structure._BaseStructure):
                a substructure to carve a map around, commonly a Residue

        Returns:
            qfit.volume.XMap: a new (smaller) map
        """
        return self.xmap.extract(substructure.coor, padding=self.options.padding)

    def _run_qfit_residue_parallel(self):
        """Run qfit independently over all residues."""
        # This function hands out the job in parallel to a Pool of Workers.
        # To create Workers, we will use "forkserver" where possible,
        #     and default to "spawn" elsewhere (e.g. on Windows).
        try:
            ctx = mp.get_context(method="forkserver")
        except ValueError:
            ctx = mp.get_context(method="spawn")

        # Extract non-protein atoms
        self.hetatms = self.structure.extract("record", "HETATM", "==")
        other_waters = self.structure.extract("record", "ATOM", "==")
        other_waters = other_waters.extract("resn", "HOH", "==")
        self.hetatms = self.hetatms.combine(other_waters)

        # Create a list of residues from single conformations of proteinaceous residues.
        # If we were to loop over all single_conformer_residues, then we end up adding HETATMs in two places
        #    First as we combine multiconformer_residues into multiconformer_model (because they won't be in ROTAMERS)
        #    And then as we re-combine HETATMs back into the multiconformer_model.
        if self.options.residue is not None:
            chainid, resi = self.options.residue.split(",")
            if ":" in resi:
                resi, icode = resi.split(":")
                residue_id = (int(resi), icode)  # pylint: disable=unused-variable
            elif "_" in resi:
                resi, icode = resi.split("_")
                residue_id = (int(resi), icode)
            else:
                residue_id = int(resi)
                icode = ""
            # Extract the residue:
            residues = list(
                self.structure.extract(f"resi {resi} and chain {chainid}")
                .extract("resn", "HOH", "!=")
                .single_conformer_residues
            )

        else:
            residues = list(
                self.structure.extract("record", "HETATM", "!=")
                .extract("resn", "HOH", "!=")
                .single_conformer_residues
            )

        backbone_coor_dict = {}
        for residue in residues:
            grouped_coords = {}
            grouped_u_matrices = {}
            residue_chain_key = (residue.resi[0], residue.chain[0].replace('"', ""))
            if residue_chain_key not in backbone_coor_dict:
                backbone_coor_dict[residue_chain_key] = {}
            for altloc in np.unique(
                self.structure.extract("chain", residue.chain[0], "==")
                .extract("resi", residue.resi[0], "==")
                .altloc
            ):
                grouped_coords[altloc] = (
                    self.structure.extract("chain", residue.chain[0], "==")
                    .extract("resi", residue.resi[0], "==")
                    .extract("altloc", altloc)
                    .coor
                )
                if residue.resn[0] == "GLY":
                    atom_name = "O"
                else:
                    atom_name = "CB"
                atom = (
                    self.structure.extract("chain", residue.chain[0], "==")
                    .extract("resi", residue.resi[0], "==")
                    .extract("name", atom_name)
                    .extract("altloc", altloc)
                )
                try:
                    u_matrix = atom.extract_anisous()[0]
                except Exception:
                    u_matrix = None
                grouped_u_matrices[altloc] = u_matrix
            backbone_coor_dict[residue_chain_key]["coords"] = grouped_coords
            backbone_coor_dict[residue_chain_key]["u_matrices"] = grouped_u_matrices

        # Filter the residues: take only those not containing checkpoints.
        def does_multiconformer_checkpoint_exist(residue):
            fname = self._get_partial_model_path(residue)
            if os.path.exists(fname):
                logger.info(f"Residue {residue.shortcode}: {fname} already exists.")
                return True
            else:
                return False

        residues_to_sample = list(
            itl.filterfalse(does_multiconformer_checkpoint_exist, residues)
        )

        # Print execution stats
        logger.info(f"Residues to sample: {len(residues_to_sample)}")
        logger.info(f"nproc: {self.options.nproc}")

        # Build a Manager, have it construct a Queue. This will conduct
        #   thread-safe and process-safe passing of LogRecords.
        # Then launch a QueueListener Thread to read & handle LogRecords
        #   that are placed on the Queue.
        mgr = mp.Manager()
        logqueue = mgr.Queue()
        listener = QueueListener(logqueue)
        listener.start()

        # Initialise progress bar
        progress = tqdm(
            total=len(residues_to_sample),
            desc="Sampling residues",
            unit="residue",
            unit_scale=True,
            leave=True,
            miniters=1,
        )

        # Define callbacks and error callbacks to be attached to Jobs
        def _cb(result):
            if result:
                logger.info(result)
            progress.update()

        def _error_cb(e):
            tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
            logger.critical(tb)
            progress.update()

        # Here, we calculate alternate conformers for individual residues.
        if self.options.nproc > 1:
            # If multiprocessing, launch a Pool and run Jobs
            with ctx.Pool(processes=self.options.nproc, maxtasksperchild=4) as pool:
                futures = [
                    pool.apply_async(
                        _run_qfit_residue,
                        kwds={
                            "residue": residue,
                            "structure": self.structure,
                            "xmap": self.get_map_around_substructure(residue),
                            "options": self.options,
                            "logqueue": logqueue,
                            "backbone_coor_dict": backbone_coor_dict,
                        },
                        callback=_cb,
                        error_callback=_error_cb,
                    )
                    for residue in residues_to_sample
                ]

                # Make sure all jobs are finished
                # #TODO If a task crashes or is OOM killed, then there is no result.
                #       f.wait waits forever. It would be good to handle this case.
                for f in futures:
                    f.wait()

            # Wait until all workers have completed
            pool.join()

        else:
            # Otherwise, run this in the MainProcess
            for residue in residues_to_sample:
                if residue.resn[0] not in ROTAMERS:
                   self.hetatms = self.hetatms.combine(residue)
                   continue
                else:
                    try:
                        result = _run_qfit_residue(
                            residue=residue,
                            structure=self.structure,
                            xmap=self.get_map_around_substructure(residue),
                            options=self.options,
                            logqueue=logqueue,
                            backbone_coor_dict=backbone_coor_dict,
                        )
                        _cb(result)
                    except Exception as e:
                        _error_cb(e)

        # Close the progressbar
        progress.close()

        # There are no more sub-processes, so we stop the QueueListener
        listener.stop()
        listener.join()

        # Combine all multiconformer residues into one structure, multiconformer_model
        multiconformer_model = None
        if self.options.residue is not None:
            for residue in residues:
                # Load the multiconformer_residue.pdb file
                fname = self._get_partial_model_path(residue)
                if not os.path.exists(fname):
                    logger.warning(
                        f"[{residue.shortcode}] Couldn't find {fname}! "
                        "Will not be present in multiconformer_model.pdb!"
                    )
                    continue
                residue_multiconformer = Structure.fromfile(fname)
                fname = f'{residue.shortcode}_qFit_residue.{self.file_ext}'
                residue_multiconformer.tofile(fname)
                multiconformer_model = residue_multiconformer

        if self.options.residue is None:
            for residue in residues:
                # Check the residue is a rotameric residue,
                # if not, we won't have a multiconformer_residue.pdb.
                # Make sure to append it to the hetatms object so it stays in the final output.
                if residue.resn[0] not in ROTAMERS:
                    hetatms = hetatms.combine(residue)
                    continue

                # Load the multiconformer_residue.pdb file
                fname = self._get_partial_model_path(residue)
                if not os.path.exists(fname):
                    logger.warning(
                        f"[{residue.shortcode}] Couldn't find {fname}! "
                        "Will not be present in multiconformer_model.pdb!"
                    )
                    continue
                residue_multiconformer = Structure.fromfile(fname)

                # Stitch them together
                if multiconformer_model is None:
                    multiconformer_model = residue_multiconformer
                else:
                    multiconformer_model = multiconformer_model.combine(
                        residue_multiconformer
                    )

            if multiconformer_model is None:
                raise RuntimeError(f"No usable model built for {residue.shortcode}")
            # Write out multiconformer_model.pdb only if in debug mode.
            # This output is not a final qFit output, so it might confuse users
            if self.options.debug:
                fname = self._get_output_model_path("multiconformer_model")
                multiconformer_model.tofile(fname,
                    self.structure.crystal_symmetry)

        return multiconformer_model

    def _run_qfit_segment(self, multiconformer):
        self.options.randomize_b = False
        self.options.bic_threshold = self.options.seg_bic_threshold
        if self.options.seg_bic_threshold:
            self.options.fragment_length = 3
        else:
            self.options.threshold = 0.2

        # Extract map of multiconformer in P1
        logger.debug("Extracting map...")
        _then = time.process_time()
        self.xmap = self.get_map_around_substructure(self.structure)
        _now = time.process_time()
        logger.debug(f"Map extraction took {(_now - _then):.03f} s.")

        qfit = QFitSegment(multiconformer, self.xmap, self.options)
        multiconformer = qfit()
        if self.hetatms:
            multiconformer = multiconformer.combine(self.hetatms)
        fname = self._get_output_model_path(f"{self.pdb}multiconformer_model2")
        multiconformer.tofile(fname, self.structure.crystal_symmetry)
        return multiconformer

def _run_qfit_residue(residue, structure, xmap, options, logqueue,
                      backbone_coor_dict):
    """Run qfit on a single residue to determine density-supported conformers."""
    # Don't run qfit if we have a ligand or water
    if residue.type != "rotamer-residue":
        raise RuntimeError(
            f"Residue {residue.id}: is not a rotamer-residue. Aborting qfit_residue sampling."
        )

    # Set up logger hierarchy in this subprocess
    poolworker_setup_logging(logqueue)

    # Exit early if we have already run qfit for this residue
    fname = os.path.join(options.directory,
                         residue.shortcode,
                         "multiconformer_model.pdb")
    if os.path.exists(fname):
        logger.info(
            f"Residue {residue.shortcode}: {fname} already exists, using this checkpoint."
        )
        return
    # Build the residue results directory
    residue_directory = os.path.dirname(fname)
    os.makedirs(residue_directory, exist_ok=True)

    # Determine if q-score is too low
    if options.qscore is not None:
        (chainid, resi, icode) = residue.identifier_tuple
        if (
            list(
                options.qscore[
                    (options.qscore["Res_num"] == resi)
                    & (options.qscore["Chain"] == chainid)
                ]["Q_sideChain"]
            )[0]
            < options.q_cutoff
        ):
            logger.info(
                f"Residue {residue.shortcode}: Q-score is too low for this residue. Using deposited structure."
            )
            resi_selstr = f"chain {chainid} and resi {resi}"
            if icode:
                resi_selstr += f" and icode {icode}"
            structure_new = structure
            structure_resi = structure.extract(resi_selstr)
            chain = structure_resi[chainid]
            conformer = chain.conformers[0]
            residue = conformer[residue.id]
            residue.tofile(fname)
            return

    # Copy the structure
    (chainid, resi, icode) = residue.identifier_tuple
    resi_selstr = f"chain {chainid} and resi {resi}"
    if icode:
        resi_selstr += f" and icode {icode}"
    structure_new = structure.copy()
    structure_resi = structure_new.extract(resi_selstr)
    chain = structure_resi[chainid]
    conformer = chain.conformers[0]
    residue = conformer[residue.id]
    altlocs = sorted(list(set(residue.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove("")
        except ValueError:
            pass
        for altloc in altlocs[1:]:
            sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
            sel_str = f"not ({sel_str})"
            structure_new = structure_new.extract(sel_str)

    # add multiple backbone positions

    chain_resi_id = f"{resi}, '{chainid}'"
    chain_resi_id = (resi, chainid)
    options.backbone_coor_dict = backbone_coor_dict[chain_resi_id]
    # Exception handling in case qFit-residue fails:
    qfit = QFitRotamericResidue(residue, structure_new, xmap, options)
    try:
        qfit.run()
    except RuntimeError as e:
        tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
        logger.warning(
            f"[{qfit.identifier}] "
            f"Unable to produce an alternate conformer. "
            f"Using deposited conformer A for this residue."
        )
        logger.info(
            f"[{qfit.identifier}] This is a result of the following exception:\n"
            f"{tb})"
        )
        qfit.reset_residue(residue, structure)

    # Save multiconformer_residue
    qfit.tofile()
    qfit_id = qfit.identifier

    # How many conformers were found?
    n_conformers = len(qfit.get_conformers())

    # Freeing up some memory to avoid memory issues:
    del xmap
    del qfit
    gc.collect()

    # Return a string about the residue that was completed.
    return f"[{qfit_id}]: {n_conformers} conformers"


def prepare_qfit_protein(options):
    """Loads files to build a QFitProtein job."""

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure).reorder()
    if not options.hydro:
        structure = structure.extract("e", "H", "!=")

    # fixing issues with terminal oxygens
    rename = structure.extract("name", "OXT", "==")
    rename.name = "O"
    structure = structure.extract("name", "OXT", "!=").combine(rename)

    xmap = load_and_scale_map(options, structure)

    if options.qscore is not None:
        with open(
            options.qscore, "r"
        ) as f:  # not all qscore header are the same 'length'
            for line_n, line_content in enumerate(f):
                if "Q_sideChain" in line_content:
                    break
            start_row = line_n + 1
        options.qscore = pd.read_csv(
            options.qscore,
            sep="\t",
            skiprows=start_row,
            skip_blank_lines=True,
            on_bad_lines="skip",
            header=None,
        )
        options.qscore = options.qscore.iloc[
            :, :6
        ]  # we only care about the first 6 columns
        options.qscore.columns = [
            "Chain",
            "Res",
            "Res_num",
            "Q_backBone",
            "Q_sideChain",
            "Q_residue",
        ]  # rename column names
        options.qscore["Res_num"] = options.qscore["Res_num"].fillna(0).astype(int)

    return QFitProtein(structure, xmap, options)


def main():
    """Default entrypoint for qfit_protein."""

    # Collect and act on arguments
    #   (When args==None, argparse will default to sys.argv[1:])
    p = build_argparser()
    args = p.parse_args(args=None)

    os.makedirs(args.directory, exist_ok=True)

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Build a QFitProtein job
    qfit = prepare_qfit_protein(options=options)

    # Run the QFitProtein job
    time0 = time.time()
    qfit.run()
    logger.info(f"Total time: {time.time() - time0}s")
