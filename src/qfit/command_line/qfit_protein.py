import gc
import multiprocessing as mp
import os.path
import os
import time
import logging
import traceback
from collections import defaultdict

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
from qfit.structure.base_structure import ANISOU_SCALE
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

    def _build_backbone_coor_dict(self, residues):
        """Collect, per residue and altloc, the coordinates of the residue's
        atoms and the anisotropic U matrix of its CB atom (O for GLY).

        Returns:
            dict: {(chain, resi, icode): {"coords": {altloc: coor},
                                          "u_matrices": {altloc: U or None}}}
        """
        # Read per-atom data once; per-residue structure.extract() calls each
        # rebuild these arrays for the whole structure.
        structure = self.structure
        chains = structure.chain
        resis = structure.resi
        icodes = structure.icode
        altlocs = structure.altloc
        names = structure.name
        coor = structure.coor
        uij = np.array(structure.atoms.extract_uij()).reshape(-1, 6)

        atoms_by_residue = defaultdict(list)
        for index, key in enumerate(zip(chains, resis, icodes)):
            atoms_by_residue[key].append(index)

        backbone_coor_dict = {}
        for residue in residues:
            chain, resi, icode = residue.chain[0], residue.resi[0], residue.id[1]
            residue_chain_key = (chain.replace('"', ""), resi, icode)
            atom_name = "O" if residue.resn[0] == "GLY" else "CB"
            indices = np.asarray(atoms_by_residue.get((chain, resi, icode), []),
                                 dtype=int)
            grouped_coords = {}
            grouped_u_matrices = {}
            for altloc in np.unique(altlocs[indices]):
                altloc_indices = indices[altlocs[indices] == altloc]
                grouped_coords[altloc] = coor[altloc_indices]
                atom_indices = altloc_indices[names[altloc_indices] == atom_name]
                grouped_u_matrices[altloc] = _anisou_matrix(uij[atom_indices])
            entry = backbone_coor_dict.setdefault(residue_chain_key, {})
            entry["coords"] = grouped_coords
            entry["u_matrices"] = grouped_u_matrices
        return backbone_coor_dict

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
        residues = _select_residues_to_run(self.structure, self.options.residue)

        backbone_coor_dict = self._build_backbone_coor_dict(residues)

        # Filter the residues: take only rotameric residues (others have
        # nothing to sample, and are kept as-is when stitching below) that
        # do not already have a checkpoint.
        def does_multiconformer_checkpoint_exist(residue):
            fname = self._get_partial_model_path(residue)
            if os.path.exists(fname):
                logger.info(f"Residue {residue.shortcode}: {fname} already exists.")
                return True
            else:
                return False

        residues_to_sample = [
            residue for residue in residues
            if residue.resn[0] in ROTAMERS
            and not does_multiconformer_checkpoint_exist(residue)
        ]
        # Start the most expensive residues first, so that a long-running
        # residue does not end up as the last job while other workers idle.
        residues_to_sample.sort(key=_estimated_sampling_cost, reverse=True)

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
            # If multiprocessing, launch a Pool and run Jobs.
            # The structure is large and slow to unpickle, so it is sent to
            # each worker once (initializer) rather than with every job; jobs
            # only carry the residue identifier and the local map.
            with ctx.Pool(
                processes=self.options.nproc,
                maxtasksperchild=WORKER_MAX_TASKS,
                initializer=_init_worker,
                initargs=(self.structure, backbone_coor_dict),
            ) as pool:
                futures = [
                    pool.apply_async(
                        _run_qfit_residue_in_worker,
                        kwds={
                            "residue_key": residue.identifier_tuple,
                            "xmap": self.get_map_around_substructure(residue),
                            "options": self.options,
                            "logqueue": logqueue,
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
            # Collect the pieces first and combine them in a single call:
            # every combine() rebuilds the hierarchy of the whole result.
            non_rotameric = []
            residue_multiconformers = []
            for residue in residues:
                # Check the residue is a rotameric residue,
                # if not, we won't have a multiconformer_residue.pdb.
                # Make sure to append it to the hetatms object so it stays in the final output.
                if residue.resn[0] not in ROTAMERS:
                    non_rotameric.append(residue)
                    continue

                # Load the multiconformer_residue.pdb file
                fname = self._get_partial_model_path(residue)
                if not os.path.exists(fname):
                    logger.warning(
                        f"[{residue.shortcode}] Couldn't find {fname}! "
                        "Will not be present in multiconformer_model.pdb!"
                    )
                    continue
                residue_multiconformers.append(Structure.fromfile(fname))

            if non_rotameric:
                self.hetatms = self.hetatms.combine(*non_rotameric)
            if residue_multiconformers:
                multiconformer_model = residue_multiconformers[0].combine(
                    *residue_multiconformers[1:]
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

# Worker processes are recycled after this many residues, to bound the
# effect of any memory leak. Each restart costs a few seconds (imports and
# unpickling the structure), so this should not be too small.
WORKER_MAX_TASKS = 32

# Per-worker-process state, set by _init_worker.
_WORKER_STATE = {}


def _select_residues_to_run(structure, residue_option=None):
    """Return the single-conformer residues that qfit_protein will process.

    Args:
        structure: the input Structure.
        residue_option: None for all residues, or the --residue string,
            e.g. "A,105" or "A,105:B".
    """
    # Create a list of residues from single conformations of proteinaceous residues.
    # If we were to loop over all single_conformer_residues, then we end up adding HETATMs in two places
    #    First as we combine multiconformer_residues into multiconformer_model (because they won't be in ROTAMERS)
    #    And then as we re-combine HETATMs back into the multiconformer_model.
    if residue_option is None:
        return list(
            structure.extract("record", "HETATM", "!=")
            .extract("resn", "HOH", "!=")
            .single_conformer_residues
        )

    chainid, resi = residue_option.split(",")
    if ":" in resi:
        resi, icode = resi.split(":")
    elif "_" in resi:
        resi, icode = resi.split("_")
    else:
        icode = ""
    # Extract the residue using attribute-based selection to handle icodes.
    structure_resi = (
        structure
        .extract("chain", chainid, "==")
        .extract("resi", int(resi), "==")
    )
    if icode:
        structure_resi = structure_resi.extract("icode", icode, "==")
    return list(
        structure_resi
        .extract("resn", "HOH", "!=")
        .single_conformer_residues
    )


def _anisou_matrix(uij_rows):
    """Scaled 3x3 U matrix from the first of `uij_rows` (cctbx order
    u11, u22, u33, u12, u13, u23), or None if the atoms are isotropic or
    there are no atoms. Matches BaseStructure.extract_anisous()[0]."""
    if len(uij_rows) == 0 or np.all(uij_rows == -1):
        return None
    u = uij_rows[0] * ANISOU_SCALE
    return np.array([
        (u[0], u[3], u[4]),
        (u[3], u[1], u[5]),
        (u[4], u[5], u[2]),
    ])


def _estimated_sampling_cost(residue):
    """Sort key approximating how long a residue takes to sample."""
    info = ROTAMERS.get(residue.resn[0])
    if info is None:
        return (0, 0)
    return (info["nchi"], len(info["rotamers"]))


def _init_worker(structure, backbone_coor_dict):
    """Pool initializer: keep the shared, read-only inputs in the worker."""
    _WORKER_STATE.clear()
    _WORKER_STATE["structure"] = structure
    _WORKER_STATE["backbone_coor_dict"] = backbone_coor_dict


def _run_qfit_residue_in_worker(residue_key, xmap, options, logqueue):
    """Pool job: run qfit on the residue identified by (chain, resi, icode)."""
    structure = _WORKER_STATE["structure"]
    residues = _WORKER_STATE.get("residues")
    if residues is None:
        residues = {
            residue.identifier_tuple: residue
            for residue in _select_residues_to_run(structure, options.residue)
        }
        _WORKER_STATE["residues"] = residues
    return _run_qfit_residue(
        residue=residues[residue_key],
        structure=structure,
        xmap=xmap,
        options=options,
        logqueue=logqueue,
        backbone_coor_dict=_WORKER_STATE["backbone_coor_dict"],
    )


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

    def _select_residue_structure(structure_obj, residue_obj, chainid, resi, icode):
        """
        Select the residue from structure_obj, handling insertion codes.
        Falls back to atom serial selection if resi/icode filtering yields no atoms.
        """
        structure_resi = (
            structure_obj
            .extract("chain", chainid, "==")
            .extract("resi", resi, "==")
        )
        if icode:
            structure_resi = structure_resi.extract("icode", icode, "==")
        if structure_resi.natoms == 0:
            structure_resi = structure_obj.extract("atomid", residue_obj.atomid, "==")
        return structure_resi

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
            structure_new = structure
            structure_resi = _select_residue_structure(
                structure, residue, chainid, resi, icode
            )
            chain = structure_resi[chainid]
            conformer = chain.conformers[0]
            residue = conformer[residue.id]
            residue.tofile(fname)
            return

    # Copy the structure
    (chainid, resi, icode) = residue.identifier_tuple
    structure_new = structure.copy()
    structure_resi = _select_residue_structure(
        structure_new, residue, chainid, resi, icode
    )
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
            remove_mask = (
                (structure_new.chain == chainid)
                & (structure_new.resi == resi)
                & (structure_new.altloc == altloc)
            )
            if icode:
                remove_mask &= (structure_new.icode == icode)
            structure_new = structure_new.extract(np.flatnonzero(~remove_mask))

    # add multiple backbone positions

    chain_resi_id = (chainid, resi, icode)
    if chain_resi_id not in backbone_coor_dict:
        logger.warning(
            f"[{residue.shortcode}] No backbone coordinates found for insertion; "
            "falling back to base residue index."
        )
        chain_resi_id = (chainid, resi, "")
    options.backbone_coor_dict = backbone_coor_dict[chain_resi_id]
    # Exception handling in case qFit-residue fails:
    qfit = None
    try:
        qfit = QFitRotamericResidue(residue, structure_new, xmap, options)
        qfit.run()
    except RuntimeError as e:
        tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
        identifier = getattr(qfit, "identifier", residue.shortcode)
        logger.warning(
            f"[{identifier}] "
            f"Unable to produce an alternate conformer. "
            f"Using deposited conformer A for this residue."
        )
        logger.info(
            f"[{identifier}] This is a result of the following exception:\n"
            f"{tb})"
        )
        if qfit is None:
            residue.tofile(fname)
            return f"[{identifier}]: 1 conformer"
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
