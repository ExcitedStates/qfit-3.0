#!/usr/bin/env python
"""
End-to-end timing benchmark for qfit_protein on the 1G8A test structure.

The benchmark always loads the *full* structure, so that costs which scale
with protein size (and are paid once per residue) show up in the numbers.

Stages timed:
  residues  -- run _run_qfit_residue serially on a set of residues
  overhead  -- run _run_qfit_residue_parallel with every residue already
               checkpointed, which isolates the per-protein preparation
               (backbone coordinates, checkpoint scan) and the stitching
               of all residues into one model
  segment   -- run qfit_segment on the stitched model

Usage:
  python scripts/benchmarks/bench_qfit_protein.py --out results.json
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np

from qfit.command_line.qfit_protein import (
    QFitOptions,
    build_argparser,
    prepare_qfit_protein,
    _run_qfit_residue,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "tests", "basic_qfit_protein_test",
)
DEFAULT_RESIDUES = [62, 63, 64, 65, 66, 67, 68, 69, 97, 109, 167]


class _NullQueue:
    def put_nowait(self, record):
        pass

    def put(self, record, *args, **kwargs):
        pass


def make_qfit(tmp_dir, nproc=1):
    args = build_argparser().parse_args([
        os.path.join(DATA_DIR, "composite_omit_map.mtz"),
        os.path.join(DATA_DIR, "1G8A_refined.pdb"),
        "-l", "2FOFCWT,PH2FOFCWT",
        "-d", tmp_dir,
        "--nproc", str(nproc),
    ])
    options = QFitOptions()
    options.apply_command_args(args)
    return prepare_qfit_protein(options)


def _residue_lookup(qfit):
    protein = (qfit.structure.extract("record", "HETATM", "!=")
               .extract("resn", "HOH", "!="))
    return {r.id[0]: r for r in protein.single_conformer_residues}


def _backbone_coor_dict(qfit, residues):
    # Use the production helper when available, otherwise fall back to the
    # inline construction used by older versions of qfit_protein.
    if hasattr(qfit, "_build_backbone_coor_dict"):
        return qfit._build_backbone_coor_dict(residues)
    d = {}
    for residue in residues:
        key = (residue.chain[0], residue.resi[0], residue.id[1])
        coords, umats = {}, {}
        sel = (qfit.structure.extract("chain", residue.chain[0], "==")
               .extract("resi", residue.resi[0], "==")
               .extract("icode", residue.id[1], "=="))
        atom_name = "O" if residue.resn[0] == "GLY" else "CB"
        for altloc in np.unique(sel.altloc):
            coords[altloc] = sel.extract("altloc", altloc).coor
            try:
                umats[altloc] = (sel.extract("name", atom_name)
                                 .extract("altloc", altloc)
                                 .extract_anisous()[0])
            except Exception:
                umats[altloc] = None
        d[key] = {"coords": coords, "u_matrices": umats}
    return d


def bench_residues(resids, tmp_dir):
    qfit = make_qfit(tmp_dir)
    lookup = _residue_lookup(qfit)
    residues = [lookup[i] for i in resids]
    bb = _backbone_coor_dict(qfit, residues)
    out = {}
    for residue in residues:
        xmap = qfit.get_map_around_substructure(residue)
        t0 = time.perf_counter()
        msg = _run_qfit_residue(
            residue=residue, structure=qfit.structure, xmap=xmap,
            options=qfit.options, logqueue=_NullQueue(),
            backbone_coor_dict=bb)
        dt = time.perf_counter() - t0
        fname = os.path.join(tmp_dir, residue.shortcode,
                             "multiconformer_residue.pdb")
        out[residue.shortcode] = {"seconds": dt, "result": msg,
                                  "has_output": os.path.exists(fname)}
        print(f"  {residue.shortcode:8s} {residue.resn[0]} {dt:8.2f}s  {msg}",
              flush=True)
    return out


def _seed_checkpoints(tmp_dir, model_file):
    """Write each residue of a multiconformer model as a checkpoint."""
    from qfit import Structure
    model = Structure.fromfile(model_file)
    n = 0
    for rg in model.extract("record", "ATOM").residue_groups:
        shortcode = f"{rg.chain[0]}_{rg.resi[0]}"
        if rg.icode[0]:
            shortcode += f"_{rg.icode[0]}"
        fname = os.path.join(tmp_dir, shortcode, "multiconformer_residue.pdb")
        if not os.path.exists(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            Structure.fromstructurelike(rg).tofile(fname)
            n += 1
    print(f"  seeded {n} residue checkpoints from {model_file}")


def bench_overhead(tmp_dir, nproc, checkpoints_from=None):
    """Checkpoint every residue, then time prep + stitching."""
    qfit = make_qfit(tmp_dir, nproc=nproc)
    if checkpoints_from:
        _seed_checkpoints(tmp_dir, checkpoints_from)
    for residue in _residue_lookup(qfit).values():
        fname = os.path.join(tmp_dir, residue.shortcode,
                             "multiconformer_residue.pdb")
        if not os.path.exists(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            residue.tofile(fname)
    t0 = time.perf_counter()
    model = qfit._run_qfit_residue_parallel()
    dt = time.perf_counter() - t0
    print(f"  overhead (prep + stitch) {dt:.2f}s, {model.natoms} atoms",
          flush=True)
    return {"seconds": dt, "natoms": int(model.natoms)}, qfit, model


def bench_segment(qfit, model):
    qfit.pdb = ""  # normally set by QFitProtein.run()
    t0 = time.perf_counter()
    result = qfit._run_qfit_segment(model)
    dt = time.perf_counter() - t0
    print(f"  segment {dt:.2f}s, {result.natoms} atoms, "
          f"avg conformers {result.average_conformers():.3f}", flush=True)
    return {"seconds": dt, "natoms": int(result.natoms),
            "average_conformers": float(result.average_conformers())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--residues", type=int, nargs="*", default=DEFAULT_RESIDUES)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--stages", nargs="*",
                   default=["residues", "overhead", "segment"])
    p.add_argument("--checkpoints-from", default=None,
                   help="Multiconformer model used to seed residue checkpoints "
                        "(gives qfit_segment realistic input), e.g. "
                        "example/qfit_protein_example/multiconformer_model2.pdb")
    p.add_argument("--out", default=None, help="Write JSON results here")
    p.add_argument("--keep", action="store_true", help="Keep the work dir")
    a = p.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="qfit-bench-")
    print(f"work dir: {tmp_dir}")
    results = {}
    try:
        if "residues" in a.stages:
            print("residues:")
            results["residues"] = bench_residues(a.residues, tmp_dir)
            results["residues_total_seconds"] = sum(
                r["seconds"] for r in results["residues"].values())
        if "overhead" in a.stages or "segment" in a.stages:
            print("overhead:")
            results["overhead"], qfit, model = bench_overhead(
                tmp_dir, a.nproc, a.checkpoints_from)
            if "segment" in a.stages:
                print("segment:")
                results["segment"] = bench_segment(qfit, model)
    finally:
        if a.out:
            with open(a.out, "w") as f:
                json.dump(results, f, indent=2)
        if not a.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
