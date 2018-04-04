import argparse
import sys
import time
from itertools import izip
from string import ascii_uppercase

import numpy as np

from .volume import Volume
from .structure import Ligand, Structure
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver
from .validator import Validator


def parse_args():

    p = argparse.ArgumentParser(description="Determine occupancies from ligand set.")

    p.add_argument("xmap", help="CCP4 map file with P1 symmetry.")
    p.add_argument("resolution", type=float, help="Map resolution in angstrom.")
    p.add_argument("ligands", nargs="+", help="PDB files containing ligand.")

    # Map preparation routines
    #p.add_argument("-ns", "--no-scale", action="store_true",
    #               help="Do not scale the density.")
    p.add_argument("-r", "--receptor", type=str, default=None,
                   help="PDB file with receptor used to scale and prepare the density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.0,
                   help="Density value to use as cutoff in sigma.")
    p.add_argument("-c", "--cardinality", type=int, default=5,
                   help="Cardinality constraint during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.2,
                   help="Threshold constraint during MIQP.")
    p.add_argument("-o", "--output", default="solve.pdb",
                   help="Name of output PDB file.")
    p.add_argument("-i", "--info", default=None,
                   help="File to write info results to.")
    args = p.parse_args()
    if args.info is None:
        args.info = sys.stdout
    else:
        args.info = open(args.info, 'w')

    return args


def scale_map(args, xmap, rmask):
    if args.receptor is not None:
        footprint = Structure.fromfile(args.receptor).select('record', 'ATOM')
    else:
        footprint = Structure.fromfile(args.ligands[0])
    model_map = Volume.zeros_like(xmap)
    transformer = Transformer(footprint, model_map, simple=True, rmax=3)
    transformer.mask(rmask)
    mask = model_map.array > 0
    transformer.reset()
    transformer.initialize()
    transformer.density()
    if args.density_cutoff is not None:
        mean = xmap.array.mean()
        std = xmap.array.std()
        cutoff_mask = ((xmap.array - mean) / std) < args.density_cutoff
    xmap_masked = xmap.array[mask]
    model_masked = model_map.array[mask]
    model_masked_mean = model_masked.mean()
    xmap_masked_mean = xmap_masked.mean()
    model_masked -= model_masked_mean
    xmap_masked -= xmap_masked_mean
    scaling_factor = ((model_masked * xmap_masked).sum() /
                      (xmap_masked * xmap_masked).sum())
    args.info.write('Scaling factor: {:.2f}\n'.format(scaling_factor))
    xmap.array -= xmap_masked_mean
    xmap.array *= scaling_factor
    xmap.array += model_masked_mean
    if args.density_cutoff is not None:
        xmap.array[cutoff_mask] = 0
    if args.receptor is not None:
        xmap.array -= model_map.array
    model_map.array.fill(0)
    return xmap


def solve():

    args = parse_args()

    # Expand the crystallography map to fill the whole unit cell
    xmap = Volume.fromfile(args.xmap).fill_unit_cell()
    xmap.set_spacegroup("P1")
    resolution = args.resolution
    conformers = [Ligand.fromfile(fname) for fname in args.ligands]
    args.info.write("Initial number of conformers: {}\n".format(len(conformers)))
    if resolution < 3.0:
        rmask = 0.7 + (resolution - 0.6) / 3.0
    else:
        rmask = 0.5 * resolution

    # Scale the map under the footprint of the receptor
    xmap = scale_map(args, xmap, rmask)

    # Analyze ligand for rscc and rmsd metrics
    validator = Validator(xmap, resolution)
    for fname, conformer in izip(args.ligands, conformers):
        conformer.rscc = validator.rscc(conformer, rmask=1.5)
        conformer.fname = fname
    conformers_sorted = sorted(conformers, key=lambda conformer: conformer.rscc, reverse=True)
    line = '{fname} {rscc:.3f}\n'
    for conformer in conformers_sorted:
        args.info.write(line.format(fname=conformer.fname, rscc=conformer.rscc))
    best_rscc = conformers_sorted[0].rscc
    # Remove conformers with a significantly lower rscc
    rscc_cutoff = 0.1 * best_rscc
    conformers = [conformer for conformer in conformers_sorted
                  if (best_rscc - conformer.rscc)  < rscc_cutoff]
    noH = np.logical_not(conformers[0].select('e', 'H', return_ind=True))
    args.info.write('Number of conformers after rscc cutoff: {}\n'.format(len(conformers)))

    coor_set = [conformers[0].coor]
    # Remove geometrically similar ligands
    filtered_conformers = [conformers[0]]
    for conformer in conformers[1:]:
        max_dist = min([np.abs(
            np.linalg.norm(conformer.coor[noH] - coor[noH], axis=1).max()
        ) for coor in coor_set])
        if max_dist < 0.5:
            continue
        coor_set.append(conformer.coor)
        filtered_conformers.append(conformer)
    conformers = filtered_conformers
    args.info.write('Number of conformers after removing duplicates: {}\n'.format(len(conformers)))

    # Remove conformers that have drifted off
    best_conformer = conformers[0]
    filtered_conformers = [best_conformer]
    center = best_conformer.coor.mean(axis=0)
    for conformer in conformers[1:]:
        rmsd = conformer.rmsd(best_conformer)
        if rmsd > 6:
            continue
        shift = np.linalg.norm(conformer.coor.mean(axis=0) - center)
        if shift > 3:
            continue
        filtered_conformers.append(conformer)
    conformers = filtered_conformers
    args.info.write('Number of conformers after removing drifters: {}\n'.format(len(conformers)))

    # Now do QP/MIQP with the remaining conformers. Check if rscc increases
    # substantially by including each conformer. Keep repeating till it is
    # consistent
    ligand_template = Ligand.fromfile(args.ligands[0])
    ligand_template.data['q'].fill(1)
    smax = 1.0 / (2.0 * resolution)
    model_map = Volume.zeros_like(xmap)
    transformer = Transformer(ligand_template, model_map, smax=smax, rmax=3)
    while True:
        # Create mask
        for conformer in conformers:
            ligand_template.coor[:] = conformer.coor
            transformer.mask(rmask)
        mask = model_map.array > 0
        model_map.array.fill(0)

        # Create densities
        nvalues = mask.sum()
        target = xmap.array[mask]
        models = np.zeros((len(conformers), nvalues))
        for n, conformer in enumerate(conformers):
            transformer.reset()
            ligand_template.coor[:] = conformer.coor
            transformer.density()
            models[n,:] = model_map.array[mask]
            model_map.array.fill(0)

        # Do MIQP
        miqpsolver = MIQPSolver(target, models)
        miqpsolver(maxfits=args.cardinality, threshold=args.threshold)
        filtered_conformers = []
        for q, conformer in izip(miqpsolver.occupancies, conformers):
            if q > 0.0001:
                conformer.data['q'].fill(q)
                filtered_conformers.append(conformer)
        conformers = filtered_conformers

        # Check fisher z correlation
        conformers[0].zscore = -1
        multiconformer = conformers[0]
        multiconformer.data['altloc'].fill('A')
        nconformers = 1
        filtered_conformers = [conformers[0]]
        for conformer in conformers[1:]:
            conformer.data['altloc'].fill(ascii_uppercase[nconformers])
            new_multiconformer = multiconformer.combine(conformer)
            diff = validator.fisher_z_difference(
                multiconformer, new_multiconformer, rmask=1.5, simple=True
            )
            if diff < 0.0:
                continue
            multiconformer = new_multiconformer
            conformer.zscore = diff
            filtered_conformers.append(conformer)
            nconformers += 1
        if len(filtered_conformers) == len(conformers):
            conformers = filtered_conformers
            break
        conformers = filtered_conformers

    line = "{fname}\t{rscc:.3f}\t{zscore:.3f}\t{occupancy:.2f}\n"
    for conformer in conformers:
        args.info.write(line.format(
            fname=conformer.fname, rscc=conformer.rscc,
            zscore=conformer.zscore, occupancy=conformer.q[0]
        ))
    multiconformer.tofile(args.output)
