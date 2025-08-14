from __future__ import division
import numpy as np
import copy
import os
from .volume import XMap
from .transformer import Transformer
from .structure import Structure
import scipy.stats as st


class Validator(object):
    def __init__(self, xmap, resolution, directory, em=False):
        self.xmap = xmap
        self.resolution = resolution
        self.em = em
        self.fname = os.path.join(directory, "validation_metrics.txt")

    def rscc(self, structure, rmask=1.5, mask_structure=None, simple=True):
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        if mask_structure is None:
            transformer = Transformer(structure, model_map, simple=simple, em=self.em)
        else:
            transformer = Transformer(
                mask_structure, model_map, simple=simple, em=self.em
            )
        transformer.mask(rmask)
        mask = model_map.array > 0
        model_map.array.fill(0)
        if mask_structure is not None:
            transformer = Transformer(structure, model_map, simple=simple, em=self.em)
        transformer.density()

        corr = np.corrcoef(self.xmap.array[mask], model_map.array[mask])[0, 1]
        return corr

    def fisher_z(self, structure, rmask=1.5, simple=True):
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        transformer = Transformer(structure, model_map)
        transformer.mask(rmask)
        mask = model_map.array > 0
        nvoxels = mask.sum()
        # mv = nvoxels * self.xmap.voxel_volume
        mv = nvoxels * np.product(self.xmap.voxelspacing) * self.xmap.unit_cell.calc_v()
        model_map.array.fill(0)
        transformer.density()
        corr = np.corrcoef(self.xmap.array[mask], model_map.array[mask])[0, 1]
        # Transform to Fisher z-score
        if self.resolution.high is not None:
            sigma = 1.0 / np.sqrt(mv / self.resolution.high - 3)
        else:
            sigma = 1.0 / np.sqrt(mv - 3)
        fisher = 0.5 * np.log((1 + corr) / (1 - corr))
        return fisher

    def fisher_z_difference(self, structure1, structure2, rmask=1.5, simple=True):
        # Create mask of combined structures
        combined = structure1.combine(structure2)
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        transformer = Transformer(combined, model_map)
        transformer.mask(rmask)
        mask = model_map.array > 0
        nvoxels = mask.sum()
        # mv = nvoxels * self.xmap.voxel_volume
        mv = nvoxels * np.product(self.xmap.voxelspacing) * self.xmap.unit_cell.calc_v()

        # Get density values of xmap, and both structures
        target_values = self.xmap.array[mask]
        transformer = Transformer(structure1, model_map, simple=simple)
        model_map.array.fill(0)
        transformer.density()
        model1_values = model_map.array[mask]
        transformer = Transformer(structure2, model_map, simple=simple)
        model_map.array.fill(0)
        transformer.density()
        model2_values = model_map.array[mask]

        # Get correlation score for structure
        target_values -= target_values.mean()
        target_values /= target_values.std()

        model1_mean = model1_values.mean()
        model1_std = model1_values.std()
        corr1 = (
            (target_values * (model1_values - model1_mean)).sum() / model1_std
        ) / nvoxels
        model2_mean = model2_values.mean()
        model2_std = model2_values.std()
        corr2 = (
            (target_values * (model2_values - model2_mean)).sum() / model2_std
        ) / nvoxels
        # Transform to Fisher Z-score
        sigma = 1.0 / np.sqrt(mv / self.resolution.high - 3)
        fisher1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
        fisher2 = 0.5 * np.log((1 + corr2) / (1 - corr2))
        return (fisher2 - fisher1) / sigma

    def GoodnessOfFit(self, conformer, coor_set, occupancies, rmask, confidence=0.95):
        # Calculate the Observed map for the masked values:
        xmap_calc = XMap.zeros_like(self.xmap)
        xmap_calc.set_space_group("P1")
        transformer = Transformer(conformer, xmap_calc)
        rscc_set = np.zeros_like(occupancies)
        for i, coor in enumerate(coor_set):
            conformer.coor = coor
            transformer.mask(rmask)
            rscc_set[i] = self.rscc(conformer, rmask)
        mask = xmap_calc.array > 0
        map_o = self.xmap.array[mask]
        Residual = map_o
        order = (-rscc_set).argsort()
        with open(self.fname, "w") as f:
            f.write("Conf.\t AIC\t AIC2\t BIC\t BIC2\t Fisher Z-score\n")
            metrics = []
            for i, idx in enumerate(order):
                conformer.coor = coor_set[idx]
                try:
                    multiconformer = multiconformer.combine(conformer)
                except Exception:
                    multiconformer = Structure.fromstructurelike(conformer.copy())
                xmap_calc.array.fill(0)
                transformer.density()
                map_calc = transformer.xmap.array[mask]
                # Calculate the Residual Sum of Squares (RSS)
                Residual = Residual - occupancies[idx] * map_calc
                RSS = np.sum(np.square(Residual))
                # Formula for the AIC: AIC = 2k + n*ln(RSS)
                # This is the formula according to qFit 2.0:
                aic = 2 * (i + 1) + len(map_o) * np.log(RSS)
                bic = len(map_o) * np.log(RSS / len(map_o)) + (i + 1) * np.log(
                    len(map_o)
                )
                # This is using an adjusted formula:
                aic2 = 2 * (i + 1) * 4 * len(conformer.coor) + len(map_o) * np.log(RSS)
                bic2 = len(map_o) * np.log(RSS / len(map_o)) + 4 * len(
                    conformer.coor
                ) * (i + 1) * np.log(len(map_o))
                # Using fish z-transform:
                z = self.fisher_z(multiconformer, rmask)
                sigma_z = 1 / np.sqrt(len(map_o) - 3)
                CI_z = [
                    z - st.norm.ppf(confidence) * sigma_z,
                    z + st.norm.ppf(confidence) * sigma_z,
                ]
                CI_r = [
                    (np.exp(2 * bound_z) - 1) / (np.exp(2 * bound_z) + 1)
                    for bound_z in CI_z
                ]
                # Where k is the number of parameters, n is the number of observations and RSS is the residual sum of squares.
                f.write(
                    "{}\t{:9.2f}\t{:9.2f}\t{:9.2f}\t{:9.2f}\t{}\n".format(
                        chr(65 + idx), aic, aic2, bic, bic2, CI_r
                    )
                )
                metrics.append([idx + 1, aic, aic2, bic, bic2, CI_r])
            f.close()
            return metrics


        
    def sample_points(radius=1.0, step=0.2):
        """Generate offsets within a sphere of given radius (Å) with a cubic grid of step (Å)."""
        pts = []
        r2 = radius * radius
        # Include center point
        pts.append((0.0, 0.0, 0.0))
        # 3D grid
        rng = np.arange(-radius, radius + 1e-6, step)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 <= r2:
                        pts.append((float(dx), float(dy), float(dz)))
        return np.array(pts, dtype=np.float64)


    def gaussian_weights(offsets: np.ndarray, sigma: float = 0.35):
        """Compute isotropic Gaussian weights for each offset vector based on distance from origin."""
        # Typical sigma tuned to give strong emphasis near the atom center.
        dists2 = np.sum(offsets**2, axis=1)
        w = np.exp(-0.5 * dists2 / (sigma * sigma))
        return w


    def pearson_corr(x: np.ndarray, y: np.ndarray):
        if x.size < 3 or y.size < 3:
            return np.nan
        xm = x - np.mean(x)
        ym = y - np.mean(y)
        denom = (np.linalg.norm(xm) * np.linalg.norm(ym))
        if denom == 0:
            return np.nan
        return float(np.dot(xm, ym) / denom)


    def edia_like_for_atom(atom, xmap, offsets, w_sigma=0.35) -> dict:
        """Compute EDIA-like score for a single atom by sampling map around it and correlating with a Gaussian profile."""
        # Map normalization (global z-score) once per grid could be faster, but we keep this per-atom flexible.
        # Here, we'll compute global mean/sigma lazily the first time via static attributes.
        if not hasattr(edia_like_for_atom, "_map_stats"):
            arr = xmap.array
            mu = float(arr.mean())
            sd = float(arr.std(ddof=0))
            if sd == 0:s
                sd = 1.0
            edia_like_for_atom._map_stats = (mu, sd)
        mu, sd = edia_like_for_atom._map_stats

        # Atom position in Cartesian
        pos = atom.coor  # Assuming atom has a 'coor' attribute for coordinates
        # Prepare weights
        offsets = sample_points(radius=1.0, step=0.2)
        weights = gaussian_weights(offsets, sigma=w_sigma)
        # Sample map at offsets
        vals = []
        for off in offsets:
            p = pos + off
            v = xmap.interpolate_value(p)
            vals.append(v)
        vals = np.array(vals, dtype=np.float64)

        if normalize_map:
            vals = (vals - mu) / sd

        # Idealized profile is simply the (normalized) Gaussian weights (rescaled to mean=0, unit var for correlation)
        ideal = weights
        # Compute Pearson correlation as the core fit metric
        corr = pearson_corr(vals, ideal)
        # Map correlation (-1..1) to [0,1]
        if math.isnan(corr):
            score = np.nan
        else:
            score = 0.5 * (corr + 1.0)

        # Also report center density (z-scored if normalized)
        v_center = xmap.interpolate_value(pos)
        if normalize_map:
            v_center = (v_center - mu) / sd

        return {
            "edia_like": score,
            "n_samples": int(len(vals)),
            "center_density": float(v_center),
        }   

    
