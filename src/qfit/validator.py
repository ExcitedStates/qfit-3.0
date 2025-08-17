import os
import math
import numpy as np
import scipy.stats as st

from qfit.xtal.volume import XMap
from qfit.xtal.transformer import get_transformer
from qfit.structure import Structure

class Validator:
    def __init__(self, xmap, resolution, directory, em=False, transformer="qfit"):
        self.xmap = xmap
        self.resolution = resolution
        self.em = em
        self.fname = os.path.join(directory, "validation_metrics.txt")
        self._transformer = transformer

    def _get_transformer(self, *args, **kwds):
        return get_transformer(self._transformer, *args, **kwds)

    def rscc(self, structure, rmask=1.5, mask_structure=None, simple=True):
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        if mask_structure is None:
            transformer = self._get_transformer(structure, model_map, simple=simple, em=self.em)
        else:
            transformer = self._get_transformer(
                mask_structure, model_map, simple=simple, em=self.em
            )
        transformer.mask(rmask)
        mask = model_map.array > 0
        model_map.array.fill(0)
        if mask_structure is not None:
            transformer = self._get_transformer(structure, model_map, simple=simple, em=self.em)
        transformer.density()

        corr = np.corrcoef(self.xmap.array[mask], model_map.array[mask])[0, 1]
        return corr

    def fisher_z(self, structure, rmask=1.5, simple=True):
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        transformer = self._get_transformer(structure, model_map)
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
            sigma = 1.0 / np.sqrt(mv / self.resolution.high - 3)  # pylint: disable=unused-variable
        else:
            sigma = 1.0 / np.sqrt(mv - 3)  # pylint: disable=unused-variable
        fisher = 0.5 * np.log((1 + corr) / (1 - corr))
        return fisher

    def fisher_z_difference(self, structure1, structure2, rmask=1.5, simple=True):
        # Create mask of combined structures
        combined = structure1.combine(structure2)
        model_map = XMap.zeros_like(self.xmap)
        model_map.set_space_group("P1")
        transformer = self._get_transformer(combined, model_map)
        transformer.mask(rmask)
        mask = model_map.array > 0
        nvoxels = mask.sum()
        # mv = nvoxels * self.xmap.voxel_volume
        mv = nvoxels * np.product(self.xmap.voxelspacing) * self.xmap.unit_cell.calc_v()

        # Get density values of xmap, and both structures
        target_values = self.xmap.array[mask]
        transformer = self._get_transformer(structure1, model_map, simple=simple)
        model_map.array.fill(0)
        transformer.density()
        model1_values = model_map.array[mask]
        transformer = self._get_transformer(structure2, model_map, simple=simple)
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
        transformer = self._get_transformer(conformer, xmap_calc)
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


        
    def sample_points(self, radius=1.0, step=0.2):
        """Generate offsets within a sphere of given radius (Å) on a cubic grid (step in Å)."""
        r2 = radius * radius
        rng = np.arange(-radius, radius + 1e-9, step)

        pts = [(0.0, 0.0, 0.0)]  # always include the center
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 <= r2:
                        pts.append((float(dx), float(dy), float(dz)))
        return np.asarray(pts, dtype=np.float64)

    def gaussian_weights(self, offsets: np.ndarray, sigma: float = 0.35):
        """
        Isotropic 3D Gaussian profile centered at the atom. The absolute scaling
        is irrelevant for correlation, but we normalize to stabilize numerics.
        """
        d2 = np.sum(offsets**2, axis=1)
        w = np.exp(-0.5 * d2 / (sigma * sigma))
        s = float(np.linalg.norm(w))
        if s > 0:
            w = w / s
        return w

    def _pearson_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Numerically stable Pearson correlation for 1-D arrays. Returns np.nan if undefined."""
        if x.size < 3 or y.size < 3:
            return np.nan
        xm = x - x.mean()
        ym = y - y.mean()
        nx = float(np.linalg.norm(xm))
        ny = float(np.linalg.norm(ym))
        if nx == 0.0 or ny == 0.0:
            return np.nan
        return float(np.dot(xm, ym) / (nx * ny))

    def _global_map_stats(self):
        """Cache and return global mean/std of the map for optional z-scoring."""
        if not hasattr(self, "_map_stats"):
            arr = self.xmap.array
            mu = float(arr.mean())
            sd = float(arr.std(ddof=0))
            if sd == 0.0 or math.isnan(sd):
                sd = 1.0
            self._map_stats = (mu, sd)
        return self._map_stats

    def edia_like_for_atom(self, atom, radius=1.0, step=0.2, w_sigma=0.35, normalize_map=True) -> dict:
        """
        EDIA-like score for a single atom using XMap.interpolate(xyz):
        1) sample map around the atom on a small spherical grid,
        2) correlate (Pearson) sampled map with a Gaussian ideal profile,
        3) map correlation from [-1,1] to [0,1].
        """
        # Offsets (Å) and Gaussian template
        offsets = self.sample_points(radius=radius, step=step)
        ideal = self.gaussian_weights(offsets, sigma=w_sigma)

        # Sample map using ONLY xmap.interpolate
        pos = atom.coor  # Cartesian Å, shape (3,)
        vals = np.empty(len(offsets), dtype=np.float64)
        for i, off in enumerate(offsets):
            vals[i] = float(self.xmap.interpolate(pos + off))

        # Optional global z-score of map samples
        if normalize_map:
            mu, sd = self._global_map_stats()
            vals = (vals - mu) / sd

        # Pearson correlation (robust)
        corr = self._pearson_corr(vals, ideal)

        # Map to [0,1] and keep NaN if undefined
        score = np.nan if math.isnan(corr) else 0.5 * (corr + 1.0)

        # Center density via interpolate at the atom center
        v_center = float(self.xmap.interpolate(pos))
        if normalize_map:
            mu, sd = self._global_map_stats()
            v_center = (v_center - mu) / sd

        return {
            "edia_like": float(score) if not math.isnan(score) else np.nan,
            "n_samples": int(vals.size),
            "center_density": float(v_center),
        }

    def edia_like_for_structure(self, structure, atom_selector=None, **kwargs):
        """
        Compute EDIA-like for many atoms.
        `atom_selector` can be a callable(atom)->bool or None (all atoms).
        Returns list of (atom, metrics_dict).
        """
        results = []
        for atom in structure.atoms:
            if atom_selector is not None and not atom_selector(atom):
                continue
            results.append((atom, self.edia_like_for_atom(atom, **kwargs)))
        return results
