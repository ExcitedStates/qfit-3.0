import logging

import numpy as np

from .transformer import Transformer


logger = logging.getLogger(__name__)


class MapScaler:

    def __init__(self, xmap, mask_radius=1.5, scale=True, cutoff=None,
                 subtract=False, scattering='xray'):
        self.xmap = xmap
        self.mask_radius = mask_radius
        self.cutoff = cutoff
        self.subtract = subtract
        self.scattering = scattering
        self.scale = scale
        self._model_map = xmap.zeros_like(xmap)

    def __call__(self, structure):
        # smax doesnt have any impact here.

        # Set values below cutoff to zero, to penalize the solvent more
        if self.cutoff is not None:
            mean = self.xmap.array.mean()
            std = self.xmap.array.std()
            cutoff_value = self.cutoff * std + mean

        transformer = Transformer(structure, self._model_map, simple=True,
                                  rmax=3, scattering=self.scattering)
        if self.scale:
            transformer.mask(self.mask_radius)
            mask = self._model_map.array > 0
            xmap_masked = self.xmap.array[mask]
            xmap_masked_mean = xmap_masked.mean()
            xmap_masked -= xmap_masked_mean

            transformer.reset()
            transformer.density()
            model_masked = self._model_map.array[mask]
            model_masked_mean = model_masked.mean()
            model_masked -= model_masked_mean
            transformer.reset()

            scaling_factor = np.dot(model_masked, xmap_masked) / np.dot(xmap_masked, xmap_masked)
            logger.info(f"Map scaling factor: {scaling_factor:.2f}")
            self.xmap.array -= xmap_masked_mean
            self.xmap.array *= scaling_factor
            self.xmap.array += model_masked_mean

        # Subtract the receptor density from the map
        if self.subtract:
            transformer.density()
            self.xmap.array -= self._model_map.array

        if self.cutoff is not None:
            if self.scale:
                cutoff_value = (cutoff_value - xmap_masked_mean) * scaling_factor + model_masked_mean
            cutoff_mask = self.xmap.array < cutoff_value
            self.xmap.array[cutoff_mask] = 0
            logger.info(f"Map cutoff value: {cutoff_value:.2f}")
