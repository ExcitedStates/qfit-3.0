import logging

import numpy as np

from .transformer import Transformer, FFTTransformer


logger = logging.getLogger(__name__)


class MapScaler:

    def __init__(self, xmap, scattering='xray'):
        self.xmap = xmap
        self.scattering = scattering
        self._model_map = xmap.zeros_like(xmap)

    def subtract(self, structure):
        if self.xmap.hkl is not None:
            hkl = self.xmap.hkl
            transformer = FFTTransformer(
                structure, self._model_map, hkl=hkl, scattering=self.scattering)
        else:
            transformer = Transformer(
                structure, self._model_map, simple=True,
                rmax=3, scattering=self.scattering)
        logger.info("Subtracting density.")
        transformer.density()
        self.xmap.array -= self._model_map.array

    def scale(self, structure, radius=1):

        if self.xmap.hkl is not None:
            hkl = self.xmap.hkl
            transformer = FFTTransformer(structure, self._model_map,
                                         hkl=hkl, scattering=self.scattering)
        else:
            transformer = Transformer(structure, self._model_map, simple=True,
                                      rmax=3, scattering=self.scattering)
        transformer.mask(radius)
        mask = self._model_map.array > 0
        xmap_masked = self.xmap.array[mask]
        xmap_masked_mean = xmap_masked.mean()
        xmap_masked -= xmap_masked_mean

        transformer.reset(full=True)
        transformer.density()
        model_masked = self._model_map.array[mask]
        model_masked_mean = model_masked.mean()
        model_masked -= model_masked_mean

        scaling_factor = np.dot(model_masked, xmap_masked) / np.dot(xmap_masked, xmap_masked)
        logger.info(f"Map scaling factor: {scaling_factor:.2f}")
        self.xmap.array -= xmap_masked_mean
        self.xmap.array *= scaling_factor
        self.xmap.array += model_masked_mean
        transformer.reset(full=True)

    def cutoff(self, cutoff_value, value=-1):
        cutoff_mask = self.xmap.array < cutoff_value
        self.xmap.array[cutoff_mask] = value
        logger.info(f"Map absolute cutoff value: {cutoff_value:.2f}")
