import logging
import os

import numpy as np

from qfit.xtal.transformer import get_transformer, get_fft_transformer


logger = logging.getLogger(__name__)
ENABLE_FFT = os.environ.get("QFIT_ENABLE_FFT", "false").lower() == "true"


class MapScaler:
    def __init__(self, xmap, em=False):
        self.xmap = xmap
        self._model_map = xmap.zeros_like(xmap)
        self.em = em

    def _get_model_transformer(self,
                               structure,
                               transformer="cctbx",
                               enable_fft=ENABLE_FFT):
        if self.xmap.hkl is not None and enable_fft:
            # FIXME this seems like the correct approach, but it currently
            # produces inferior results for CCTBX
            logger.info("HKLs available, will perform full FFT")
            return get_fft_transformer(
                transformer,
                structure,
                self._model_map,
                hkl=self.xmap.hkl,
                em=self.em
            )
        else:
            logger.info("Using simple density transformer")
            return get_transformer(
                transformer,
                structure,
                self._model_map,
                simple=True,
                rmax=3,
                em=self.em,
            )

    def scale(self, structure, radius=1, transformer="cctbx"):
        """
        Compute and apply in place the transformation required to put the
        experimental map on the same scale as the model-computed map,
        and return the scaling factor S and constant k.
        """
        transformer = self._get_model_transformer(structure,
                                                  transformer=transformer)
        # Get all map coordinates of interest:
        logger.info("Masking with radius %f", radius)
        transformer.mask(radius)
        self._model_map.tofile("scaler_mask.ccp4")
        mask = self._model_map.array > 0
        logger.info("Masked %d grid points out of %d", mask.sum(), mask.size)

        # Calculate map based on structure:
        transformer.reset(full=True)
        transformer.density()
        self._model_map.tofile("scaler_model.ccp4")

        # Get all map values of interest
        xmap_masked = self.xmap.array[mask]
        model_masked = self._model_map.array[mask]

        # Get the mean of masked observed and masked calculated map values
        xmap_masked_mean = xmap_masked.mean()
        model_masked_mean = model_masked.mean()

        # Get optimal scaling factor and mean-difference.
        xmap_masked -= xmap_masked_mean
        model_masked -= model_masked_mean
        s2 = np.dot(model_masked, xmap_masked)
        s1 = np.dot(xmap_masked, xmap_masked)
        scaling_factor = s2 / s1
        k = model_masked_mean - scaling_factor * xmap_masked_mean
        logger.info(f"L2 scaling: S = {scaling_factor:.2f}\tk = {k:.2f}")

        # Scale the observed map to the calculated map
        self.xmap.array = scaling_factor * self.xmap.array + k
        self.xmap.tofile("scaled_map.ccp4")
        transformer.reset(full=True)
        return (scaling_factor, k)

    # XXX currently unused
    def subtract(self, structure):
        transformer = self._get_model_transformer(structure)
        logger.info("Subtracting density.")
        transformer.density()
        self.xmap.array -= self._model_map.array

    # XXX currently unused
    def cutoff(self, cutoff_value, value=-1):
        cutoff_mask = self.xmap.array < cutoff_value
        self.xmap.array[cutoff_mask] = value
        logger.info(f"Map absolute cutoff value: {cutoff_value:.2f}")
