'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import logging

import numpy as np
from .atomsf import ATOM_STRUCTURE_FACTORS
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
    
        cutoff_dict = {}
        four_pi2 = 4 * np.pi * np.pi
        for atom in ["O", "C", "N", "S"]:
            asf = ATOM_STRUCTURE_FACTORS[atom]
            density = np.zeros(100)
            bfactor = np.linspace(1,100,100)
            for i in range(6):
                mask = asf[1][i] + bfactor > 1e-4
                bw = -four_pi2 / (asf[1][i] + bfactor[mask])
                density[mask] += (asf[0][i] * (-bw / np.pi) ** 1.5)
            density -= xmap_masked_mean
            density *= scaling_factor
            density += model_masked_mean
            density *= 0.15
            cutoff_dict[atom] = density
        return cutoff_dict
        

    def cutoff(self, cutoff_value, value=-1):
        cutoff_mask = self.xmap.array < cutoff_value
        self.xmap.array[cutoff_mask] = value
        logger.info(f"Map absolute cutoff value: {cutoff_value:.2f}")
