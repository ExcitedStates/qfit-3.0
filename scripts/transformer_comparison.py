"""
Script to compare the behavior of the three different transformer
implementations given the same model/phenix.fmodel input
"""
import os
import os.path as op
import sys

import numpy as np

from qfit.structure import Structure
from qfit.xtal.volume import XMap
from qfit.xtal.transformer import get_transformer
from qfit.utils.mock_utils import create_fmodel

def run(argv):
    pdb_file = op.abspath(argv[1])
    s = Structure.fromfile(pdb_file)
    mtz_file = create_fmodel(pdb_file, 1.0)
    #print(f"fmodel mtz_file is {mtz_file}")
    x = XMap.fromfile(mtz_file, label="FWT,PHIFWT")
    print(f"SHAPE={x.shape}")
    x1, x2, x3 = XMap.zeros_like(x), XMap.zeros_like(x), XMap.zeros_like(x)
    t1 = get_transformer("cctbx", s, x1)
    t2 = get_transformer("qfit", s, x2)
    t3 = get_transformer("fft", s, x3)
    t1.density()
    t2.density()
    t3.density()
    print(f"qfit_cctbx: min={np.min(x1.array)} max={np.max(x1.array)} mean={np.mean(x1.array)}")
    print(f"qfit_classic: min={np.min(x2.array)} max={np.max(x2.array)} mean={np.mean(x2.array)}")
    print(f"qfit_cctbx_fft: min={np.min(x3.array)} max={np.max(x3.array)} mean={np.mean(x3.array)}")
    cc_cvq = np.corrcoef(x1.array.flatten(), x2.array.flatten())[0][1]
    cc_cvf = np.corrcoef(x1.array.flatten(), x.array.flatten())[0][1]
    cc_qvf = np.corrcoef(x2.array.flatten(), x.array.flatten())[0][1]
    cc_tvq = np.corrcoef(x3.array.flatten(), x2.array.flatten())[0][1]
    cc_tvc = np.corrcoef(x3.array.flatten(), x1.array.flatten())[0][1]
    print(f"CC(qfit_cctbx,qfit_classic)={cc_cvq}")
    print(f"CC(qfit_cctbx,mmtbx_fmodel)={cc_cvf}")
    print(f"CC(qfit_classic,mmtbx_fmodel)={cc_qvf}")
    print(f"CC(qfit_cctbx_fft,qfit_classic)={cc_tvq}")
    print(f"CC(qfit_cctbx_fft,qfit_cctbx)={cc_tvc}")
    x.tofile("density_fmodel.ccp4")
    x1.tofile("density_cctbx.ccp4")
    x2.tofile("density_qfit.ccp4")

if __name__ == "__main__":
    run(sys.argv)
