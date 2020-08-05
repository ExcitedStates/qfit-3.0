#from Pavel Afonine

import sys
from iotbx import reflection_file_reader
import iotbx.pdb
import mmtbx.utils
from cStringIO import StringIO
from iotbx import reflection_file_utils
from cctbx import maptbx
import mmtbx.map_tools
from libtbx import group_args

def compute(f_obs, r_free_flags, xrs, use_new):
  fmodel = mmtbx.f_model.manager(
    f_obs          = f_obs.deep_copy(),
    r_free_flags   = r_free_flags.deep_copy(),
    xray_structure = xrs.deep_copy_scatterers())
  fmodel.update_all_scales(
    update_f_part1=False, remove_outliers=True, fast=True)
  if(use_new): 
    result = mmtbx.bulk_solvent.multi_mask_bulk_solvent(
      fmodel = fmodel).fmodel_result
  else:        
    result = fmodel
  rw  = result.r_work()
  rf  = result.r_free()
  rwl = result.r_work_low()
  rwh = result.r_work_high()
  mc_diff = mmtbx.map_tools.electron_density_map(
    fmodel = result).map_coefficients(
      map_type         = "mFo-DFc",
      isotropize       = True,
      fill_missing     = False)
  return group_args(
    rw=rw, rf=rf, rwl=rwl, rwh=rwh, mc_diff=mc_diff)

def run(args):
  pdb_file_name, hkl_file_name = args
  pdb_inp = iotbx.pdb.input(file_name = pdb_file_name)
  xrs = pdb_inp.xray_structure_simple()
  xrs.scattering_type_registry(table = "wk1995")
  #
  reflection_file = reflection_file_reader.any_reflection_file(
    file_name=hkl_file_name, ensure_read_access=False)
  rfs = reflection_file_utils.reflection_file_server(
    crystal_symmetry=xrs.crystal_symmetry(),
    force_symmetry=True,
    reflection_files=[reflection_file],
    err=StringIO())
  determine_data_and_flags_result = mmtbx.utils.determine_data_and_flags(
    reflection_file_server  = rfs,
    keep_going              = True,
    log                     = StringIO())
  f_obs = determine_data_and_flags_result.f_obs
  sel = f_obs.data()>0
  f_obs = f_obs.select(selection=sel)
  #
  r_free_flags = determine_data_and_flags_result.r_free_flags
  if r_free_flags is None:
    print("No R-free flags available.")
    return
  merged = f_obs.as_non_anomalous_array().merge_equivalents()
  f_obs = merged.array().set_observation_type(f_obs)
  
  merged = r_free_flags.as_non_anomalous_array().merge_equivalents()
  r_free_flags = merged.array().set_observation_type(r_free_flags)
  f_obs, r_free_flags = f_obs.common_sets(r_free_flags)
  
  crystal_gridding = f_obs.crystal_gridding(
    d_min             = f_obs.d_min(),
    symmetry_flags    = maptbx.use_space_group_symmetry,
    resolution_factor = 1./4)
  
  sel = xrs.hd_selection()
  xrs = xrs.select(~sel)
  sel = xrs.scatterers().extract_occupancies()>0.
  xrs = xrs.select(sel)
  
  old = compute(f_obs=f_obs, r_free_flags=r_free_flags, xrs=xrs, use_new=False)
  new = compute(f_obs=f_obs, r_free_flags=r_free_flags, xrs=xrs, use_new=True)
  
  print("OLD:Rw_l/Rw_h/Rw/Rf %6.4f %6.4f %6.4f %6.4f"%(old.rwl, old.rwh, old.rw, old.rf))
  print("NEW:Rw_l/Rw_h/Rw/Rf %6.4f %6.4f %6.4f %6.4f"%(new.rwl, new.rwh, new.rw, new.rf))
  #
  mtz_dataset = old.mc_diff.as_mtz_dataset(column_root_label="FoFc_old")
  mtz_dataset.add_miller_array(
    miller_array      = new.mc_diff,
    column_root_label = "FoFc_new")
  mtz_object = mtz_dataset.mtz_object()
  mtz_object.write(file_name = "FoFc.mtz")

if (__name__ == "__main__"):
  run(args=sys.argv[1:])

