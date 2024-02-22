
from __future__ import absolute_import, division, print_function
from libtbx.utils import Sorry
from libtbx.str_utils import make_header
from libtbx import Auto, adopt_init_args
from libtbx import easy_mp
import os.path
import random
import time
import sys
from six.moves import range


def write_partial_omit_map(
      fmodel,
      selection,
      selection_delete=None,
      map_file_name="partial_omit_map_coeffs.mtz",
      partial_occupancy=0.5):
  """
  Generate an mFo-DFc map with a selection of atoms at reduced occupancy.
  Will write the map coefficients (along with 2mFo-DFc map) to an MTZ file
  if desired.  Reflections flagged for calculating R-free will always be
  omitted.
  """
  import iotbx.map_tools
  assert (0 <= partial_occupancy <= 1.0)
  xrs = fmodel.xray_structure
  occ = xrs.scatterers().extract_occupancies()
  occ.set_selected(selection, partial_occupancy)
  xrs_tmp = xrs.deep_copy_scatterers()
  xrs_tmp.set_occupancies(occ)
  if (selection_delete is not None):
    xrs_tmp = xrs_tmp.select(~selection_delete)
  fmodel.update_xray_structure(xrs_tmp, update_f_calc=True)
  fofc_coeffs = fmodel.map_coefficients(map_type="mFo-DFc",
    exclude_free_r_reflections=True)
  two_fofc_coeffs = fmodel.map_coefficients(map_type="2mFo-DFc",
    exclude_free_r_reflections=True)
  if map_file_name is not None:
    iotbx.map_tools.write_map_coeffs(two_fofc_coeffs, fofc_coeffs,
      map_file_name)
  fmodel.update_xray_structure(xrs, update_f_calc=False)
  return two_fofc_coeffs, fofc_coeffs


def get_master_phil():
  from mmtbx.command_line import generate_master_phil_with_inputs
  return generate_master_phil_with_inputs(
    enable_twin_law=True,
    enable_experimental_phases=True,
    enable_pdb_interpretation_params=False,
    enable_stop_for_unknowns=False,
    enable_full_geometry_params=True,
    phil_string="""
selection = None
  .type = atom_selection
whole_residues = False
  .type = bool
  .help = If the initial selection includes partial residues, expand it to \
    include each residue in its entirety.
selection_delete = None
  .type = atom_selection
  .help = Designates atoms to be removed from the structure before \
    calculating the target map.
target_map = *mFo-DFc 2mFo-DFc
  .type = choice
occ = 0.5
  .type = float
  .help = Partial occupancy for selected atoms for map calculation.
output {
  map_file_name = partial_omit_map_coeffs.mtz
    .type = path
  verbose = False
    .type = bool
  debug = False
    .type = bool
}
""")

def run(args, out=sys.stdout):
  import mmtbx.command_line
  import mmtbx.building
  import iotbx.pdb.hierarchy
  cmdline = mmtbx.command_line.load_model_and_data(
    args=args,
    master_phil=get_master_phil(),
    process_pdb_file=False,
    create_fmodel=True,
    out=out,
    usage_string="""\
generate_partial_omit_map.py model.pdb data.mtz selection="resname ATP" [occ=0.6]

Compute 2mFo-DFc and mFo-DFc maps for a structure where a selection of atoms
has been reduced in occupancy, to "unmask" partially superimposed secondary
conformations in difference density.

Citation: Fraser et al. Mol Microbiol. 2007 Jul 1; 65(2): 319–332.
""")
  params = cmdline.params
  fmodel = cmdline.fmodel
  validate_params(params)
  pdb_hierarchy = cmdline.pdb_hierarchy
  make_header("Computing partial omit map", out=out)
  selection = pdb_hierarchy.atom_selection_cache().selection(params.selection)
  if (params.whole_residues):
    selection = iotbx.pdb.atom_selection.expand_selection_to_entire_atom_groups(
      selection=selection,
      pdb_atoms=pdb_hierarchy.atoms())
  n_sel = selection.count(True)
  assert (n_sel > 0)
  print("%d atoms selected" % n_sel, file=out)
  selection_delete = None
  if (params.selection_delete is not None):
    selection_delete = model.selection(params.selection_delete)
  two_fofc_map, fofc_map = write_partial_omit_map(
    fmodel=fmodel,
    selection=selection,
    selection_delete=selection_delete,
    map_file_name=params.output.map_file_name,
    partial_occupancy=params.occ)
  print(f"Wrote 2mFo-DFc and mFo-DFc maps to {params.output.map_file_name}")
  return 0

def validate_params(params):
  if (params.selection is None):
    raise Sorry("You must specificy an atom selection to reduce occupancy.")

if (__name__ == "__main__"):
  run(sys.argv[1:])
