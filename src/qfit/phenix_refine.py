"""
Phenix extensions
"""

import subprocess
import os

def run_phenix_aniso(structure,
                     chain_id,
                     resid,
                     prev_resid,
                     high_resolution,
                     options):
    # Identify which atoms to refine anisotropically:
    if high_resolution < 1.45:
        adp = "not (water or element H)"
    else:
        adp = f"chain {chain_id} and resid {resid}"

    # Generate the parameter file for phenix refinement:
    labels = options.label.split(",")
    with open(f"chain_{chain_id}_res_{resid}_adp.params", "w") as params:
        params.write(f"""
            refinement {{
              electron_density_maps {{
                map_coefficients {{
                  mtz_label_amplitudes = {labels[0]}
                  mtz_label_phases = {labels[1]}
                  map_type = 2mFo-DFc
                }}
              }}
              refine {{
                strategy = *individual_sites *individual_adp
                adp {{
                  individual {{
                    anisotropic = {adp}
                  }}
                }}
              }}
            }}""")
    # Set the occupancy of the side chain to zero for omit map calculation
    out_root = f"out_{chain_id}_{resid}"
    structure.tofile(f"{out_root}.pdb")
    # FIXME this should just call mmtbx directly
    subprocess.run(
        [
            "phenix.pdbtools",
            "modify.selection="
            f'"chain {chain_id} and '
            f"( resid {resid} and not "
            f"( name n or name ca or name c or name o or name cb ) or "
            f'( resid {prev_resid} and name n ) )"',
            "modify.occupancies.set=0",
            "stop_for_unknowns=False",
            f"{out_root}.pdb",
            f"output.file_name={out_root}_modified.pdb",
        ]
    )

    # Add hydrogens to the structure:
    with open(f"{out_root}_modified_H.pdb", "w") as out_mod_H:
        subprocess.run(
            ["phenix.reduce", f"{out_root}_modified.pdb"], stdout=out_mod_H
        )

    # Generate CIF file of unknown ligands for refinement:
    subprocess.run(["phenix.elbow", "--do_all", f"{out_root}_modified_H.pdb"])

    # Run the refinement protocol:
    if os.path.isfile(f"elbow.{out_root}_modified_H_pdb.all.001.cif"):
        elbow = f"elbow.{out_root}_modified_H_pdb.all.001.cif"
        subprocess.run(
            [
                "phenix.refine",
                f"{options.map}",
                f"{out_root}_modified_H.pdb",
                "--overwrite",
                f"chain_{chain_id}_res_{resid}_adp.params",
                f"refinement.input.xray_data.labels=F-obs",
                f"{elbow}",
            ]
        )
    else:
        # Run the refinement protocol:
        subprocess.run(
            [
                "phenix.refine",
                f"{options.map}",
                f"{out_root}_modified_H.pdb",
                "--overwrite",
                f"chain_{chain_id}_res_{resid}_adp.params",
                f"refinement.input.xray_data.labels=F-obs",
            ]
        )
    return (f"{out_root}_modified_H_refine_001.pdb",
            f"{out_root}_modified_H_refine_001.mtz")
