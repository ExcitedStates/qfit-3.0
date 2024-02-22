#!/bin/bash -e
#
# Run qFit on structures used in Fraser et al. 2011
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3182744/

# these are the structures that download and convert easily
readonly STRUCTURES="1wme 2wt4 3k0o 3k0n 3djg 1x6p 3kyw 1kwn 3tgp"

# FIXME it is surprisingly difficult for mmtbx to identify suitable R-free
# flags in all of these
readonly STRUCTURES_NO_RFREE="1a2p 3btk 1bzr 1jxu 4gcr 1bzp 1fdn 4lzt 1bx8 1bz6 1plc 1rgg 9rnt 2jcw 1tgc 1gdu 1i1x 2dfc 1l63 1l90"

# XXX this does not have downloadable structure factors CIF
readonly STRUCTURES_NO_SF="1do6"

# XXX hack to make up for lack of phenix.* launchers in conda dist of cctbx
readonly phenix_cif_as_mtz=$(which phenix.cif_as_mtz || echo "")
if [ -z "$phenix_cif_as_mtz" ]; then
  mkdir -p bin
  echo "#\!/bin/bash -e" > bin/phenix.cif_as_mtz
  echo "mmtbx.python -m mmtbx.command_line.cif_as_mtz \$@" >> bin/phenix.fetch_pdb
  chmod 755 bin/phenix.fetch_pdb
  export PATH=$PATH:$(pwd)/bin
fi

function run_structure {
  PDB_ID="$1"
  MAPS_ARGS="$2"
  echo "Structure: $PDB_ID..."
  if [ ! -f "${PDB_ID}/${PDB_ID}_2mFo-DFc.mtz" ]; then
    echo "  fetching PDB and generating maps..."
    mkdir -p $PDB_ID
    (cd $PDB_ID && \
     iotbx.fetch_pdb --mtz $PDB_ID > fetch.log && \
      mmtbx.python -m mmtbx.command_line.compute_map_coefficients \
      $MAPS_ARGS ${PDB_ID}.mtz ${PDB_ID}.pdb > maps.log)
  fi
  echo "  running qfit_protein..."
  QFIT_ARGS="--label 2FOFCWT,PH2FOFCWT"
  (cd $PDB_ID && \
    qfit_protein $QFIT_ARGS ${PDB_ID}_2mFo-DFc.mtz ${PDB_ID}.pdb > qfit.log)
}

for ID in $STRUCTURES; do
  run_structure $ID
done

# FIXME boooo
for PDB_ID in $STRUCTURES_NO_RFREE; do
  run_structure $ID r_free_flags.generate=True
done
