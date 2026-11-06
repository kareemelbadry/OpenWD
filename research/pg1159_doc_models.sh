#!/bin/bash
# PG 1159 models for docs/models/PG1159.md, at published parameters (nothing fitted).
#   research/pg1159_doc_models.sh [OUTPUT_DIR] [TAG ...]   (default results/pg1159/docs, all stars)
# Each model is one cold start of the public command (static, standard quality,
# default refinement, gate fallback and Rydberg upgrades) on a 900-9300 A vacuum
# grid with 0.02 A steps. Run one model per thread; the models are independent.
#
# Parameters and mass fractions:
#   PG 1707+427, PG 1424+535  Werner et al. (2015); PG 1424+535 uses the PG1159Config default mixture
#   MCT 0130-1937             Dreizler & Heber (1998); Werner & Rauch (2014)
#   PG 1159-035               Jahn et al. (2007), Ne from Werner et al. (2004)
#   Abell 21, PG 1151-029     Werner et al. (2004); Werner & Herwig (2006), Ne fixed at 0.02
#   PG 1144+005               Werner et al. (2016), Table 1
#   HE 1429-1209              Werner et al. (2004)
#   RX J0122.9-7521           Mackensen et al. (2025), Table 2
set -u
OUT=${1:-results/pg1159/docs}
shift || true
mkdir -p "$OUT"
export OPENWD_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src
GRID="--wavelength-min 900 --wavelength-max 9300 --wavelength-step 0.02"

star() {  # TAG NAME TEFF LOGG [--option ...] [El=X ...]
  local tag=$1 name=$2 teff=$3 logg=$4; shift 4
  if [ -n "${SELECTED:-}" ] && [[ " $SELECTED " != *" $tag "* ]]; then return; fi
  [ -f "$OUT/$tag/spectrum.txt" ] && return
  local fractions=()
  for value in "$@"; do
    case "$value" in --*) fractions+=("$value") ;; *) fractions+=(--mass-fraction "$value") ;; esac
  done
  python examples/one_shot_pg1159.py --teff "$teff" --logg "$logg" --target-name "$name" \
    "${fractions[@]}" $GRID --output "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  echo "=== $tag exit $? $(date +%H:%M:%S)"
}

SELECTED="$*"
star pg1707 "PG 1707+427" 85000 7.5 He=0.52 C=0.45 O=0.03 N=0.01 F=1e-4 Si=3.2e-4 P=1e-5 S=3.2e-4
star mct0130 "MCT 0130-1937" 90000 7.5 He=0.7462 C=0.2239 O=0.0298
star pg1424 "PG 1424+535" 110000 7.0
star pg1159 "PG 1159-035" 140000 7.0 He=0.33 C=0.48 O=0.17 Ne=0.02
star pg1159-radacc "PG 1159-035" 140000 7.0 --radiative-acceleration He=0.33 C=0.48 O=0.17 Ne=0.02
star abell21 "Abell 21" 140000 6.5 He=0.33 C=0.48 O=0.17 Ne=0.02
star pg1151 "PG 1151-029" 140000 6.0 He=0.33 C=0.48 O=0.17 Ne=0.02
star pg1144 "PG 1144+005" 150000 6.5 He=0.38 C=0.57 O=0.016 N=0.015 F=1e-5 Ne=0.02 Si=6.6e-4 S=1e-4 Fe=1.3e-3
star he1429 "HE 1429-1209" 160000 6.0 He=0.38 C=0.54 O=0.06 Ne=0.02
star rxj0122 "RX J0122.9-7521" 175000 7.7 He=0.626 C=0.372 O=0.0027
