#!/bin/bash
# Re-extract the HIRES data of LSPM J0207+3331 (and EGGR 180) with MAKEE, one run per CCD and frame.
#   trace : BD+28 4211 (HI.20201008.17118), same ECHANGL/XDANGL as the science frames
#   flat  : mean of 11 quartz flats (prep_makee.py)
#   bias  : median of 11 biases (prep_makee.py)
#   arc   : ThAr HI.20201008.40554 (11:15 UT, right after the science) for LSPM J0207;
#           ThAr HI.20201008.47629 (13:13 UT) for EGGR 180
# MAKEE defaults: optimal extraction, vacuum wavelengths, heliocentric correction.
# usage: MAKEE_SETUP=/path/to/setup_makee.sh bash reduce_makee.sh REDUCTIONDIR
#        (expects REDUCTIONDIR/split from prep_makee.py; MAKEE_SETUP is the script that puts makee on PATH)
set -euo pipefail
source "${MAKEE_SETUP:?set MAKEE_SETUP to the MAKEE setup script}"
export PGPLOT_DEV=/null
red=$(readlink -f "$1")
star=HI.20201008.17118
run() {  # frame arc ccd
    local frame=$1 arc=$2 ccd=$3 dir="$red/makee/${1}_ccd${3}"
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
    makee ${frame}_${ccd}.fits ${star}_${ccd}.fits flatmean_${ccd}.fits ${arc}_${ccd}.fits \
        raw=../../split bias=../../split/biasmed_${ccd}.fits ccd=${ccd} log=makee.log > stdout.txt 2>&1
    echo "$frame ccd$ccd: $(grep -c 'Extracting echelle order' makee.log) orders, $(grep -m1 'All done' makee.log || echo FAILED)"
}
pids=()
for ccd in 1 2 3; do
    run HI.20201008.32858 HI.20201008.40554 $ccd & pids+=($!)
    run HI.20201008.36786 HI.20201008.40554 $ccd & pids+=($!)
    run HI.20201008.55363 HI.20201008.47629 $ccd & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
