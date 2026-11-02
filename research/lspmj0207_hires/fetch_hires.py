#!/usr/bin/env python3
"""Download the raw (level-0) Keck/HIRES frames needed to re-extract LSPM J0207+3331.

Night UT 2020-10-08, program U061 (PI Melis), C5 decker, UV collimator, 2x1 binning,
ECHANGL = -0.00317, XDANGL = 1.078.  These are the data of Le Bourdais & Dufour (2025).

Why raw frames: the KOA automatic (level-1) extraction of this faint, cool target
traced only 4 blue-CCD orders, losing most of 3130-4000 A.  The frames below let
MAKEE re-extract every order, using a bright star of the same setup for the trace.

usage: python -I fetch_hires.py OUTDIR
"""
import sys
from pathlib import Path

import time
import urllib.request

FRAMES = {
    "science": ["HI.20201008.32858.fits", "HI.20201008.36786.fits"],  # LSPM J0207+3331, 2 x 3600 s
    "science_short": ["HI.20201008.36687.fits"],                      # 36 s, not used
    "trace_star": ["HI.20201008.17118.fits"],                         # BD+28 4211, 300 s, same XDANGL
    "response": ["HI.20201008.55363.fits"],                           # EGGR 180 (DC), 900 s
    "arc": ["HI.20201008.40554.fits", "HI.20201008.47629.fits"],      # ThAr, same XDANGL as science
    "flat": [f"HI.20201008.{n:05d}.fits" for n in
             (8715, 8786, 8857, 8927, 8999, 9068, 9139, 9210, 9279, 9350, 9421)],
    "bias": [f"HI.20201008.{n:05d}.fits" for n in
             (9887, 9939, 9990, 10042, 10095, 10147, 10199, 10251, 10303, 10355, 10406)],
}

out = Path(sys.argv[1]).resolve() / "lev0"
out.mkdir(parents=True, exist_ok=True)
URL = "https://koa.ipac.caltech.edu/cgi-bin/getKOA/nph-getKOA?filehand=/koadata13/HIRES/20201008/lev0/{}"
for group, koaids in FRAMES.items():
    for koaid in koaids:
        target = out / koaid
        if target.exists() and target.stat().st_size > 2_000_000:
            continue
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(URL.format(koaid), target)
                with open(target, "rb") as handle:
                    assert handle.read(6) == b"SIMPLE", "not a FITS file"
                break
            except Exception as error:  # KOA connections drop occasionally
                print(f"retry {koaid}: {error}", flush=True)
                time.sleep(5)
        else:
            sys.exit(f"failed to download {koaid}")
        print(group, koaid, target.stat().st_size, flush=True)
