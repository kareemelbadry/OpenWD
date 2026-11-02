#!/usr/bin/env python3
"""Prepare the raw HIRES frames of 2020-10-08 for MAKEE.

1. Split every raw 3-CCD mosaic frame into *_1/_2/_3.fits with MAKEE's HIRES2_readwrite.
2. Average the 11 quartz flats per CCD (float32) -> flatmean_N.fits.
3. Median-combine the 11 biases per CCD (float32) -> biasmed_N.fits.

The headers of the combined frames are those of the first input frame, with history cards.

usage: python -I prep_makee.py RAWDIR WORKDIR
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

raw, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
FLATS = [8715, 8786, 8857, 8927, 8999, 9068, 9139, 9210, 9279, 9350, 9421]
BIASES = [9887, 9939, 9990, 10042, 10095, 10147, 10199, 10251, 10303, 10355, 10406]

for frame in sorted(raw.glob("HI.20201008.*.fits")):
    if all((work / f"{frame.stem}_{c}.fits").exists() for c in (1, 2, 3)):
        continue
    link = work / frame.name
    if not link.exists():
        link.symlink_to(frame)
    subprocess.run(["HIRES2_readwrite", link.name, "mode=5"], cwd=work, check=True, capture_output=True)
    link.unlink()


def combine(numbers, ccd, how, name):
    paths = [work / f"HI.20201008.{n:05d}_{ccd}.fits" for n in numbers]
    cube = np.array([fits.getdata(p).astype(np.float64) for p in paths])
    header = fits.getheader(paths[0])
    data = cube.mean(axis=0) if how == "mean" else np.median(cube, axis=0)
    header["HISTORY"] = f"{how} of {len(paths)} frames: " + ",".join(str(n) for n in numbers)
    for key in ("BZERO", "BSCALE", "BLANK"):
        header.remove(key, ignore_missing=True)
    fits.writeto(work / f"{name}_{ccd}.fits", data.astype(np.float32), header, overwrite=True)
    print(f"{name}_{ccd}.fits  median={np.median(data):.1f}  max={data.max():.0f}")


for ccd in (1, 2, 3):
    combine(FLATS, ccd, "mean", "flatmean")
    combine(BIASES, ccd, "median", "biasmed")
