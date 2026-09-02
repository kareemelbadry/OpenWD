#!/usr/bin/env python3
"""One command from Teff/log(g) to a DB atmosphere and spectrum."""
from wd_spectra.models.cli import one_shot_main

if __name__ == "__main__":
    one_shot_main("DB")
