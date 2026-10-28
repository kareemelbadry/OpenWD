"""Immutable predecessor paths for the combined PR #52/#54 fixed controls."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
APPROVED = DATA / "approved_regressions"
COMBINED = APPROVED / "hydrogen_combined_2026_10_28"
DISSOLUTION = APPROVED / "hydrogen_dissolution_2026_10_27"
MONTREAL = APPROVED / "montreal_h_radius_2026_10_27"


def main_control_path(name):
    """The approved PR #54 output, before adding full hydrogen dissolution."""
    if name.startswith("dah-"):
        _, preset, target = name.split("-", 2)
        return MONTREAL / ("dah_" + preset) / (target + ".npz")
    if name in {"da-20000", "dab-20000", "dz-pg1225", "dz-j0738"}:
        return MONTREAL / "fixed" / (name + ".npz")
    if name == "public-da-12000-public":
        return MONTREAL / "da_cubic/da-12000-public.npz"
    if name == "public-da-20000":
        return DATA / "da_cubic_regressions/da-20000.npz"
    if name == "dab-9000":
        return APPROVED / "stark_density_floor_2026_10_05/fixed/dab-9000.npz"
    if name == "daz-g149_28":
        return MONTREAL / "daz/daz-g149_28.npz"
    if name == "daz-galex1931":
        return APPROVED / "independent_kurucz_2026_10_14/fixed/daz-galex1931.npz"
    raise ValueError(f"unknown hydrogen integration control: {name}")
