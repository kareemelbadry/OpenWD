"""Explicit data and source locations for the checkout-only cool workflows."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def data_directory():
    """External research tables; never search another checkout implicitly."""
    return Path(os.environ.get("OPENWD_RESEARCH_DATA", REPOSITORY / ".cache" / "molecular-opacity")).expanduser().resolve()


def source_paths():
    """Archive this checkout's sources, rejecting an accidentally installed copy."""
    import wd_spectra
    actual = Path(wd_spectra.__file__).resolve()
    if not actual.is_relative_to(REPOSITORY / "src"):
        raise RuntimeError("Cool research workflows require this checkout: pip install -e .")
    return sorted([p.relative_to(REPOSITORY) for p in
        list((REPOSITORY / "research/cool_models").glob("*.py"))
        + list((REPOSITORY / "src/wd_spectra").rglob("*.py"))])
