from __future__ import annotations

import pytest

from wd_spectra import (
    PG1424_BEST_LINE_ATOM_COUNTS,
    mutable_atom_counts,
)


def test_pg1424_best_atom_is_an_immutable_accepted_configuration() -> None:
    assert dict(PG1424_BEST_LINE_ATOM_COUNTS["C"]) == {
        2: 97,
        3: 54,
        4: 1,
    }
    assert dict(PG1424_BEST_LINE_ATOM_COUNTS["O"]) == {
        2: 47,
        3: 83,
        4: 105,
        5: 14,
        6: 1,
    }
    with pytest.raises(TypeError):
        PG1424_BEST_LINE_ATOM_COUNTS["O"][5] = 9  # type: ignore[index]


def test_pg1424_mutable_atom_copy_cannot_change_the_preset() -> None:
    copied = mutable_atom_counts(PG1424_BEST_LINE_ATOM_COUNTS)
    copied["O"][5] = 9
    assert PG1424_BEST_LINE_ATOM_COUNTS["O"][5] == 14
