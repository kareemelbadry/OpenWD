import hashlib
from pathlib import Path
import pytest
import research_paths
from run_cool_db import commands
import check_data


def test_data_path_is_explicit_and_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENWD_RESEARCH_DATA", raising=False)
    expected = research_paths.data_directory()
    monkeypatch.chdir(tmp_path)
    assert research_paths.data_directory() == expected
    monkeypatch.setenv("OPENWD_RESEARCH_DATA", str(tmp_path / "tables"))
    assert research_paths.data_directory() == tmp_path / "tables"


def test_dense_recipe_has_no_temperature_switch_or_supplied_state(tmp_path):
    low, audit = commands(5000, tmp_path, tmp_path / "table.npz")
    high, _ = commands(8000, tmp_path, tmp_path / "table.npz")
    assert [x.replace("5000", "8000") for x in low] == high
    assert not any("resume" in x or "predictor" in x or "refine-atmosphere" in x for x in low)
    assert "--physical-only" in low and "--no-continuations" in low
    assert "--mass-conservative-transfer" in low and "--exact-convection-tangent" in low
    assert audit[audit.index("--maximum-wavelength") + 1] == "1e9"


def test_external_data_check_fails_closed(monkeypatch, tmp_path):
    payload = b"unit-test table, not physical data"
    monkeypatch.setattr(check_data, "SHA256", {"table": hashlib.sha256(payload).hexdigest()})
    with pytest.raises(FileNotFoundError):
        check_data.verify(tmp_path)
    (tmp_path / "table").write_bytes(payload)
    check_data.verify(tmp_path)
    (tmp_path / "table").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        check_data.verify(tmp_path)


def test_bundled_hnc_cache_is_the_qualified_generated_table():
    table = Path(__file__).parent / "data/molecular-hnc-electron-domain-table.npz"
    assert hashlib.sha256(table.read_bytes()).hexdigest() == "b72c73cec42f5f2661742bb9a4051283abc36185f5059d213d4ab0db2c8607f4"
