"""Prose-only provenance classification, including the real Git/CI entry point."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/validation_policy.py"
spec = importlib.util.spec_from_file_location("provenance_policy", SCRIPT)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
PROVENANCE = "src/wd_spectra/data/validation/dah/re-j0317-853-vi.json"


@pytest.fixture
def provenance():
    return {"sigma": "empirical 1-sigma", "v_over_i_level": "about 8 percent",
            "sha256": "observation-checksum", "n_pixels": 877,
            "grid_step_angstrom": 3.3, "nested": {"values": [1, 2.0]}}


def contents(before, after):
    return {PROVENANCE: (json.dumps(before), json.dumps(after))}


def test_description_edits_skip_models_but_not_other_numerical_changes(provenance):
    updated = dict(provenance, sigma="formal propagated uncertainty",
                   v_over_i_level="about 5 percent; k=1.578")
    blobs = contents(provenance, updated)
    event = {"pull_request": {"draft": False, "labels": []}}
    assert policy.plan("pull_request", event,
                       [PROVENANCE, "tools/validation_policy.py"], contents=blobs) == (False, False)
    for numerical in ("src/wd_spectra/opacity.py", "tools/validate.py",
                      "tests/data/reference.npz", ".github/workflows/canaries.yml"):
        assert policy.requires_full([PROVENANCE, numerical], contents=blobs)
    assert policy.requires_full([PROVENANCE])  # No blobs means no exemption.
    assert policy.requires_full(["src/wd_spectra/data/other.json"], contents=blobs)
    assert policy.plan("schedule", {}, [PROVENANCE], contents=blobs) == (True, True)
    assert policy.plan("workflow_dispatch", {}, [PROVENANCE], contents=blobs) == (True, True)


@pytest.mark.parametrize("change", [
    {"sha256": "different-data"}, {"n_pixels": 878}, {"n_pixels": 877.0},
    {"grid_step_angstrom": 3.4}, {"nested": {"values": [1, 3.0]}},
    {"nested": {"values": [True, 2.0]}}, {"sigma": 0.1},
    {"v_over_i_level": None}, {"new_field": "unreviewed"},
])
def test_non_descriptive_changes_still_require_models(provenance, change):
    updated = dict(provenance, sigma="formal uncertainty")
    updated.update(change)
    assert policy.requires_full([PROVENANCE], contents=contents(provenance, updated))


@pytest.mark.parametrize("bad", [
    None, b"invalid JSON", b"\xff", '[]', 'null',
    '{"sigma":"a","sigma":"b","v_over_i_level":"c"}',
    '{"sigma":"a","v_over_i_level":"b","value":NaN}',
    '{"sigma":"a","v_over_i_level":"b","value":1e999}',
])
def test_missing_or_malformed_blobs_fail_closed(provenance, bad):
    good = json.dumps(provenance)
    assert not policy.descriptive_json_change(PROVENANCE, good, bad)
    assert not policy.descriptive_json_change(PROVENANCE, bad, good)


def test_key_removal_and_numeric_changes_cannot_hide_in_formatting(provenance):
    updated = copy.deepcopy(provenance)
    del updated["sha256"]
    assert policy.requires_full([PROVENANCE], contents=contents(provenance, updated))
    formatted = json.dumps(provenance, sort_keys=True, indent=4)
    assert not policy.requires_full([PROVENANCE], contents={
        PROVENANCE: (json.dumps(provenance), formatted)})


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
            cwd=tmp_path, stderr=subprocess.DEVNULL, text=True).strip()

    git("init", "-q", "--initial-branch=main")
    git("config", "user.name", "CI policy test")
    git("config", "user.email", "policy@example.invalid")
    return tmp_path, git


def write_provenance(root, data):
    path = root / PROVENANCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def commit(git):
    git("add", "--all")
    git("commit", "--allow-empty", "-qm", "fixture")
    return git("rev-parse", "HEAD")


@pytest.mark.parametrize("operation", ["add", "delete", "rename"])
def test_new_deleted_and_renamed_provenance_files_require_models(repository, provenance, operation,
                                                               monkeypatch):
    root, git = repository
    if operation != "add":
        write_provenance(root, provenance)
    base = commit(git)
    if operation == "add":
        write_provenance(root, provenance)
    elif operation == "delete":
        (root / PROVENANCE).unlink()
    else:
        (root / PROVENANCE).rename(root / "renamed-provenance.json")
    head = commit(git)
    monkeypatch.chdir(root)
    paths, blobs = policy.changed_inputs(base, head)
    assert policy.requires_full(paths, contents=blobs)


def test_real_planner_uses_merge_base_for_both_names_and_contents(repository, provenance):
    root, git = repository
    write_provenance(root, provenance)
    commit(git)
    git("switch", "-qc", "prose-pr")
    write_provenance(root, dict(provenance, sigma="formal propagated error"))
    head = commit(git)
    git("switch", "-q", "main")
    # This numerical change belongs to an advanced base, not to the PR.
    write_provenance(root, dict(provenance, n_pixels=999))
    base = commit(git)
    event = {"pull_request": {"base": {"sha": base}, "head": {"sha": head},
                              "draft": False, "labels": []}}
    event_path, output_path = root / "event.json", root / "github-output"
    event_path.write_text(json.dumps(event))
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=root, text=True, capture_output=True,
        env=dict(os.environ, GITHUB_EVENT_NAME="pull_request", GITHUB_EVENT_PATH=str(event_path),
                 GITHUB_OUTPUT=str(output_path)), check=True)
    planned = json.loads(result.stdout)
    assert planned["changed"] == [PROVENANCE]
    assert planned["required"] is False and planned["requested"] is False
    assert "required=false\n" in output_path.read_text()
    assert "requested=false\n" in output_path.read_text()


@pytest.mark.parametrize("required,returncode", [("false", 0), ("true", 1)])
def test_actual_qualification_gate_accepts_only_a_non_numerical_unrequested_plan(tmp_path,
                                                                             required, returncode):
    workflow = (ROOT / ".github/workflows/canaries.yml").read_text()
    script = textwrap.dedent(workflow.rsplit("        run: |\n", 1)[1])
    result = subprocess.run(
        ["bash", "-e", "-c", script], text=True, capture_output=True,
        env=dict(os.environ, PLAN_RESULT="success", REQUIRED=required, REQUESTED="false",
                 PREFLIGHT_RESULT="skipped", COLD_RESULT="skipped", D6_RESULT="skipped",
                 EVENT_NAME="pull_request", GITHUB_STEP_SUMMARY=str(tmp_path / "summary")))
    assert result.returncode == returncode, result.stdout + result.stderr
    if required == "false":
        assert "No numerical inputs changed" in result.stdout
    else:
        assert "Add the full-validation PR label" in result.stdout
