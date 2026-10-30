"""Exercise validation orchestration with tiny subprocesses, not atmospheres."""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("validate")
policy = module("validation_policy")
sharder = module("ci_shard")


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/limitations.md",
        "tests/test_documentation.py",
        "tests/test_metals.py",
        "examples/generate_spectrum.ipynb",
        "tools/validation_policy.py",
    ],
)
def test_non_numerical_changes_do_not_trigger_cold_models(path):
    assert not policy.requires_full([path])


@pytest.mark.parametrize(
    "path",
    [
        "src/wd_spectra/opacity.py",
        "csrc/rt_core.c",
        "pyproject.toml",
        "tests/data/spectral_regressions/da-3000.npz",
        "tests/test_protected_model_canaries.py",
        "tests/test_dq_spectral_regression.py",
        "tests/test_dq_release.py",
        "tests/test_hot_cold_canary.py",
        "tests/test_sdb_cold_canary.py",
        "tests/test_sdb_spectral_regression.py",
        "tests/test_released_family_canaries.py",
        "tests/test_d6_regression.py",
        "tests/d6_regression_support.py",
        "tools/validate.py",
        ".github/workflows/canaries.yml",
        "unknown-input.bin",
    ],
)
def test_physics_and_unknown_changes_require_full(path):
    assert policy.requires_full(["README.md", path])


def test_draft_and_label_selection_does_not_forge_a_physical_pass():
    event = {"pull_request": {"draft": False, "labels": []}}
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, False)
    event["pull_request"]["labels"] = [{"name": "full-validation"}]
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, True)
    event["pull_request"]["draft"] = True
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, False)
    assert policy.plan("schedule", {}, []) == (True, True)
    assert policy.plan("workflow_dispatch", {}, []) == (True, True)


def test_default_fast_stage_excludes_both_expensive_markers():
    tasks = runner.commands("fast")
    assert tasks["components"][-1] == "not canary and not spectral"
    assert "research/cool_models" in tasks["cool-components"]
    with pytest.raises(ValueError):
        runner.commands("cold", ["unknown"])


def test_dq_fixed_spectrum_and_true_cold_are_separate_tiers():
    cold = runner.commands("cold", ["dq-j1235"])["dq-j1235"]
    assert cold[-1] == runner.COLD_TESTS["dq-j1235"]
    assert "true_cold" in cold[-1]
    assert runner.commands("spectra", ["dq-j1311"])["dq-j1311"][-1] == (
        "tests/test_dq_spectral_regression.py::test_completed_default_retains_j1311_absolute_flux"
    )


def test_every_released_family_has_a_cold_case():
    assert {case.split("-")[0] for case in runner.COLD} >= {
        "da", "daz", "db", "dab", "dz", "dq", "do", "dao", "pg1159", "d6", "dah", "sdb"
    }


def test_sdb_fixed_spectrum_and_cold_start_are_separate_checks():
    assert runner.commands('cold', ['sdb-hd4539'])['sdb-hd4539'][-1] == (
        'tests/test_sdb_cold_canary.py::test_hd4539_public_cold_start_and_research_spectrum'
    )
    assert runner.commands('spectra', ['sdb-hd4539'])['sdb-hd4539'][-1] == (
        'tests/test_sdb_spectral_regression.py::test_hd4539_fixed_spectrum_matches_research_driver'
    )


@pytest.mark.parametrize("variable", ["OPENWD_DATA", "OPENWD_DQ_DATA"])
def test_custom_runtime_data_disables_reuse(variable, monkeypatch, tmp_path):
    monkeypatch.setenv(variable, str(tmp_path))
    assert not runner.numerical_identity(tmp_path)["reuse_eligible"]


def test_plan_never_starts_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_task", lambda *a: pytest.fail("executed"))
    output = tmp_path / "plan"
    assert runner.main(["full", "--plan", "--output", str(output)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert set(planned) == {"fast", "spectra", "cold"}
    assert set(planned["cold"]) == set(runner.COLD)
    assert not output.exists()
    with pytest.raises(SystemExit):
        runner.main(["full", "--case", "da-3000", "--plan"])


def test_failed_subprocess_logs_output_and_cannot_pass(tmp_path):
    result = runner.run_task(
        [sys.executable, "-u", "-c", "print('live failure'); raise SystemExit(3)"],
        tmp_path / "failed",
        "tiny",
        threading.Event(),
        10,
    )
    assert result["status"] == "failed" and result["returncode"] == 3
    assert "live failure" in Path(result["log"]).read_text()


def test_timeout_is_unqualified_not_a_silent_restart(tmp_path):
    result = runner.run_task(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
        tmp_path / "timeout",
        "tiny",
        threading.Event(),
        0.1,
    )
    assert result["status"] == "timed-out"
    assert result["returncode"] != 0


def test_completed_evidence_requires_same_inputs_and_kept_log(tmp_path):
    log = tmp_path / "output.log"
    log.write_text("passed")
    identity = dict(sha256="actual-inputs", reuse_eligible=True)
    previous = dict(
        identity=identity,
        inputs_unchanged_during_run=True,
        tasks={
            "cold/x": dict(
                status="passed", returncode=0, command=["test"], log=str(log)
            )
        },
    )
    assert runner.reusable(previous, identity, "cold/x", ["test"])
    assert not runner.reusable(
        previous, dict(identity, sha256="changed"), "cold/x", ["test"]
    )
    assert not runner.reusable(previous, identity, "cold/x", ["other-test"])
    assert not runner.reusable(
        dict(previous, inputs_unchanged_during_run=False), identity, "cold/x", ["test"]
    )
    previous["tasks"]["cold/x"]["status"] = "interrupted"
    assert not runner.reusable(previous, identity, "cold/x", ["test"])


def test_prose_edits_preserve_numerical_key_but_source_and_data_invalidate(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENWD_DATA", raising=False)
    for name in ("src", "docs", "csrc"):
        (tmp_path / name).mkdir()
    source = tmp_path / "src/new.py"
    source.write_text("x = 1")
    first = runner.numerical_identity(tmp_path)["sha256"]
    (tmp_path / "docs/new.md").write_text("explanation")
    assert runner.numerical_identity(tmp_path)["sha256"] == first
    source.write_text("x = 2")
    second = runner.numerical_identity(tmp_path)["sha256"]
    assert first != second
    (tmp_path / "csrc/kernel.c").write_text("changed kernel")
    assert runner.numerical_identity(tmp_path)["sha256"] != second


@pytest.mark.parametrize("name", ["test_d6_regression.py", "d6_regression_support.py"])
def test_warm_checker_changes_invalidate_completed_evidence(tmp_path, name):
    (tmp_path / "tests").mkdir()
    path = tmp_path / "tests" / name
    path.write_text("before")
    before = runner.numerical_identity(tmp_path)["sha256"]
    path.write_text("after")
    assert runner.numerical_identity(tmp_path)["sha256"] != before


def test_failed_fast_stage_prevents_expensive_stages(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        runner, "numerical_identity", lambda: dict(sha256="same", reuse_eligible=True)
    )
    monkeypatch.setattr(runner, "commands", lambda tier, cases: {tier: [tier]})

    def fail(command, *args):
        called.append(command)
        return dict(status="failed", returncode=1)

    monkeypatch.setattr(runner, "run_task", fail)
    assert runner.main(["full", "--output", str(tmp_path / "run")]) == 1
    assert called == [["fast"]]
    assert not json.loads((tmp_path / "run/summary.json").read_text())[
        "full_qualification"
    ]


def test_pr_warm_regression_and_scheduled_cold_convergence_have_distinct_coverage():
    pr_cases = policy.selected_cold_cases("pull_request", runner.COLD)
    assert set(pr_cases) == set(runner.COLD) - {"d6-j1637"}
    for event in ("schedule", "workflow_dispatch"):
        assert policy.selected_cold_cases(event, runner.COLD) == list(runner.COLD)
    warm = runner.commands("regression", ["d6-j1637"])
    cold = runner.commands("cold", ["d6-j1637"])["d6-j1637"]
    spectrum = runner.commands("spectra", ["d6-j1637"])["d6-j1637"]
    assert warm == runner.commands("regression")
    assert set(warm) == set(runner.REGRESSIONS)
    assert all("two_step_warm_trajectory" in command[-1] for command in warm.values())
    assert "cold_start[d6-j1637]" in cold[-1]
    assert "fixed_atmosphere_spectrum" in spectrum[-1]
    with pytest.raises(ValueError):
        runner.commands("regression", ["da-5000"])


def test_bounded_regression_pass_does_not_report_full_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "numerical_identity", lambda: dict(sha256="same", reuse_eligible=True))
    monkeypatch.setattr(runner, "commands", lambda *args: {"tiny": [sys.executable, "-c", "pass"]})
    output = tmp_path / "regression"
    assert runner.main(["regression", "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "passed" and report["full_qualification"] is False


def test_workflow_preserves_full_cold_dispatch_and_has_no_duplicate_push_trigger():
    workflow = (ROOT / ".github/workflows/canaries.yml").read_text()
    assert "case: ${{ fromJSON(needs.plan.outputs.cold_cases) }}" in workflow
    assert "  push:" not in workflow
    assert "cancel-in-progress:" in workflow
    assert "  qualification:" in workflow and "if: always()" in workflow
    assert 'test "$COLD_RESULT" = success' in workflow
    assert 'test "$D6_RESULT" = success' in workflow
    assert "Full-suite cold-start qualification was not performed" in workflow
    warm = workflow.split("  d6-regression:\n", 1)[1].split("\n  qualification:", 1)[0]
    assert "github.event_name == 'pull_request'" in warm


def test_preflight_tiers_are_independent_and_both_gate_atmosphere_jobs():
    workflow = (ROOT / ".github/workflows/canaries.yml").read_text()
    preflight = workflow.split("  preflight:\n", 1)[1].split("\n  cold-model:", 1)[0]
    tiers = re.search(r"tier:\s*\[([^\]]+)\]", preflight)
    assert tiers, "complete fast and spectral tiers need independent preflight workers"
    assert [tier.strip() for tier in tiers[1].split(",")] == ["fast", "spectra"]
    assert "fail-fast: false" in preflight
    assert "needs: plan" in preflight
    assert "if: needs.plan.outputs.requested == 'true'" in preflight
    commands = re.findall(r"^\s+- run: (python tools/validate.py .+)$", preflight, re.MULTILINE)
    assert len(commands) == 1 and "'${{ matrix.tier }}'" in commands[0]
    assert "--case" not in commands[0]  # Neither full tier may be filtered.
    assert "--output 'results/preflight-${{ matrix.tier }}'" in commands[0]
    assert "if: always()" in preflight  # Keep failed or interrupted diagnostics.
    assert "name: qualification-preflight-${{ matrix.tier }}" in preflight
    assert "path: results/preflight-${{ matrix.tier }}" in preflight
    cold = workflow.split("  cold-model:\n", 1)[1].split("\n  d6-regression:", 1)[0]
    warm = workflow.split("  d6-regression:\n", 1)[1].split("\n  qualification:", 1)[0]
    assert "needs: [plan, preflight]" in cold
    assert "needs: [plan, preflight]" in warm
    assert 'test "$PREFLIGHT_RESULT" = success' in workflow


def test_pr_cold_starts_skip_preflight_only_because_tests_workflow_runs_both_tiers():
    canaries = (ROOT / ".github/workflows/canaries.yml").read_text()
    preflight = canaries.split("  preflight:\n", 1)[1].split("\n  cold-model:", 1)[0]
    cold = canaries.split("  cold-model:\n", 1)[1].split("\n  d6-regression:", 1)[0]
    assert "github.event_name != 'pull_request'" in preflight
    # Scheduled and manual qualification may never start atmospheres before
    # both cheap tiers pass.
    assert "needs.preflight.result == 'success'" in cold
    assert "needs.preflight.result == 'skipped' && github.event_name == 'pull_request'" in cold
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    assert "  pull_request:\n" in tests
    unit = tests.split("  unit-and-component:\n", 1)[1].split("\n  fixed-spectra:", 1)[0]
    spectra = tests.split("  fixed-spectra:\n", 1)[1].split("\n  cool-workflow-components:", 1)[0]
    assert "python -m pytest -m 'not canary and not spectral'" in unit
    assert 'python-version: "3.12"' in spectra
    assert "python tools/validate.py spectra $cases" in spectra
    assert "if: needs.changes.outputs.spectra == 'true'" in spectra
    assert "python -m pytest research/cool_models" in tests


@pytest.mark.parametrize("job,kind", [("unit-and-component", "tests"), ("fixed-spectra", "spectra")])
def test_workflow_shards_cover_every_item_exactly_once(job, kind):
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    block = re.split(r"\n  (?=\S)", tests.split(f"\n  {job}:\n", 1)[1], maxsplit=1)[0]
    matrix = re.search(r"(?:shard|lane): \[([0-9, ]+)\]", block)[1]
    indices = [int(value) for value in matrix.split(",")]
    count = len(indices)
    assert indices == list(range(1, count + 1))
    assert f"python tools/ci_shard.py {kind} ${{{{ matrix." in block and f" {count})" in block
    items = sharder.test_files() if kind == "tests" else sharder.spectra_cases()
    weights = json.loads(sharder.WEIGHTS.read_text())[kind]
    parts = sharder.shards(items, weights, count)
    assert all(parts)
    assert sorted(item for part in parts for item in part) == sorted(items)


def test_sharding_is_a_partition_even_for_unweighted_items():
    items = [f"tests/test_{index}.py" for index in range(11)]
    for count in range(1, 12):
        parts = sharder.shards(items, {items[0]: 50.0}, count)
        assert len(parts) == count and all(parts)
        assert sorted(sum(parts, [])) == sorted(items)
    with pytest.raises(ValueError):
        sharder.shards(items, {}, 12)


def test_shard_discovery_matches_pytest_collection():
    output = subprocess.check_output(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "tests"],
        cwd=ROOT, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    collected = {line.split("::", 1)[0] for line in output.splitlines() if "::" in line}
    assert collected <= set(sharder.test_files())


def test_prose_only_prs_skip_fixed_spectra_but_other_events_do_not(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    event = {"pull_request": {"base": {"sha": "b"}, "head": {"sha": "h"}}}
    monkeypatch.setattr(policy, "changed_inputs", lambda base, head: (["docs/limitations.md"], {}))
    policy.spectra_main("pull_request", event)
    monkeypatch.setattr(policy, "changed_inputs", lambda base, head: (["src/wd_spectra/opacity.py"], {}))
    policy.spectra_main("pull_request", event)
    for event in ("push", "workflow_dispatch"):
        policy.spectra_main(event, {})
    assert outputs.read_text().split() == [
        "spectra=false", "spectra=true", "spectra=true", "spectra=true"]


def test_d6_regression_runs_the_current_reference_with_unchanged_guards(tmp_path, capsys):
    workflow = (ROOT / ".github/workflows/canaries.yml").read_text()
    job = workflow.split("  d6-regression:\n", 1)[1].split("\n  qualification:", 1)[0]
    assert set(runner.REGRESSIONS) == {"d6-j1637"}
    assert "timeout-minutes: 350" in job and "--timeout 19800" in job
    assert "--case d6-j1637 " in job
    output = tmp_path / "plan"
    assert runner.main(["regression", "--case", "d6-j1637", "--plan", "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["regression"] == {"d6-j1637": runner.commands("regression")["d6-j1637"]}
    assert plan["regression"]["d6-j1637"][-1] == runner.D6_REGRESSION_TEST
    assert not output.exists()


def test_every_spectra_node_id_collects():
    nodes = sorted({part for command in runner.commands("spectra").values()
                    for part in command if part.startswith("tests/")})
    output = subprocess.check_output(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", *nodes],
        cwd=ROOT, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    collected = {line for line in output.splitlines() if "::" in line}
    for node in nodes:
        assert any(line == node or line.startswith(node + "[") for line in collected), node


def test_protected_node_ids_still_collect_without_running_atmospheres():
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            *runner.COLD_TESTS.values(),
            runner.D6_REGRESSION_TEST,
        ],
        cwd=ROOT,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    for node in (*runner.COLD_TESTS.values(), *runner.REGRESSIONS.values()):
        assert node in output


@pytest.mark.parametrize('name', ['test_sdb_cold_canary.py', 'test_sdb_spectral_regression.py'])
def test_sdb_numerical_checkers_invalidate_validation_reuse(tmp_path, name):
    checker = tmp_path / 'tests' / name
    checker.parent.mkdir()
    checker.write_text('previous = 1\n')
    before = runner.numerical_identity(tmp_path)['sha256']
    checker.write_text('previous = 2\n')
    assert runner.numerical_identity(tmp_path)['sha256'] != before
