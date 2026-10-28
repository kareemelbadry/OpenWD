"""Conservative CI selection for PR regression checks and full cold runs."""

import json
import os
from pathlib import PurePosixPath
import subprocess


# These fields describe an observation; no numerical consumer reads them.
# Do not exempt the whole JSON file, its checksum, or its numerical fields.
_DESCRIPTIVE_JSON_FIELDS = {
    "src/wd_spectra/data/validation/dah/re-j0317-853-vi.json": {
        "sigma", "v_over_i_level",
    },
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def descriptive_json_change(path, before, after):
    """Fail closed unless only known description strings differ."""
    fields = _DESCRIPTIVE_JSON_FIELDS.get(path)
    if fields is None or before is None or after is None:
        return False
    try:
        old, new = (
            json.loads(blob, object_pairs_hook=_unique_object,
                       parse_constant=_invalid_constant)
            for blob in (before, after)
        )
        if not isinstance(old, dict) or not isinstance(new, dict) or old.keys() != new.keys():
            return False
        if any(not isinstance(obj.get(key), str) for obj in (old, new) for key in fields):
            return False
        # Canonical JSON retains numeric types (1 differs from 1.0, and true
        # differs from 1) while ignoring whitespace and object-key ordering.
        other = [json.dumps({k: v for k, v in obj.items() if k not in fields},
                            sort_keys=True, allow_nan=False) for obj in (old, new)]
        return other[0] == other[1]
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return False


def requires_full(paths, *, contents=None):
    for path in paths:
        p = PurePosixPath(path)
        if path in {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore"}:
            continue
        if p.parts[0] == "docs":
            continue
        # The selection planner is tested by the ordinary CI policy tests;
        # changing it does not change a model equation or numerical checker.
        if path == "tools/validation_policy.py":
            continue
        if contents is not None and path in contents:
            if descriptive_json_change(path, *contents[path]):
                continue
        if (
            p.parts[0] == "tests"
            and p.suffix == ".py"
            and p.name
            not in {
                "conftest.py",
                "test_protected_model_canaries.py",
                "test_spectral_regressions.py",
                "test_daz_regressions.py",
                "test_dq_spectral_regression.py",
                "test_dq_release.py",
                "test_hot_cold_canary.py",
                "test_released_family_canaries.py",
                "test_dah_paper.py",
                "test_radiative_da_convergence.py",
                "test_d6_regression.py",
                "d6_regression_support.py",
            }
        ):
            continue
        if p.parts[0] == "research" and p.name.startswith("test_"):
            continue
        if path == "examples/generate_spectrum.ipynb":
            continue
        return True
    return False


def plan(event_name, event, paths, *, contents=None):
    if event_name in {"workflow_dispatch", "schedule"}:
        return True, True
    if event_name != "pull_request":
        raise ValueError(f"unsupported qualification event: {event_name}")
    pr = event["pull_request"]
    required = requires_full(paths, contents=contents)
    requested = not pr["draft"] and any(
        label["name"] == "full-validation" for label in pr["labels"]
    )
    return required, requested


def selected_cold_cases(event_name, cases):
    """Only PRs replace J1637 cold convergence with its warm regression."""
    return [case for case in cases if event_name != "pull_request" or case != "d6-j1637"]


def changed_inputs(base, head):
    """Read the same merge-base diff for names and raw provenance blobs."""
    merge_base = subprocess.check_output(["git", "merge-base", base, head]).decode().strip()
    paths = subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", "-z", merge_base, head]
    ).decode().strip("\0").split("\0")
    paths = [path for path in paths if path]

    def blob(ref, path):
        try:
            return subprocess.check_output(
                ["git", "cat-file", "blob", ref + ":" + path],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return None

    contents = {
        path: (blob(merge_base, path), blob(head, path))
        for path in paths if path in _DESCRIPTIVE_JSON_FIELDS
    }
    return paths, contents


def main():
    from pathlib import Path

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    paths = []
    contents = {}
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
        pr = event["pull_request"]
        # Include both old and new names for renames. Failure must not certify
        # an empty diff. Never interpolate PR-controlled strings into a shell.
        paths, contents = changed_inputs(pr["base"]["sha"], pr["head"]["sha"])
    required, requested = plan(os.environ["GITHUB_EVENT_NAME"], event, paths, contents=contents)
    from validate import COLD

    # PRs use J1637's bounded warm regression. Scheduled/manual runs retain
    # its full cold start; neither mode borrows a previous qualification pass.
    cold_cases = selected_cold_cases(os.environ["GITHUB_EVENT_NAME"], COLD)
    print(json.dumps(dict(required=required, requested=requested, changed=paths,
                         cold_cases=cold_cases)))
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"required={str(required).lower()}\n")
        stream.write(f"requested={str(requested).lower()}\n")
        stream.write("cold_cases=" + json.dumps(cold_cases) + "\n")


if __name__ == "__main__":
    main()
