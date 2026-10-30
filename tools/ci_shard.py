"""Split CI work into balanced, disjoint shards whose union is everything.

Each shard runs on its own hosted worker, single-threaded, so wall-clock
time drops without parallelism inside any job. Weights are measured
seconds from CI logs; a file or case without a weight gets the median, so
new tests are always assigned and only the balance can go stale.

usage: python tools/ci_shard.py tests K N     (test files for shard K of N)
       python tools/ci_shard.py spectra K N   (--case arguments for validate.py)
"""

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(__file__).with_name("ci_shard_weights.json")


def test_files(root=ROOT):
    # pytest's default python_files under testpaths = ["tests"].
    tests = root / "tests"
    found = set(tests.rglob("test_*.py")) | set(tests.rglob("*_test.py"))
    return sorted(str(path.relative_to(root)) for path in found)


def spectra_cases():
    sys.path.insert(0, str(ROOT / "tools"))
    from validate import SPECTRA

    return list(SPECTRA)


def shards(items, weights, count):
    """Longest-processing-time-first assignment; deterministic for fixed inputs.

    Each shard lists its heaviest item first, so validate.py's two workers
    start the longest case immediately instead of leaving it for last.
    """
    if not 1 <= count <= len(items):
        raise ValueError(f"cannot split {len(items)} items into {count} shards")
    known = [weights[item] for item in items if item in weights]
    default = statistics.median(known) if known else 1.0
    lanes = [[0.0, index, []] for index in range(count)]
    for item in sorted(items, key=lambda item: (-weights.get(item, default), item)):
        lane = min(lanes, key=lambda lane: (lane[0], lane[1]))
        lane[0] += weights.get(item, default)
        lane[2].append(item)
    return [lane[2] for lane in lanes]


def main(argv=None):
    kind, shard, count = (argv or sys.argv[1:])
    shard, count = int(shard), int(count)
    if not 1 <= shard <= count:
        raise SystemExit(f"shard must be in 1..{count}")
    weights = json.loads(WEIGHTS.read_text())[kind]
    items = test_files() if kind == "tests" else spectra_cases()
    selected = shards(items, weights, count)[shard - 1]
    if not selected:
        raise SystemExit("empty shard")  # Never fall through to "run everything".
    if kind == "spectra":
        print(" ".join(f"--case {case}" for case in selected))
    else:
        print("\n".join(selected))


if __name__ == "__main__":
    main()
