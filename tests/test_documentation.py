"""Keep the user guide navigable as research records are reorganized."""

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md"]
NOTEBOOK = ROOT / "examples/generate_spectrum.ipynb"


def markdown(path):
    if path.suffix == ".ipynb":
        return "\n".join(
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "markdown"
        )
    return path.read_text()


def local_links(path):
    for target in re.findall(r"\]\(([^)\n]+)\)", markdown(path)):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("/"):
            continue
        resolved = (
            (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        )
        yield resolved, unquote(parsed.fragment)


def heading_anchors(path):
    anchors = set()
    counts = {}
    fenced = False
    for line in markdown(path).splitlines():
        if line.startswith(("```", "~~~")):
            fenced = not fenced
        if fenced or not re.match(r"^#{1,6} ", line):
            continue
        heading = re.sub(r"^#+\s+", "", line).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(slug if not count else f"{slug}-{count}")
        counts[slug] = count + 1
    return anchors


def test_local_documentation_links_and_headings_resolve():
    problems = []
    for source in PAGES + [NOTEBOOK]:
        for target, fragment in local_links(source):
            if not target.exists():
                problems.append(f"{source.relative_to(ROOT)} -> missing {target}")
            elif fragment and target.suffix == ".md":
                if fragment not in heading_anchors(target):
                    problems.append(
                        f"{source.relative_to(ROOT)} -> missing {target}#{fragment}"
                    )
    assert not problems, "\n".join(problems)


def test_every_documentation_page_is_reachable_from_the_home_page():
    seen = set()
    queue = [DOCS / "README.md"]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        queue.extend(
            target
            for target, _ in local_links(path)
            if target.is_file()
            and target.suffix == ".md"
            and DOCS in target.parents
            and target not in seen
        )
    assert not set(DOCS.rglob("*.md")) - seen
