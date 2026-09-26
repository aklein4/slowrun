"""Check local inline links and ATX heading anchors in maintained Markdown."""

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = (
    "AGENTS.md",
    "README.md",
    "agent_notes",
    "docs",
    "baselines/README.md",
    "agent_scripts/README.md",
    "agent_reports/README.md",
    "research/loss_budget/README.md",
)
LINK = re.compile(r"\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+\"[^\"\n]*\")?\s*\)")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def prose_lines(path):
    """Yield numbered lines outside fenced code, retaining source line numbers."""
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = FENCE.match(line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end() :].strip()
            ):
                fence = None
        elif marker:
            fence = marker[1]
        else:
            yield number, line


def heading_anchors(path):
    """Build GitHub-style anchors for the simple ATX headings used in notes."""
    anchors = set()
    for _, line in prose_lines(path):
        match = HEADING.match(line)
        if not match:
            continue
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
        base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchor = base
        suffix = 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{base}-{suffix}"
        anchors.add(anchor)
    return anchors


def check(paths):
    """Return diagnostics for missing local targets or Markdown heading anchors."""
    errors = []
    anchor_cache = {}
    for path in paths:
        for number, line in prose_lines(path):
            # Inline code can contain illustrative Markdown syntax.
            line = re.sub(r"(`+).*?\1", "", line)
            for match in LINK.finditer(line):
                destination = match[1].strip("<>")
                url = urlsplit(destination)
                if url.scheme or url.netloc:
                    continue
                target = (
                    (path.parent / unquote(url.path)).resolve() if url.path else path
                )
                error = None
                if not target.exists():
                    error = "missing target"
                elif (
                    url.fragment and target.is_file() and target.suffix.lower() == ".md"
                ):
                    if target not in anchor_cache:
                        anchor_cache[target] = heading_anchors(target)
                    if unquote(url.fragment) not in anchor_cache[target]:
                        error = "missing heading anchor"
                if error:
                    errors.append(f"{path}:{number}: {error}: {destination}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files/directories (default: maintained docs)",
    )
    args = parser.parse_args()
    inputs = args.paths or [ROOT / name for name in DEFAULTS]
    paths = set()
    for path in inputs:
        if not path.exists():
            parser.error(f"path does not exist: {path}")
        if path.is_dir():
            paths.update(
                item.resolve() for item in path.rglob("*.md") if item.is_file()
            )
        else:
            paths.add(path.resolve())
    if not paths:
        parser.error("no Markdown files selected")
    errors = check(sorted(paths))
    for error in errors:
        print(error)
    print(
        f"Checked {len(paths)} Markdown files; {len(errors)} local link/anchor errors."
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
