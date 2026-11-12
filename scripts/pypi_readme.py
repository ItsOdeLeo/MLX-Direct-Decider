"""Rewrite README.md relative links to absolute GitHub URLs for the PyPI page.

PyPI renders the long description without the repository, so relative image and
document links break there. The release workflow runs this script on a fresh
checkout before building. The committed README keeps its relative links.

    python scripts/pypi_readme.py --ref v0.2.0 [--path README.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = "ItsOdeLeo/MLX-Direct-Decider"
SKIP = ("http://", "https://", "#", "mailto:")

MD_IMAGE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)")
MD_LINK = re.compile(r"((?<!!)\[[^\]]*\]\()([^)\s]+)")
HTML_ATTR = re.compile(r'\b(src|href)="([^"]+)"')


def absolute(target: str, ref: str, raw: bool) -> str:
    if target.startswith(SKIP):
        return target
    path = target.removeprefix("./")
    if raw:
        return f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"
    return f"https://github.com/{REPO}/blob/{ref}/{path}"


def rewrite(text: str, ref: str) -> str:
    text = MD_IMAGE.sub(lambda m: m.group(1) + absolute(m.group(2), ref, True), text)
    text = MD_LINK.sub(lambda m: m.group(1) + absolute(m.group(2), ref, False), text)
    return HTML_ATTR.sub(
        lambda m: f'{m.group(1)}="{absolute(m.group(2), ref, m.group(1) == "src")}"', text
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="git tag or commit the links point to")
    parser.add_argument("--path", default="README.md", type=Path)
    args = parser.parse_args()
    args.path.write_text(rewrite(args.path.read_text(encoding="utf-8"), args.ref),
                         encoding="utf-8")


if __name__ == "__main__":
    main()
