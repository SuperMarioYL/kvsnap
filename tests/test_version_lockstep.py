"""quality-version-lockstep-test — no future bump can land half-done.

The portfolio's most-repeated iteration defect is version drift (one surface
bumped, the others stale). v0.2.0 shipped exactly that: the guard then pinned
only VERSION / kvsnap.config.VERSION / pyproject.toml / site.json
``content_version``, so it passed while the README badge (both languages), the
site footer tag and the recorded demo version stayed at v0.1.0.

Every surface a release bump must touch is asserted here, so a half-done bump
fails the suite instead of shipping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kvsnap import config

ROOT = Path(__file__).resolve().parents[1]


def _norm(version: str) -> str:
    return version.strip().lstrip("v")


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"no {rel} in this checkout")
    return path.read_text(encoding="utf-8")


def _site() -> dict:
    return json.loads(_read("web/site.json"))


def _readme_badge(rel: str) -> str:
    match = re.search(r"^`v(\d+\.\d+\.\d+)`", _read(rel), re.M)
    assert match, f"no `vX.Y.Z` version line in {rel}"
    return match.group(1)


#: surface name -> how to read the version it advertises
SURFACES = {
    "VERSION": lambda: _read("VERSION"),
    "kvsnap.config.VERSION": lambda: config.VERSION,
    "pyproject.toml": lambda: re.search(
        r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.M
    ).group(1),
    "web/site.json content_version": lambda: _site()["meta"]["content_version"],
    "web/site.json footer.tag": lambda: _site()["footer"]["tag"].split("·", 1)[0],
    "docs/demo-results.json": lambda: json.loads(_read("docs/demo-results.json"))[
        "version"
    ],
    "README.md": lambda: _readme_badge("README.md"),
    "README.en.md": lambda: _readme_badge("README.en.md"),
    "CHANGELOG.md head": lambda: re.search(
        r"^## \[(\d+\.\d+\.\d+)\]", _read("CHANGELOG.md"), re.M
    ).group(1),
}


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_surface_matches_config(surface):
    assert _norm(SURFACES[surface]()) == _norm(config.VERSION)
