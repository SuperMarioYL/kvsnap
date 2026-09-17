"""quality-version-lockstep-test — no future bump can land half-done.

The portfolio's most-repeated iteration defect is version drift (one surface
bumped, the others stale). This test pins every surface of THIS repo to the
same version: VERSION, kvsnap.config.VERSION, pyproject.toml, and (when the
site exists) web/site.json meta.content_version.
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


def test_version_file_matches_config():
    assert _norm((ROOT / "VERSION").read_text()) == _norm(config.VERSION)


def test_pyproject_version_matches_config():
    match = re.search(
        r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M
    )
    assert match, "no version field in pyproject.toml"
    assert _norm(match.group(1)) == _norm(config.VERSION)


def test_site_json_content_version_matches_config():
    site = ROOT / "web" / "site.json"
    if not site.exists():
        pytest.skip("no web/site.json in this checkout")
    data = json.loads(site.read_text())
    assert _norm(data["meta"]["content_version"]) == _norm(config.VERSION)
