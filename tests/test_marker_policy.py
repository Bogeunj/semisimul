from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_all_test_files_define_explicit_marker() -> None:
    tests_root = Path(__file__).resolve().parent
    repo_root = tests_root.parent
    missing: list[str] = []

    for path in sorted(tests_root.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark" in text or "@pytest.mark." in text:
            continue
        missing.append(str(path.relative_to(repo_root)))

    assert not missing, "Missing explicit pytest marker in: " + ", ".join(missing)
