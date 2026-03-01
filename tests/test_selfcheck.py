from __future__ import annotations

import pytest

from proc2d.selfcheck import run_selfcheck


pytestmark = pytest.mark.unit


def test_selfcheck_imports_pass() -> None:
    report = run_selfcheck(smoke=False)
    assert report.ok
    assert report.rows
