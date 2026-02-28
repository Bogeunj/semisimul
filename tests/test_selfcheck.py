from __future__ import annotations

from proc2d.selfcheck import run_selfcheck


def test_selfcheck_imports_pass() -> None:
    report = run_selfcheck(smoke=False)
    assert report.ok
    assert report.rows
