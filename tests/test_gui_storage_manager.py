"""Unit tests for GUI storage management helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from proc2d.app.gui.storage import GuiStorageManager


pytestmark = pytest.mark.unit


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_list_entries_and_total_size(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_bytes(root / "a.bin", 10)
    _write_bytes(root / "run1" / "b.bin", 7)
    _write_bytes(root / "run1" / "c.bin", 3)

    mgr = GuiStorageManager(root)
    entries = mgr.list_entries()
    rels = {entry.rel_path for entry in entries}
    assert "a.bin" in rels
    assert "run1" in rels
    assert mgr.get_total_size_bytes() == 20


def test_delete_rename_move(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_bytes(root / "run1" / "result.bin", 11)
    mgr = GuiStorageManager(root)

    new_rel = mgr.rename("run1/result.bin", "renamed.bin")
    assert new_rel == "run1/renamed.bin"
    assert (root / new_rel).exists()

    moved_rel = mgr.move(new_rel, "archive")
    assert moved_rel == "archive/renamed.bin"
    assert (root / moved_rel).exists()

    mgr.delete("archive")
    assert not (root / "archive").exists()


def test_rejects_unsafe_paths(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_bytes(root / "run1" / "result.bin", 5)
    mgr = GuiStorageManager(root)

    with pytest.raises(ValueError, match=r"outside root|invalid path"):
        mgr.delete("../run1/result.bin")
    with pytest.raises(ValueError, match=r"outside root|invalid path"):
        mgr.move("run1/result.bin", "../outside")
    with pytest.raises(ValueError, match=r"basename"):
        mgr.rename("run1/result.bin", "nested/new.bin")
