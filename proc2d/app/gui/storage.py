"""Storage management helpers for GUI artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageEntry:
    """Represents one storage entry relative to the managed root."""

    rel_path: str
    entry_type: str
    size_bytes: int


class GuiStorageManager:
    """Safe file operations scoped to one root directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_relative(self, rel_path: str) -> Path:
        rel_text = str(rel_path).strip()
        rel = Path(rel_text)
        if rel.is_absolute() or str(rel) in {"", "."}:
            raise ValueError("invalid path")
        if any(part == ".." for part in rel.parts):
            raise ValueError("outside root")
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("outside root") from exc
        return target

    @staticmethod
    def _compute_size_bytes(path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            return sum(int(p.stat().st_size) for p in path.rglob("*") if p.is_file())
        return 0

    def list_entries(self) -> list[StorageEntry]:
        if not self.root.exists():
            return []

        entries: list[StorageEntry] = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            rel_path = path.relative_to(self.root).as_posix()
            entry_type = "dir" if path.is_dir() else "file"
            entries.append(
                StorageEntry(
                    rel_path=rel_path,
                    entry_type=entry_type,
                    size_bytes=self._compute_size_bytes(path),
                )
            )
        return entries

    def get_total_size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(
            int(path.stat().st_size) for path in self.root.rglob("*") if path.is_file()
        )

    def delete(self, rel_path: str) -> None:
        target = self._resolve_relative(rel_path)
        if target == self.root or not target.exists():
            raise ValueError("invalid path")
        if target.is_dir():
            shutil.rmtree(target)
            return
        target.unlink()

    def rename(self, rel_path: str, new_basename: str) -> str:
        src = self._resolve_relative(rel_path)
        if not src.exists():
            raise ValueError("invalid path")
        if Path(new_basename).name != new_basename or new_basename in {"", ".", ".."}:
            raise ValueError("basename")

        dst = src.with_name(new_basename)
        if dst.exists():
            raise ValueError("destination exists")
        src.rename(dst)
        return dst.relative_to(self.root).as_posix()

    def move(self, rel_path: str, destination_dir: str) -> str:
        src = self._resolve_relative(rel_path)
        if not src.exists():
            raise ValueError("invalid path")

        dst_text = str(destination_dir).strip()
        dst_dir = (
            self.root if dst_text in {"", "."} else self._resolve_relative(dst_text)
        )
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not dst_dir.is_dir():
            raise ValueError("invalid path")

        dst = dst_dir / src.name
        if dst.exists():
            raise ValueError("destination exists")

        src.rename(dst)
        return dst.relative_to(self.root).as_posix()


__all__ = ["GuiStorageManager", "StorageEntry"]
