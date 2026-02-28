from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .deck import SimulationState, run_deck_data


@dataclass
class RunResult:
    state: SimulationState
    metrics: dict[str, Any]
    history: list[dict[str, float]]
    exports: dict[str, str]


class ArtifactSink(Protocol):
    def save_bytes(self, relpath: str, data: bytes) -> None: ...

    def save_text(self, relpath: str, text: str) -> None: ...

    def save_array(self, relpath: str, arr: np.ndarray) -> None: ...

    def make_subdir(self, relpath: str) -> "ArtifactSink": ...


@dataclass
class FileSystemSink:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, relpath: str, data: bytes) -> None:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_text(self, relpath: str, text: str) -> None:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def save_array(self, relpath: str, arr: np.ndarray) -> None:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def make_subdir(self, relpath: str) -> "FileSystemSink":
        return FileSystemSink(self.root / relpath)


@dataclass
class MemorySink:
    objects: dict[str, bytes] = field(default_factory=dict)
    prefix: str = ""

    def _full_path(self, relpath: str) -> str:
        if not self.prefix:
            return relpath
        return f"{self.prefix}/{relpath}".replace("//", "/")

    def save_bytes(self, relpath: str, data: bytes) -> None:
        self.objects[self._full_path(relpath)] = bytes(data)

    def save_text(self, relpath: str, text: str) -> None:
        self.save_bytes(relpath, text.encode("utf-8"))

    def save_array(self, relpath: str, arr: np.ndarray) -> None:
        from io import BytesIO

        buf = BytesIO()
        np.save(buf, arr)
        self.save_bytes(relpath, buf.getvalue())

    def make_subdir(self, relpath: str) -> "MemorySink":
        next_prefix = self._full_path(relpath)
        return MemorySink(objects=self.objects, prefix=next_prefix)


def run_deck_mapping(
    deck: dict[str, Any],
    *,
    base_dir: Path | None = None,
    out_override: Path | None = None,
    sink: ArtifactSink | None = None,
    hooks: Any | None = None,
) -> RunResult:
    del hooks

    base = Path.cwd() if base_dir is None else Path(base_dir)
    base = base.resolve()

    out_path: Path | None = Path(out_override).resolve() if out_override is not None else None
    if out_path is None and isinstance(sink, FileSystemSink):
        out_path = sink.root

    state = run_deck_data(
        deck,
        deck_path=base / "_inline_deck.yaml",
        out_override=out_path,
    )

    if sink is not None:
        for path in state.exports:
            if path.exists() and path.is_file():
                sink.save_bytes(path.name, path.read_bytes())

    exports = {path.name: str(path) for path in state.exports}
    return RunResult(
        state=state,
        metrics=dict(state.metrics or {}),
        history=list(state.history),
        exports=exports,
    )
