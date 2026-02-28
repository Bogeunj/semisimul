"""Application-layer CLI adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..deck import DeckError, run_deck


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(prog="proc2d", description="proc2d process simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a YAML process deck")
    run_p.add_argument("deck", type=str, help="Path to deck YAML")
    run_p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override export output directory",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            state = run_deck(args.deck, out_override=args.out)
        except DeckError as exc:
            parser.exit(2, f"Error: {exc}\n")

        print(f"Done. Grid={state.grid.shape}, exports={len(state.exports)}")
        if state.exports:
            outdirs = sorted({str(Path(p).parent) for p in state.exports})
            for outdir in outdirs:
                print(f"Output: {outdir}")
        return 0

    parser.exit(2, "Unknown command\n")
    return 2
