"""Application-layer CLI adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..deck import DeckError, run_deck
from ..selfcheck import run_selfcheck


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        prog="proc2d", description="proc2d process simulator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a YAML process deck")
    run_p.add_argument("deck", type=str, help="Path to deck YAML")
    run_p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override export output directory",
    )

    selfcheck_p = sub.add_parser(
        "selfcheck", help="Run dependency and smoke self-check"
    )
    selfcheck_p.add_argument(
        "--no-smoke",
        action="store_true",
        help="Run import checks only (skip smoke simulation).",
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

    if args.command == "selfcheck":
        report = run_selfcheck(smoke=not bool(args.no_smoke))
        print(report.to_text())
        return 0 if report.ok else 1

    parser.exit(2, "Unknown command\n")
    return 2
