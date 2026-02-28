"""Shared error types for proc2d orchestration layers."""

from __future__ import annotations


class DeckError(ValueError):
    """Raised when a deck is invalid or execution fails."""
